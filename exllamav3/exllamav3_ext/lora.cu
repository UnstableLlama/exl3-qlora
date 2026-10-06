#include <cuda_fp16.h>
#include "lora.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define LORA_THREADS 256

// One block per (rank column, row): fp32 dot product of length K, block-reduced

__launch_bounds__(LORA_THREADS)
__global__ void lora_a_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ a_t,
    float* __restrict__ t,
    const int K,
    const int a_stride,
    const int R
)
{
    const half* xr = x + (size_t) blockIdx.y * K;
    const half* ar = a_t + (size_t) blockIdx.x * a_stride;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += LORA_THREADS)
        acc += __half2float(xr[k]) * __half2float(ar[k]);

    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);

    __shared__ float warp_sums[LORA_THREADS / 32];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
        for (int w = 0; w < LORA_THREADS / 32; ++w) sum += warp_sums[w];
        t[(size_t) blockIdx.y * R + blockIdx.x] = sum;
    }
}

// One thread per output element: fp32 dot product of length R, accumulated onto y

#define KERNEL_DEF(yt, kernel, load, store) \
__launch_bounds__(LORA_THREADS) \
__global__ void kernel \
( \
    const float* __restrict__ t, \
    const half* __restrict__ b, \
    yt* __restrict__ y, \
    const int R, \
    const int t_stride, \
    const int N \
) \
{ \
    int n = blockIdx.x * LORA_THREADS + threadIdx.x; \
    if (n >= N) return; \
    const float* tr = t + (size_t) blockIdx.y * t_stride; \
    float acc = 0.0f; \
    for (int j = 0; j < R; ++j) \
        acc += tr[j] * __half2float(b[(size_t) j * N + n]); \
    yt* yp = y + (size_t) blockIdx.y * N + n; \
    float v = load + acc; \
    *yp = store; \
}

KERNEL_DEF(half,  lora_b_kernel_h, __half2float(*yp), __float2half_rn(v))
KERNEL_DEF(float, lora_b_kernel_f, *yp,               v)

#undef KERNEL_DEF

void lora_a_gr
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    at::Tensor& t,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(a_t, kHalf);
    TORCH_CHECK_DTYPE(t, kFloat);
    TORCH_CHECK_DIM(a_t, 2);
    TORCH_CHECK_DIM(t, 2);
    TORCH_CHECK(x.is_contiguous() && a_t.is_contiguous() && t.is_contiguous(), "lora_a: tensors must be contiguous");

    int K = (int) x.size(-1);
    int m = (int) (x.numel() / K);
    int R = (int) a_t.size(0);
    TORCH_CHECK(a_t.size(1) >= K, "lora_a: A is narrower than x");
    TORCH_CHECK(t.size(0) >= m && t.size(1) == R, "lora_a: t has the wrong shape");

    dim3 grid(R, m);
    lora_a_kernel<<<grid, LORA_THREADS, 0, stream>>>
    (
        (const half*) x.data_ptr(),
        (const half*) a_t.data_ptr(),
        (float*) t.data_ptr(),
        K,
        (int) a_t.size(1),
        R
    );
    if (graph) graph->record_param((void*) &lora_a_kernel, GP_lora_x, 0);
    if (graph) graph->record_param((void*) &lora_a_kernel, GP_end, 0);
    cuda_check(cudaPeekAtLastError());
}

void lora_b_gr
(
    const at::Tensor& t,
    int t_offset,
    const at::Tensor& b,
    at::Tensor& y,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(t, kFloat);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(t, 2);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(t.is_contiguous() && b.is_contiguous() && y.is_contiguous(), "lora_b: tensors must be contiguous");

    int R = (int) b.size(0);
    int N = (int) b.size(1);
    TORCH_CHECK(y.size(-1) == N, "lora_b: B and y widths differ");
    int m = (int) (y.numel() / N);
    TORCH_CHECK(t.size(0) >= m && t_offset >= 0 && t_offset + R <= t.size(1), "lora_b: t has the wrong shape");

    dim3 grid(CEIL_DIVIDE(N, LORA_THREADS), m);
    const float* t_ptr = ((const float*) t.data_ptr()) + t_offset;
    int t_stride = (int) t.size(1);

    #define INSTANCE(yt_, yt__, kernel) \
    if (y.dtype() == yt_) \
    { \
        kernel<<<grid, LORA_THREADS, 0, stream>>>(t_ptr, (const half*) b.data_ptr(), (yt__*) y.data_ptr(), R, t_stride, N); \
        if (graph) graph->record_param((void*) &kernel, GP_lora_y, 2); \
        if (graph) graph->record_param((void*) &kernel, GP_end, 0); \
    }

    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kFloat, "lora_b: y must be float16 or float32");
    INSTANCE(at::kHalf,  half,  lora_b_kernel_h)
    INSTANCE(at::kFloat, float, lora_b_kernel_f)

    #undef INSTANCE

    cuda_check(cudaPeekAtLastError());
}
