#include <cuda_fp16.h>
#include "lora.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define LORA_A_THREADS 128
#define LORA_B_THREADS 256

// One block per (rank row, token): fp32 dot product over K, half2 loads, block-reduced. A is
// read once per token; at decode shapes (a handful of rows, rank <= a few hundred) the grid is
// small enough that the launch, not the arithmetic, is the cost

__launch_bounds__(LORA_A_THREADS)
__global__ void lora_a_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ a,
    float* __restrict__ t,
    const int K,
    const int a_stride,
    const int R
)
{
    const half* xr = x + (size_t) blockIdx.y * K;
    const half* ar = a + (size_t) blockIdx.x * a_stride;

    float acc = 0.0f;
    if ((K & 1) == 0 && (a_stride & 1) == 0)
    {
        // Even widths keep every row 4-byte aligned for half2 loads
        const half2* xr2 = (const half2*) xr;
        const half2* ar2 = (const half2*) ar;
        for (int k = threadIdx.x; k < (K >> 1); k += LORA_A_THREADS)
        {
            float2 xv = __half22float2(xr2[k]);
            float2 av = __half22float2(ar2[k]);
            acc += xv.x * av.x + xv.y * av.y;
        }
    }
    else
    {
        for (int k = threadIdx.x; k < K; k += LORA_A_THREADS)
            acc += __half2float(xr[k]) * __half2float(ar[k]);
    }

    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);

    __shared__ float warp_sums[LORA_A_THREADS / 32];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
        for (int w = 0; w < LORA_A_THREADS / 32; ++w) sum += warp_sums[w];
        t[(size_t) blockIdx.y * R + blockIdx.x] = sum;
    }
}

// One thread per output column: fp32 dot product over the rank, accumulated onto y in place. The
// t row is a broadcast read, b is coalesced across the block

#define KERNEL_DEF(yt, kernel, load, store) \
__launch_bounds__(LORA_B_THREADS) \
__global__ void kernel \
( \
    const float* __restrict__ t, \
    const half* __restrict__ b, \
    yt* __restrict__ y, \
    const int R, \
    const int t_stride, \
    const int b_stride, \
    const int N \
) \
{ \
    int n = blockIdx.x * LORA_B_THREADS + threadIdx.x; \
    if (n >= N) return; \
    const float* tr = t + (size_t) blockIdx.y * t_stride; \
    const half* bc = b + n; \
    float acc = 0.0f; \
    for (int j = 0; j < R; ++j) \
        acc += tr[j] * __half2float(bc[(size_t) j * b_stride]); \
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
    const at::Tensor& a,
    at::Tensor& t,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(t, kFloat);
    TORCH_CHECK_DIM(a, 2);
    TORCH_CHECK_DIM(t, 2);
    TORCH_CHECK(x.is_contiguous() && a.is_contiguous() && t.is_contiguous(), "lora_a: tensors must be contiguous");

    int K = (int) x.size(-1);
    int m = (int) (x.numel() / K);
    int R = (int) a.size(0);
    TORCH_CHECK(a.size(1) >= K, "lora_a: A is narrower than x");
    TORCH_CHECK(t.size(0) >= m && t.size(1) == R, "lora_a: t has the wrong shape");
    if (!m) return;

    dim3 grid(R, m);
    lora_a_kernel<<<grid, LORA_A_THREADS, 0, stream>>>
    (
        (const half*) x.data_ptr(),
        (const half*) a.data_ptr(),
        (float*) t.data_ptr(),
        K,
        (int) a.size(1),
        R
    );
    if (graph) graph->record_param((void*) &lora_a_kernel, GP_lora_x, 0);
    if (graph) graph->record_param((void*) &lora_a_kernel, GP_end, 0);
    cuda_check(cudaPeekAtLastError());
}

void lora_b_gr
(
    const at::Tensor& t,
    const at::Tensor& b,
    at::Tensor& y,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(t, kFloat);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_FLOAT_HALF(y);
    TORCH_CHECK_DIM(t, 2);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(t.is_contiguous() && b.is_contiguous() && y.is_contiguous(), "lora_b: tensors must be contiguous");

    int R = (int) b.size(0);
    int N = (int) y.size(-1);
    int m = (int) (y.numel() / N);
    TORCH_CHECK(b.size(1) >= N, "lora_b: B is narrower than y");
    TORCH_CHECK(t.size(0) >= m && t.size(1) == R, "lora_b: t has the wrong shape");
    if (!m) return;

    dim3 grid(CEIL_DIVIDE(N, LORA_B_THREADS), m);
    int t_stride = (int) t.size(1);
    int b_stride = (int) b.size(1);

    #define INSTANCE(yt_, yt__, kernel) \
    if (y.dtype() == yt_) \
    { \
        kernel<<<grid, LORA_B_THREADS, 0, stream>>> \
            ((const float*) t.data_ptr(), (const half*) b.data_ptr(), (yt__*) y.data_ptr(), R, t_stride, b_stride, N); \
        if (graph) graph->record_param((void*) &kernel, GP_lora_y, 2); \
        if (graph) graph->record_param((void*) &kernel, GP_end, 0); \
    }

    INSTANCE(at::kHalf,  half,  lora_b_kernel_h)
    INSTANCE(at::kFloat, float, lora_b_kernel_f)

    #undef INSTANCE

    cuda_check(cudaPeekAtLastError());
}
