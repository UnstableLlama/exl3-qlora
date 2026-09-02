#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include "lora.cuh"
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"

namespace cg = cooperative_groups;

#define LORA_THREADS 256

// Persistent blocks, three phases separated by grid.sync():
//   1. block b owns K-slice b: partial[b, m, r] = sum_{k in slice} x[m, k] * A[k, r]
//      (threads stride over (m, r); consecutive r read consecutive A addresses)
//   2. t[m, r] = sum_b partial[b, m, r]  (grid-striped over (m, r))
//   3. block b owns N-stripe b: C[m, n] += sum_r t[m, r] * B[r, n]
//      (threads stride over n; consecutive n read consecutive B addresses, t broadcasts)
// The workspace holds [gridDim, M, R] partials followed by [M, R] for t.

template <bool c_fp32>
__global__ __launch_bounds__(LORA_THREADS)
void lora_gemv_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ A,
    const half* __restrict__ B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int rank,
    float* ws
)
{
    auto grid = cg::this_grid();
    const int nb = gridDim.x;
    const int b = blockIdx.x;
    const int MR = size_m * rank;
    float* partial = ws;
    float* t = ws + (size_t) nb * MR;

    // Phase 1
    {
        int k_per = (size_k + nb - 1) / nb;
        int k0 = b * k_per;
        int k1 = min(size_k, k0 + k_per);
        for (int i = threadIdx.x; i < MR; i += LORA_THREADS)
        {
            int m = i / rank;
            int r = i - m * rank;
            const half* xr = x + (size_t) m * size_k;
            const half* ar = A + r;
            float acc = 0.0f;
            for (int k = k0; k < k1; ++k)
                acc += __half2float(xr[k]) * __half2float(ar[(size_t) k * rank]);
            partial[(size_t) b * MR + i] = acc;
        }
    }
    grid.sync();

    // Phase 2
    for (int i = b * LORA_THREADS + threadIdx.x; i < MR; i += nb * LORA_THREADS)
    {
        float acc = 0.0f;
        for (int j = 0; j < nb; ++j)
            acc += partial[(size_t) j * MR + i];
        t[i] = acc;
    }
    grid.sync();

    // Phase 3
    {
        int n_per = (size_n + nb - 1) / nb;
        int n0 = b * n_per;
        int n1 = min(size_n, n0 + n_per);
        for (int n = n0 + threadIdx.x; n < n1; n += LORA_THREADS)
        {
            for (int m = 0; m < size_m; ++m)
            {
                const float* tm = t + (size_t) m * rank;
                float acc = 0.0f;
                for (int r = 0; r < rank; ++r)
                    acc += tm[r] * __half2float(B[(size_t) r * size_n + n]);
                size_t ci = (size_t) m * size_n + n;
                if constexpr (c_fp32)
                {
                    ((float*) C)[ci] += acc;
                }
                else
                {
                    half* ch = (half*) C;
                    ch[ci] = __float2half(__half2float(ch[ci]) + acc);
                }
            }
        }
    }
}

// One workspace per device, sized once for the largest supported problem so its address never
// changes after a graph baked it (allocated through the caching allocator and held here for the
// process lifetime)
static std::map<int, at::Tensor> g_lora_ws;

static float* lora_workspace(int device, int num_sms)
{
    at::Tensor& t = g_lora_ws[device];
    if (!t.defined())
    {
        int64_t floats = (int64_t) (num_sms + 1) * LORA_MAX_M * LORA_MAX_RANK;
        t = at::empty({floats}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kFloat));
    }
    return t.data_ptr<float>();
}

void lora_gemv_gr
(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& c,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    bool c_fp32 = c.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(c, kHalf);
    TORCH_CHECK(x.is_contiguous() && a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "lora_gemv: all operands must be contiguous");
    TORCH_CHECK_DIM(a, 2);
    TORCH_CHECK_DIM(b, 2);

    int size_k = (int) x.size(-1);
    int size_m = (int) (x.numel() / size_k);
    int size_n = (int) c.size(-1);
    int rank = (int) a.size(1);
    TORCH_CHECK(a.size(0) == size_k, "lora_gemv: A rows must equal x width");
    TORCH_CHECK(b.size(0) == rank, "lora_gemv: B rows must equal A cols (rank)");
    TORCH_CHECK(b.size(1) == size_n, "lora_gemv: B cols must equal C width");
    TORCH_CHECK(c.numel() / size_n == size_m, "lora_gemv: C rows must equal x rows");
    TORCH_CHECK(size_m >= 1 && size_m <= LORA_MAX_M, "lora_gemv: M out of range");
    TORCH_CHECK(rank >= 1 && rank <= LORA_MAX_RANK, "lora_gemv: rank out of range");

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    float* ws = lora_workspace(device, num_sms);

    const half* x_ptr = (const half*) x.data_ptr();
    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();
    void* c_ptr = (void*) c.data_ptr();

    void* kernelArgs[] =
    {
        (void*)& x_ptr,
        (void*)& a_ptr,
        (void*)& b_ptr,
        (void*)& c_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& rank,
        (void*)& ws
    };

    void* kernel = c_fp32 ? (void*) lora_gemv_kernel<true> : (void*) lora_gemv_kernel<false>;
    cuda_check(cudaLaunchCooperativeKernel(kernel, num_sms, LORA_THREADS, kernelArgs, 0, stream));

    if (graph)
    {
        graph->record_param(kernel, GP_lora_x, 0);
        graph->record_param(kernel, GP_lora_A, 1);
        graph->record_param(kernel, GP_lora_B, 2);
        graph->record_param(kernel, GP_lora_C, 3);
        graph->record_param(kernel, GP_lora_rank, 7, 4);
        graph->record_param(kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

void lora_gemv
(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& c
)
{
    lora_gemv_gr(x, a, b, c, nullptr);
}
