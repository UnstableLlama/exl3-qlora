#include <Python.h>
#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../add.cuh"
#include "../lora.cuh"

void BC_LinearFP16::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.dtype() == y.dtype() && !graph)
        at::matmul_out(y, x, weight);
    else
        hgemm_gr(x, weight, y, graph);

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

//void BC_LinearFP16::run_cublas(const at::Tensor& x, at::Tensor& y)
//{
//    hgemm(x, weight, y);
//    if (bias)
//        y.add_(bias.value());
//}

void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.numel() == x.size(-1))
    {
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
    }
    else
    {
        TORCH_CHECK(!graph || graph->disabled, "BC_LinearEXL3 invoked with graph and bsz > 1");
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

at::Tensor BC_LinearEXL3::run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );
    if (out_features == 0) return y;

    at::Tensor x_flat = x.view({-1, x.size(-1)});
    at::Tensor y_flat = y.view({-1, out_features});
    run(x_flat, y_flat);
    return y;
}

void BC_LoRA::set(c10::optional<at::Tensor> a_, c10::optional<at::Tensor> b_, int64_t K, int64_t N, const at::Device& device)
{
    TORCH_CHECK(a_.has_value() == b_.has_value(), "BC_LoRA::set: A and B must be given together");
    if (!a_ && !a) return;

    if (a_)
    {
        const at::Tensor& av = a_.value();
        const at::Tensor& bv = b_.value();
        TORCH_CHECK_DTYPE(av, kHalf);
        TORCH_CHECK_DTYPE(bv, kHalf);
        TORCH_CHECK_DIM(av, 2);
        TORCH_CHECK_DIM(bv, 2);
        TORCH_CHECK(av.is_contiguous() && bv.is_contiguous(), "BC_LoRA::set: A and B must be contiguous");
        TORCH_CHECK(av.device() == device && bv.device() == device, "BC_LoRA::set: A and B must be on the weight's device");
        TORCH_CHECK(av.size(0) == bv.size(0) && av.size(0) > 0, "BC_LoRA::set: A (R, K) and B (R, N) ranks differ");
        TORCH_CHECK(av.size(1) == K, "BC_LoRA::set: A width must be the linear's (padded) in_features");
        TORCH_CHECK(bv.size(1) == N, "BC_LoRA::set: B width must be the linear's (padded) out_features");
        t = at::empty({MAX_ROWS, av.size(0)}, av.options().dtype(at::kFloat));
    }
    else
        t = at::Tensor();

    a = std::move(a_);
    b = std::move(b_);
    Graph::bump_structure_epoch();
}

void BC_LoRA::gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (!a) return;
    TORCH_CHECK(x.numel() / x.size(-1) <= MAX_ROWS, "BC_LoRA::gr: too many rows for the LoRA scratch");
    lora_a_gr(x, a.value(), t, graph);
    lora_b_gr(t, b.value(), y, graph);
}

void BC_LoRA::args(std::vector<PPTR>& params, void* x, void* y) const
{
    if (!a) return;
    params.emplace_back(GP_lora_x, x);
    params.emplace_back(GP_lora_y, y);
}
