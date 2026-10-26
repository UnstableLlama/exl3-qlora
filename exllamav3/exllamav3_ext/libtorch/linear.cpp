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

void BC_LinearEXL3::set_lora(c10::optional<at::Tensor> a, c10::optional<at::Tensor> b)
{
    TORCH_CHECK(a.has_value() == b.has_value(), "BC_LinearEXL3::set_lora: A and B must be given together");
    if (!a && !lora_a) return;

    if (a)
    {
        const at::Tensor& a_ = a.value();
        const at::Tensor& b_ = b.value();
        TORCH_CHECK_DTYPE(a_, kHalf);
        TORCH_CHECK_DTYPE(b_, kHalf);
        TORCH_CHECK_DIM(a_, 2);
        TORCH_CHECK_DIM(b_, 2);
        TORCH_CHECK(a_.is_contiguous() && b_.is_contiguous(), "BC_LinearEXL3::set_lora: A and B must be contiguous");
        TORCH_CHECK(a_.device() == trellis.device() && b_.device() == trellis.device(), "BC_LinearEXL3::set_lora: A and B must be on the weight's device");
        TORCH_CHECK(a_.size(0) == b_.size(0) && a_.size(0) > 0, "BC_LinearEXL3::set_lora: A (R, K) and B (R, N) ranks differ");
        TORCH_CHECK(a_.size(1) == trellis.size(0) * 16, "BC_LinearEXL3::set_lora: A width must be the padded in_features");
        TORCH_CHECK(b_.size(1) == trellis.size(1) * 16, "BC_LinearEXL3::set_lora: B width must be the padded out_features");
        lora_t = at::empty({LORA_MAX_ROWS, a_.size(0)}, a_.options().dtype(at::kFloat));
    }
    else
        lora_t = at::Tensor();

    lora_a = std::move(a);
    lora_b = std::move(b);
    Graph::bump_structure_epoch();
}

void BC_LinearEXL3::lora_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (!lora_a) return;
    TORCH_CHECK(x.numel() / x.size(-1) <= LORA_MAX_ROWS, "BC_LinearEXL3::lora_gr: too many rows for the LoRA scratch");
    lora_a_gr(x, lora_a.value(), lora_t, graph);
    lora_b_gr(lora_t, lora_b.value(), y, graph);
}

void BC_LinearEXL3::run_lora(const at::Tensor& x, at::Tensor& y)
{
    lora_gr(x, y, nullptr);
}

void BC_LinearEXL3::lora_args(std::vector<PPTR>& params, void* x, void* y) const
{
    if (!lora_a) return;
    params.emplace_back(GP_lora_x, x);
    params.emplace_back(GP_lora_y, y);
}
