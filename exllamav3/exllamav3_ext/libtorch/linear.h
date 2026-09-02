#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../graph.cuh"

struct BC_LinearFP16
{
    at::Tensor weight;
    c10::optional<at::Tensor> bias;

    BC_LinearFP16
    (
        at::Tensor _weight,
        c10::optional<at::Tensor> _bias
    ) :
        weight(std::move(_weight)),
        bias(std::move(_bias))
    {}

    void run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph);
    void run(const at::Tensor& x, at::Tensor& y);
    // void run_cublas(const at::Tensor& x, at::Tensor& y);
};

struct BC_LinearEXL3
{
    at::Tensor trellis;
    at::Tensor suh;
    at::Tensor svh;
    int K;
    c10::optional<at::Tensor> bias;
    bool mcg;
    bool mul1;
    at::Tensor xh;

    BC_LinearEXL3
    (
        at::Tensor _trellis,
        at::Tensor _suh,
        at::Tensor _svh,
        int _K,
        c10::optional<at::Tensor> _bias,
        bool _mcg,
        bool _mul1,
        at::Tensor _xh
    ) :
        trellis(std::move(_trellis)),
        suh(std::move(_suh)),
        svh(std::move(_svh)),
        K(_K),
        bias(std::move(_bias)),
        mcg(_mcg),
        mul1(_mul1),
        xh(std::move(_xh))
    {}

    void run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph);
    void run(const at::Tensor& x, at::Tensor& y);
    at::Tensor run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32);

    // Runtime LoRA (doc/lora_inference_plan.md, stage 1): the packed A (K, R) / B (R, N) fp16
    // pair for this linear, pushed from Python (Linear.sync_lora_bc). DATA ONLY: run()/run_gr()
    // never apply it -- the graph classes that support an in-graph adapter (BC_GatedMLP,
    // BC_Attention) emit the lora_gemv node explicitly, because its input pointer has to be
    // patched per replay exactly like the GEMM's, which only the owning graph can do
    c10::optional<at::Tensor> lora_a;
    c10::optional<at::Tensor> lora_b;

    void set_lora(at::Tensor a, at::Tensor b)
    {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "set_lora: A (K, R) / B (R, N) shape mismatch");
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "set_lora: A/B must be contiguous");
        lora_a = std::move(a);
        lora_b = std::move(b);
    }
    void clear_lora() { lora_a.reset(); lora_b.reset(); }
    bool has_lora() const { return lora_a.has_value(); }
    int lora_rank() const { return has_lora() ? (int) lora_a->size(1) : 0; }
};
