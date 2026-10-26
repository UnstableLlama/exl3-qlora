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
    float K;                      // bitrate, integer or half-integer (see quant/bits_k.cuh)
    c10::optional<at::Tensor> bias;
    bool mcg;
    bool mul1;
    at::Tensor xh;

    // Runtime LoRA (set_lora): y += (x @ A) @ B on top of the quantized product, emitted by
    // lora_gr() as two graph nodes (lora.cuh) right after the GEMM of a captured decode block.
    // A is stored transposed, (R, K) with K the padded in_features; B is (R, N) with the
    // alpha/r scale folded in; several adapters on one linear are stacked along R. lora_t is
    // the fp32 rank intermediate for up to LORA_MAX_ROWS tokens (the largest graphed shape,
    // BC_Attention::MAX_BSZ * MAX_QLEN). run_gr() itself stays adapter-free: the python
    // Linear.forward applies the adapter on that path, graph owners call lora_gr() explicitly
    // per projection and lora_args() to patch its sites at replay
    static constexpr int LORA_MAX_ROWS = 128;
    c10::optional<at::Tensor> lora_a;
    c10::optional<at::Tensor> lora_b;
    at::Tensor lora_t;

    BC_LinearEXL3
    (
        at::Tensor _trellis,
        at::Tensor _suh,
        at::Tensor _svh,
        float _K,
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

    // Attach (both given) or detach (both nullopt) the runtime LoRA. Bumps the graph structure
    // epoch so every captured graph re-records on its next run
    void set_lora(c10::optional<at::Tensor> a, c10::optional<at::Tensor> b);
    bool has_lora() const { return lora_a.has_value(); }
    // y += (x @ A) @ B; no-op without an adapter. x (.., K) half, y (.., N) half or fp32
    void lora_gr(const at::Tensor& x, at::Tensor& y, Graph* graph);
    void run_lora(const at::Tensor& x, at::Tensor& y);
    // The replay patches for lora_gr's nodes, in recording order; nothing without an adapter
    void lora_args(std::vector<PPTR>& params, void* x, void* y) const;
};
