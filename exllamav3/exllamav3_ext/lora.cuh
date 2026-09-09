#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// Runtime-LoRA delta for the graph-captured decode paths (doc/lora_inference_plan.md, stage 1):
//
//     C[m, n] += sum_r (sum_k x[m, k] * A[k, r]) * B[r, n]
//
// as ONE cooperative launch: phase 1 partitions K across the persistent blocks for the partial
// products of t = x @ A, phase 2 reduces the partials, phase 3 gives every block a stripe of N for
// t @ B, added into C in place. Accumulation is fp32 through both stages and the delta is rounded
// once, into C's dtype, on the add (the contract in modules/lora_state.py::lora_delta_reference).
//
// x: (M, K) half, A: (K, R) half, B: (R, N) half (scale pre-folded), C: (M, N) half or float, all
// contiguous. M <= LORA_MAX_M (the BC slots' bsz * q_len bound), R <= LORA_MAX_RANK (a rank-2r
// PiSSA export at r = 128, or several adapters packed along rank). Grid geometry depends on the
// device only, never on R, so A/B/rank patch per replay (GP_lora_*) and an adapter swap or
// real-time update never recaptures.

#define LORA_MAX_M 128
#define LORA_MAX_RANK 256

void lora_gemv_gr
(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& c,
    Graph* graph
);

void lora_gemv
(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& c
);
