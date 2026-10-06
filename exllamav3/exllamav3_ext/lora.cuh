#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// Runtime LoRA delta for the graph-captured decode paths: y += (x @ A) @ B, as two small
// kernels so the rank-R intermediate can be shared between projections that read the same
// input (stack their A matrices, run lora_a once, then one lora_b per projection).
//
// lora_a: t[m, :R] = x[m, :K] @ a_t[:R, :K]^T    x half (.., K) contiguous, a_t half (R, >= K),
//                                                t float (>= m, R)
// lora_b: y[m, :N] += t[m, t_offset : t_offset + R] @ b[:R, :N]
//                                                b half (R, N), y half or float (.., N) contiguous
//
// Graph sites: GP_lora_x (lora_a's x) and GP_lora_y (lora_b's y)

void lora_a_gr
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    at::Tensor& t,
    Graph* graph
);

void lora_b_gr
(
    const at::Tensor& t,
    int t_offset,
    const at::Tensor& b,
    at::Tensor& y,
    Graph* graph
);
