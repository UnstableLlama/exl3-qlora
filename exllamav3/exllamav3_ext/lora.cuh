#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// Runtime LoRA delta for the graph-captured decode paths: y += (x @ A) @ B as two small kernels
// around a fp32 rank intermediate, both recorded with patchable I/O sites so they can sit inside
// a captured graph next to the quantized GEMM they extend (BC_LinearEXL3::lora_gr).
//
// lora_a: t[m, :R]  = x[m, :K] . a[:R, :K]      x half (.., K) contiguous, a half (R, >= K)
//                                               contiguous, t fp32 (>= m, R) contiguous
// lora_b: y[m, :N] += t[m, :R] @ b[:R, :N]      b half (R, >= N) contiguous, y half or fp32
//                                               (.., N), unit column stride, any row stride
//                                               (a column slice of a wider static is fine)
//
// The rows of x beyond a's width, or of b beyond y's, are never read: a padded adapter over an
// unpadded activation (and vice versa) needs no staging. Graph sites: GP_lora_x on lora_a's x,
// GP_lora_y on lora_b's y

void lora_a_gr
(
    const at::Tensor& x,
    const at::Tensor& a,
    at::Tensor& t,
    Graph* graph
);

void lora_b_gr
(
    const at::Tensor& t,
    const at::Tensor& b,
    at::Tensor& y,
    Graph* graph
);
