"""
Experimental training support for ExLlamaV3.

This subpackage explores QLoRA-style fine-tuning on top of EXL3-quantized
weights. The supported path is transformers-free: a self-contained,
autograd-friendly Llama/Mistral decoder forward built directly on exllamav3's
own loaded modules (:class:`NativeLlamaQLoRA`), with a memory-efficient
differentiable EXL3 linear (:class:`EXL3LoRAFunction`) and a streaming fused
cross-entropy head. See ``doc/qlora_handoff.md`` for status and
``doc/qlora_feasibility.md`` for the design rationale.

All reach-through into exllamav3's internal module layout (the ``..modules``
types, the loaded RoPE table, the trellis weight reconstruction) is funnelled
through ``exllamav3.training.backbone`` -- the single seam this code depends on,
so that promoting it into exllamav3 as a supported training entry point (or
pinning it from a standalone trainer) touches one file rather than many.
"""

from .qlora_linear import (
    EXL3LoRAFunction,
    qlora_linear_forward,
    QLoRALinear,
    reference_forward,
)
from .fused_ce import (
    FusedLinearCrossEntropy,
    fused_linear_cross_entropy,
    DEFAULT_CHUNK,
)
# native_llama imports exllamav3.modules lazily (inside its functions), so
# importing the symbols here does not pull in the CUDA extension at package
# import time.
from .native_llama import (
    NativeLlamaQLoRA,
    DiffLinear,
    DEFAULT_TARGET_MODULES,
)
from .preference import (
    dpo_loss,
    kto_loss,
    mismatched_kl_shift,
)
# realtime builds on native_llama the same lazy way; safe to import here.
from .realtime import (
    RealtimeQLoRA,
    RealtimeConfig,
)
from .aux_offload import (
    ModelParker,
)

__all__ = [
    "EXL3LoRAFunction",
    "qlora_linear_forward",
    "QLoRALinear",
    "reference_forward",
    "FusedLinearCrossEntropy",
    "fused_linear_cross_entropy",
    "DEFAULT_CHUNK",
    "NativeLlamaQLoRA",
    "DiffLinear",
    "DEFAULT_TARGET_MODULES",
    "dpo_loss",
    "kto_loss",
    "mismatched_kl_shift",
    "RealtimeQLoRA",
    "RealtimeConfig",
    "ModelParker",
]
