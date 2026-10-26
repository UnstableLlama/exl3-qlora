"""
Synthetic PEFT adapters for runtime-LoRA tests and benchmarks: random lora_A / lora_B pairs over the linear
projections of a loaded model, written in the layout exllamav3.LoRA.from_directory reads
(adapter_config.json + adapter_model.safetensors, base_model.model.<module key>.lora_{A,B}.weight).
"""

import json
import os

import torch

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def adapter_targets(model, targets = DEFAULT_TARGETS, layers: int | None = None) -> list:
    """The Linear modules of `model` (its iteration is recursive) whose key ends in one of `targets`, on the first
    `layers` decoder layers (all when None). Routed MoE experts are left out (they have their own, unfused path)"""
    from exllamav3.modules import Linear
    out = []
    for sub in model:
        if not isinstance(sub, Linear) or sub.is_sliced:
            continue
        key = sub.key
        if ".experts." in key or key.rsplit(".", 1)[-1] not in targets:
            continue
        if layers is not None:
            parts = key.split(".")
            try:
                idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                idx = 0
            if idx >= layers:
                continue
        out.append(sub)
    return out


def write_random_adapter(model, directory: str, rank: int = 16, alpha: float | None = None, seed: int = 0,
                         std: float = 0.02, targets = DEFAULT_TARGETS, layers: int | None = None) -> str:
    """Write a random PEFT adapter for `model` to `directory` and return it. A and B are N(0, std) in fp16
    (B is NOT zero, so the adapter visibly changes the model); alpha defaults to rank (scale 1)"""
    from safetensors.torch import save_file
    gen = torch.Generator(device = "cpu").manual_seed(seed)
    tensors = {}
    names = []
    for l in adapter_targets(model, targets, layers):
        k, n = l.in_features_unpadded, l.out_features_unpadded
        a = (torch.randn(rank, k, generator = gen) * std).half()   # PEFT lora_A: [r, in]
        b = (torch.randn(n, rank, generator = gen) * std).half()   # PEFT lora_B: [out, r]
        tensors[f"base_model.model.{l.key}.lora_A.weight"] = a.contiguous()
        tensors[f"base_model.model.{l.key}.lora_B.weight"] = b.contiguous()
        names.append(l.key.rsplit(".", 1)[-1])
    assert tensors, "no adapter targets found in the model"
    os.makedirs(directory, exist_ok = True)
    save_file(tensors, os.path.join(directory, "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": float(alpha if alpha is not None else rank),
        "target_modules": sorted(set(names)),
        "fan_in_fan_out": False,
        "use_rslora": False,
    }
    with open(os.path.join(directory, "adapter_config.json"), "w", encoding = "utf8") as f:
        json.dump(config, f, indent = 2)
    return directory
