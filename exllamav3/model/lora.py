"""
LoRA (Low-Rank Adaptation) support for ExLlamaV3.

Loads PEFT-format LoRA adapters and applies them at runtime without
modifying base model weights. Multiple adapters can be loaded
simultaneously; all loaded adapters are applied during forward pass.

Usage::

    lora = LoRA.from_directory(model, "/path/to/peft-adapter")
    # All generation now includes this adapter's contribution
    response = generator.generate(prompt = "Hello", ...)
    # Unload to revert to base model
    lora.unload()

Compatible with adapters trained via PEFT/Unsloth on the full-precision
base model. LoRA weights are applied on top of the dequantized output
of each target linear layer.
"""

from __future__ import annotations
import os
import json
import math
import re
import torch
from safetensors.torch import load_file as safe_load_file
from ..modules.linear import Linear
from ..modules.embedding import Embedding

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model


class LoRA:
    """
    LoRA adapter loaded from PEFT format.

    Stores pre-transposed and pre-scaled A/B weight matrices on target
    Linear modules. During forward pass, Linear.forward() computes:
    ``output += input @ A @ B`` for each loaded adapter.
    """

    @staticmethod
    def from_directory(
            model: Model,
            directory: str,
            lora_scaling: float = 1.0,
            module_lora_scale: float | None = None,
    ) -> LoRA:
        """
        Load LoRA adapter from a PEFT directory.

        :param model:
            Loaded ExLlamaV3 model instance.

        :param directory:
            Path to directory containing adapter_config.json and
            adapter_model.safetensors (or .bin).

        :param lora_scaling:
            Additional scaling factor applied on top of alpha/r.

        :param module_lora_scale:
            Base scale for the embed/head LoRAs in lora_modules.safetensors,
            overriding the adapter config. See :meth:`_load_module_adapters`.
        """
        config_path = os.path.join(directory, "adapter_config.json")
        weights_st = os.path.join(directory, "adapter_model.safetensors")
        weights_bin = os.path.join(directory, "adapter_model.bin")
        modules_to_save = os.path.join(directory, "modules_to_save.safetensors")
        lora_modules = os.path.join(directory, "lora_modules.safetensors")

        if os.path.exists(weights_st):
            return LoRA(model, config_path, weights_st, lora_scaling, module_lora_scale)
        if os.path.exists(weights_bin):
            return LoRA(model, config_path, weights_bin, lora_scaling, module_lora_scale)
        if os.path.exists(config_path) and (
            os.path.exists(modules_to_save) or os.path.exists(lora_modules)
        ):
            return LoRA(model, config_path, None, lora_scaling, module_lora_scale)
        raise FileNotFoundError(f"No LoRA adapter found in {directory}")

    @torch.inference_mode()
    def __init__(
            self,
            model: Model,
            config_path: str,
            weights_path: str | None,
            lora_scaling: float = 1.0,
            module_lora_scale: float | None = None,
    ):
        self.target_modules = {}
        # modules_to_save / embed-LoRA targets we mutate in place, tracked so unload()
        # can revert them (head full-weight override; restored embedding weight).
        self._full_weight_targets = []          # [Linear] -- reset .lora_full_weight
        self._embed_restore = []                # [(Embedding, original_weight)]
        self.name = os.path.basename(os.path.dirname(config_path))

        # Read adapter config
        with open(config_path, encoding="utf8") as f:
            config = json.load(f)

        self.lora_r = config["r"]
        self.lora_alpha = float(config["lora_alpha"])
        self.use_rslora = bool(config.get("use_rslora", False))
        # Per-module alpha overrides (PEFT alpha_pattern: regex -> alpha).
        # Per-module RANKS need no config: each module's rank is read off its
        # own lora_A/lora_B tensor shape below, so rank_pattern adapters
        # (e.g. a smaller rank on MoE routed experts) scale correctly.
        self.alpha_pattern = {k: float(v)
                              for k, v in config.get("alpha_pattern", {}).items()}
        self.user_scaling = lora_scaling

        effective_alpha = self.lora_alpha
        if self.use_rslora:
            effective_alpha *= math.sqrt(self.lora_r)

        # Global scaling (config-level r/alpha) -- reported in the load line and
        # correct for uniform-rank adapters; mixed-rank/alpha modules get their
        # own value from _module_scaling.
        self.lora_scaling = lora_scaling * effective_alpha / self.lora_r

        # Scale for the embed/head LoRAs (lora_modules.safetensors), which are
        # saved RAW -- no scale folded in -- and so must NOT reuse lora_scaling.
        # The trainer applies alpha/denom (denom = sqrt(r) under rslora, else r)
        # with the ORIGINAL r/alpha; a pissa export rewrites the config triple to
        # the converted 2r/2r/rslora=False so the per-linear path resolves to 1.0.
        # Reusing that here silently under-applies the module deltas (a pissa
        # r32/alpha32/rslora run wants 5.657, not 1.0). Priority: explicit
        # argument > config key written by save_adapter > legacy alpha/r fallback
        # (which reproduces the pre-fix behavior for adapters saved without it).
        ms = module_lora_scale
        if ms is None:
            ms = config.get("module_lora_scale")
        if ms is None:
            ms = effective_alpha / self.lora_r
            self.module_scale_is_guess = True
        else:
            self.module_scale_is_guess = False
        self.module_lora_scale = lora_scaling * float(ms)

        if config.get("fan_in_fan_out", False):
            raise ValueError("fan_in_fan_out mode is not supported")

        # Build modules dict if needed
        if model.modules_dict is None:
            model.modules_dict = {m.key: m for m in model}

        # Load per-linear LoRA weights, if present. Native QLoRA can also save
        # module-only adapters (--targets [] with --lora-head/--lora-embed or
        # --train-head/--train-embeddings); those are loaded below from their
        # sidecar safetensors files.
        if weights_path is None:
            raw_tensors = {}
        elif weights_path.endswith(".safetensors"):
            raw_tensors = safe_load_file(weights_path, device="cpu")
        else:
            raw_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)

        loaded = 0
        skipped_keys = []
        tp_skipped = []

        for key, tensor in raw_tensors.items():
            # Skip non-LoRA keys (e.g. modules_to_save, original_module)
            if ".lora_A." not in key and ".lora_B." not in key:
                continue

            # Extract full path and lora half from PEFT key
            full_path, lora_half = self._parse_key(key)
            if full_path is None:
                skipped_keys.append(key)
                continue

            # Match against model modules by suffix to handle any
            # PEFT key prefix (base_model.model.*, etc.)
            target = None
            module_key = None
            path_parts = full_path.split(".")
            for start in range(len(path_parts)):
                candidate = ".".join(path_parts[start:])
                t = model.modules_dict.get(candidate)
                if t is not None and isinstance(t, Linear):
                    target = t
                    module_key = candidate
                    break

            if target is None:
                skipped_keys.append(key)
                continue

            # Tensor-parallel sliced modules not supported
            if target.is_sliced:
                tp_skipped.append(key)
                continue

            if tensor.dtype in (torch.bfloat16, torch.float32):
                tensor = tensor.to(torch.float16)

            # Transpose for efficient matmul: x @ A @ B
            # PEFT stores lora_A as [rank, in_features] and
            # lora_B as [out_features, rank].
            # We want A as [in_features, rank] and B as [rank, out_features].
            tensor = tensor.T.contiguous()

            # Pre-scale B matrix. The module's rank is the tensor's own leading
            # dim (post-transpose [r, out]), NOT config r: a rank_pattern
            # adapter (e.g. smaller rank on MoE routed experts) has per-module
            # ranks, and alpha_pattern can override alpha per module.
            if lora_half == "lora_B":
                scaling = self._module_scaling(full_path, tensor.shape[0])
                if scaling != 1.0:
                    tensor.mul_(scaling)

            # Pad to match target dimensions (quantized layers may pad features to multiples of block size)
            if lora_half == "lora_A" and tensor.shape[0] < target.in_features:
                padded = torch.zeros(target.in_features, tensor.shape[1], dtype=tensor.dtype)
                padded[:tensor.shape[0]] = tensor
                tensor = padded
            elif lora_half == "lora_B" and tensor.shape[1] < target.out_features:
                padded = torch.zeros(tensor.shape[0], target.out_features, dtype=tensor.dtype)
                padded[:, :tensor.shape[1]] = tensor
                tensor = padded

            tensor = tensor.to(target.device)

            # Register on target module
            if lora_half == "lora_A":
                target.lora_a_tensors[self] = tensor
            else:
                target.lora_b_tensors[self] = tensor

            self.target_modules[module_key] = target
            loaded += 1

        print(
            f" -- LoRA '{self.name}': loaded {loaded} tensors "
            f"(r={self.lora_r}, alpha={self.lora_alpha:.0f}, "
            f"scaling={self.lora_scaling:.4f})"
        )
        if skipped_keys:
            print(
                f" -- LoRA '{self.name}': skipped {len(skipped_keys)} "
                f"unmatched keys"
            )
        if tp_skipped:
            print(
                f" -- LoRA '{self.name}': skipped {len(tp_skipped)} tensors "
                f"on tensor-parallel sliced modules"
            )

        # Routed-expert projections (BlockSparseMLP): while an adapter is loaded
        # on any expert projection, the MoE dispatch skips the fused expert
        # kernels (exl3_moe / BC graphs / mgemm) and runs the per-expert torch
        # path so the adapter is applied -- at a real decode-speed cost. Note it
        # so a slow MoE decode isn't mysterious (merge the adapter for deploy).
        expert_pat = re.compile(r"\.experts\.\d+\.")
        n_expert = sum(1 for k in self.target_modules if expert_pat.search(k))
        if n_expert:
            print(
                f" -- LoRA '{self.name}': {n_expert} target modules are routed-"
                f"expert projections; MoE layers will use the unfused per-expert "
                f"path while this adapter is loaded (slower decode -- merge the "
                f"adapter for deployment)"
            )

        # Embedding / LM-head adapters trained by the native QLoRA trainer live in
        # SEPARATE files next to adapter_model.safetensors (the per-linear loader above
        # only handles lora_A/lora_B on Linear modules). Apply them here so a head/embed
        # adapter actually takes effect at inference instead of being silently dropped.
        self._load_module_adapters(model, os.path.dirname(config_path))

    def _find_module(self, model, key, cls):
        """Locate a loaded module by exact key, falling back to the unique module of
        the given type (handles key-prefix variation across architectures)."""
        m = model.modules_dict.get(key)
        if m is not None and isinstance(m, cls):
            return m
        of_type = [mod for mod in model.modules_dict.values() if isinstance(mod, cls)]
        return of_type[0] if len(of_type) == 1 else None

    @torch.inference_mode()
    def _load_module_adapters(self, model, directory):
        """Apply the native trainer's embedding / LM-head adapters:

          * ``modules_to_save.safetensors`` -- fully fine-tuned embed/head
            (``--train-embeddings`` / ``--train-head``): the head becomes a full-weight
            override on the LM-head Linear; the embedding weight is replaced in place.
          * ``lora_modules.safetensors`` -- low-rank embed/head LoRA
            (``--lora-embed`` / ``--lora-head``): the head LoRA rides the LM-head
            Linear's runtime LoRA slot; the embed LoRA is folded into the embedding
            weight (scaled to undo the module's multiplier/normalize, since the trainer
            adds the shift *after* that scaling).

        Both files store tensors in the trainer's internal orientation already sized to
        the head Linear's padded in/out features, so no transpose/pad is needed here.
        """
        head = self._find_module(model, "lm_head", Linear)
        embed = self._find_module(model, "model.embed_tokens", Embedding)

        # --- modules_to_save: fully fine-tuned embed / head ---
        ms_path = os.path.join(directory, "modules_to_save.safetensors")
        if os.path.exists(ms_path):
            ms = safe_load_file(ms_path, device="cpu")
            applied = []
            if "lm_head.weight" in ms and head is not None and not head.is_sliced:
                # Saved [out, in]; the override wants [in, out].
                w = ms["lm_head.weight"].t().contiguous().to(torch.float16).to(head.device)
                head.lora_full_weight = w
                self._full_weight_targets.append(head)
                applied.append("lm_head")
            if "model.embed_tokens.weight" in ms and embed is not None:
                self._embed_restore.append((embed, embed.embedding.weight.data))
                new_w = ms["model.embed_tokens.weight"].to(
                    embed.embedding.weight.dtype).to(embed.embedding.weight.device)
                embed.embedding.weight.data = new_w.contiguous()
                applied.append("embed_tokens")
            if applied:
                print(f" -- LoRA '{self.name}': applied modules_to_save {applied}")

        # --- embed / head LoRA (low-rank) ---
        ml_path = os.path.join(directory, "lora_modules.safetensors")
        if os.path.exists(ml_path):
            ml = safe_load_file(ml_path, device="cpu")
            applied = []
            if "lm_head.lora_a" in ml and head is not None and not head.is_sliced:
                a = ml["lm_head.lora_a"].to(torch.float16).to(head.device)   # [in, r]
                b = ml["lm_head.lora_b"].to(torch.float16).to(head.device)   # [r, out]
                # apply_lora computes x @ a @ b with no scaling, so bake the module
                # scale (and the user's --lora-scaling) into b. NOT lora_scaling:
                # see the module_lora_scale note in __init__.
                b = b * self.module_lora_scale
                head.lora_a_tensors[self] = a
                head.lora_b_tensors[self] = b
                self.target_modules.setdefault("lm_head", head)
                applied.append("lm_head")
            if "embed_tokens.lora_a" in ml and embed is not None:
                a = ml["embed_tokens.lora_a"].to(torch.float32)              # [V, r]
                b = ml["embed_tokens.lora_b"].to(torch.float32)              # [r, d]
                # The trainer adds scale * (a@b) to the embedding output AFTER the
                # module's multiplier / sqrt(d) normalize. Folding into the weight means
                # that scaling will reapply, so divide it out first.
                factor = float(getattr(embed, "multiplier", 1.0) or 1.0)
                if getattr(embed, "normalize", False):
                    factor *= float(embed.hidden_size) ** 0.5
                delta = (self.module_lora_scale / factor) * (a @ b)          # [V, d]
                w = embed.embedding.weight
                self._embed_restore.append((embed, w.data))
                w.data = (w.data + delta.to(w.dtype).to(w.device)).contiguous()
                applied.append("embed_tokens")
            if applied:
                print(f" -- LoRA '{self.name}': applied module LoRA {applied} "
                      f"(scaling={self.module_lora_scale:.4f})")
                if self.module_scale_is_guess:
                    print(f" -- LoRA '{self.name}': WARNING -- no 'module_lora_scale' "
                          f"in adapter_config.json; guessed {self.module_lora_scale:.4f} "
                          f"from alpha/r. If this adapter was trained with rslora "
                          f"and/or exported through pissa, that guess is WRONG "
                          f"(pass module_lora_scale=alpha/sqrt(r) explicitly).")

    def _module_scaling(self, full_path: str, module_r: int) -> float:
        """
        Effective B pre-scale for one module: ``user_scaling * alpha / r`` with
        the module's OWN rank (taken from its tensor shape by the caller) and
        any ``alpha_pattern`` override. Pattern keys use PEFT's matching rule
        (``(^|.*\\.)pattern$`` against the module path). Equals the global
        ``lora_scaling`` for a uniform-rank adapter.
        """
        alpha = self.lora_alpha
        for pat, a in self.alpha_pattern.items():
            if re.match(rf"(^|.*\.){pat}$", full_path):
                alpha = a
                break
        if self.use_rslora:
            alpha *= math.sqrt(module_r)
        return self.user_scaling * alpha / module_r

    @staticmethod
    def _parse_key(key: str) -> tuple[str | None, str | None]:
        """
        Parse PEFT tensor key to (full_path, lora_half).

        Returns the full dotted path before lora_A/lora_B and the half
        name. The caller matches this path against model modules by
        suffix, so any PEFT key prefix format is handled automatically.

            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            -> ("base_model.model.model.layers.0.self_attn.q_proj", "lora_A")
        """
        parts = key.split(".")
        for j, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                return ".".join(parts[:j]), p
        return None, None

    def unload(self):
        """Remove this adapter's tensors from all target modules."""
        for target in self.target_modules.values():
            target.lora_a_tensors.pop(self, None)
            target.lora_b_tensors.pop(self, None)

        # Revert in-place module adapters (modules_to_save head/embed, folded embed LoRA).
        for linear in self._full_weight_targets:
            linear.lora_full_weight = None
        for embed, original in reversed(self._embed_restore):
            embed.embedding.weight.data = original

        self.target_modules = {}
        self._full_weight_targets = []
        self._embed_restore = []
