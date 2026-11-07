"""
CPU tests for the YAML launcher's argument forwarding (``training/qlora_train.py``):
structured keys (``lr_groups`` list of mappings, an inline ``trainable_tokens_init``
mapping) travel to the backend as JSON, token lists as repeated values, and the
per-backend key sets accept the new knobs.

Run:  python tests/test_qlora_launcher.py
"""
from __future__ import annotations
import json
import os
import sys
import importlib.util

import pytest

pytestmark = pytest.mark.nogpu

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "qlora_train_launcher", os.path.join(_ROOT, "training", "qlora_train.py"))
_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_launcher)


def _argv(cfg):
    cfg = dict(cfg)
    cfg.setdefault("model", "/m")
    cfg.setdefault("out", "/o")
    cfg.setdefault("dataset", "d")
    return _launcher.build_backend_argv(_launcher.flatten_config(cfg), "cfg.yaml")


def test_lr_groups_forwarded_as_json():
    groups = [{"name": "global_attn", "lr": 1e-5, "layers": ["global_attention"]},
              {"name": "early", "lr_mul": 0.5, "modules": ["layers.0.", "layers.1."]}]
    argv = _argv({"lr": 4e-5, "lr_groups": groups})
    i = argv.index("--lr-groups")
    got = [json.loads(argv[i + 1]), json.loads(argv[i + 2])]
    assert got == groups
    # DDP, EBFT, preference and GRPO backends accept the key too.
    for extra in ({"parallel": "ddp"}, {"method": "ebft"}, {"method": "dpo"},
                  {"method": "grpo", "reward": ["length"]}):
        _argv({"lr_groups": groups, **extra})
    print("[launcher] lr_groups list -> repeated --lr-groups JSON objects PASSED")


def test_trainable_tokens_forwarded():
    cfg = {"trainable_tokens": ["<unused6>", "<unused7>", "40-45"],
           "trainable_tokens_init": {"<unused6>": "IceWing, an ice dragon",
                                     "<unused7>": "MudWing, a mud dragon"},
           "trainable_tokens_head": "tied", "trainable_tokens_lr": 1e-3,
           "targets": []}
    argv = _argv(cfg)
    i = argv.index("--trainable-tokens")
    assert argv[i + 1:i + 4] == ["<unused6>", "<unused7>", "40-45"]
    j = argv.index("--trainable-tokens-init")
    assert json.loads(argv[j + 1]) == cfg["trainable_tokens_init"]
    assert argv[argv.index("--trainable-tokens-head") + 1] == "tied"
    assert float(argv[argv.index("--trainable-tokens-lr") + 1]) == 1e-3
    # Single/split only: DDP rejects it.
    with pytest.raises(SystemExit):
        _argv({**cfg, "parallel": "ddp"})
    # A path-valued init is forwarded verbatim.
    argv = _argv({"trainable_tokens": ["<unused6>"], "trainable_tokens_init": "names.json"})
    assert argv[argv.index("--trainable-tokens-init") + 1] == "names.json"
    print("[launcher] trainable_tokens list / init mapping / head / lr forwarded PASSED")


def test_engram_keys_forwarded():
    cfg = {"train_ple": True, "ple_lr": 1e-4, "train_ngram_rows": True,
           "ngram_rows_lr": 5e-3, "ngram_rows_max": 500000,
           "targets": ["ple_proj", "q_proj"]}
    argv = _argv(cfg)
    assert "--train-ple" in argv and "--train-ngram-rows" in argv
    assert float(argv[argv.index("--ple-lr") + 1]) == 1e-4
    assert float(argv[argv.index("--ngram-rows-lr") + 1]) == 5e-3
    assert argv[argv.index("--ngram-rows-max") + 1] == "500000"
    i = argv.index("--targets")
    assert argv[i + 1:i + 3] == ["ple_proj", "q_proj"]
    # Native single/split only.
    with pytest.raises(SystemExit):
        _argv({**cfg, "parallel": "ddp"})
    print("[launcher] train_ple / train_ngram_rows keys forwarded PASSED")


def main():
    test_lr_groups_forwarded_as_json()
    test_trainable_tokens_forwarded()
    test_engram_keys_forwarded()
    print("\nAll launcher forwarding checks passed.")


if __name__ == "__main__":
    main()
