"""
CPU test for ``training/repurpose_tokens.py``: on a synthetic BPE tokenizer
with ``<unusedN>`` entries, renaming them to new names makes each name a
single token (its old id) alone and inside text, the companion
``tokenizer_config.json`` / ``added_tokens.json`` follow, a dry run writes
nothing, and renaming something that is not an ``<unused>`` token is refused.

Run:  python tests/test_repurpose_tokens.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.nogpu

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "training", "repurpose_tokens.py")

tokenizers = pytest.importorskip("tokenizers")


def _make_model_dir(d):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=200,
        special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"] + [f"<unused{i}>" for i in range(4)])
    tok.train_from_iterator(
        ["the dragon flew home", "ice and mud and night", "wings of the tribes",
         "the little dragonet", "flew over the ice"] * 20, trainer)
    tok.save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({"added_tokens_decoder": {}, "model_max_length": 1024}, f)
    with open(os.path.join(d, "added_tokens.json"), "w") as f:
        json.dump({}, f)
    return tok


def _run(*args):
    return subprocess.run([sys.executable, _SCRIPT, *args], capture_output=True, text=True)


def test_repurpose_unused_tokens():
    from tokenizers import Tokenizer
    with tempfile.TemporaryDirectory() as d:
        base = _make_model_dir(d)
        unused0 = base.token_to_id("<unused0>")
        unused1 = base.token_to_id("<unused1>")
        assert len(base.encode("IceWing", add_special_tokens=False).ids) > 1

        # --list shows the spare tokens.
        r = _run("--model", d, "--list")
        assert r.returncode == 0 and "<unused0>" in r.stdout, r.stdout + r.stderr

        # Dry run: verified, nothing written.
        before = open(os.path.join(d, "tokenizer.json")).read()
        r = _run("--model", d, "--map", "<unused0>=IceWing", "<unused1>=MudWing")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "dry run" in r.stdout and "OK" in r.stdout
        assert open(os.path.join(d, "tokenizer.json")).read() == before

        # Write to a copy.
        out = os.path.join(d, "tok_out")
        r = _run("--model", d, "--map", "<unused0>=IceWing", "<unused1>=MudWing",
                 "--out", out, "--write")
        assert r.returncode == 0, r.stdout + r.stderr
        edited = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
        assert edited.encode("IceWing", add_special_tokens=False).ids == [unused0]
        assert edited.encode("MudWing", add_special_tokens=False).ids == [unused1]
        sent = edited.encode("the IceWing flew home", add_special_tokens=False).ids
        assert sent.count(unused0) == 1
        assert edited.token_to_id("<unused0>") is None
        assert edited.decode([unused0]) == "IceWing"
        # Other tokens unchanged.
        assert edited.encode("the dragon flew home", add_special_tokens=False).ids == \
            base.encode("the dragon flew home", add_special_tokens=False).ids
        cfg = json.load(open(os.path.join(out, "tokenizer_config.json")))
        assert cfg["added_tokens_decoder"][str(unused0)]["content"] == "IceWing"
        assert cfg["added_tokens_decoder"][str(unused0)]["special"] is False
        assert json.load(open(os.path.join(out, "added_tokens.json")))["MudWing"] == unused1
        # The source dir is untouched.
        assert open(os.path.join(d, "tokenizer.json")).read() == before

        # Refusals: not an <unused> token; an existing token as the new name.
        r = _run("--model", d, "--map", "<eos>=Foo")
        assert r.returncode != 0 and "not an <unusedN> token" in (r.stdout + r.stderr)
        r = _run("--model", d, "--map", "<unused2>=dragon")
        assert r.returncode != 0 and "already a token" in (r.stdout + r.stderr)
    print("[repurpose_tokens] <unusedN> -> single whole-word tokens, sidecars updated, "
          "dry run inert, refusals PASSED")


def main():
    test_repurpose_unused_tokens()
    print("\nAll repurpose_tokens checks passed.")


if __name__ == "__main__":
    main()
