"""
CPU tests for the preference-training reference-logp cache
(``exllamav3/training/ref_cache.py``): content keying, partial-batch
compute + merge, duplicate handling, atomic save / load round trip, and
fingerprint-mismatch invalidation. No GPU / extension / model needed.

    python -m pytest tests/test_ref_cache.py -q
"""

from __future__ import annotations
import importlib.util
import os
import sys
import types

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train_rc")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train_rc"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "exl3train_rc.ref_cache", os.path.join(_TRAIN_DIR, "ref_cache.py"))
_rc = importlib.util.module_from_spec(_spec)
sys.modules["exl3train_rc.ref_cache"] = _rc
_spec.loader.exec_module(_rc)

RefLogpCache = _rc.RefLogpCache
row_key = _rc.row_key
model_fingerprint = _rc.model_fingerprint


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        p = torch.randint(5, 1000, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g).tolist()
        c = torch.randint(5, 1000, (int(torch.randint(2, 7, (1,), generator=g)),), generator=g).tolist()
        out.append((p, c))
    return out


def _compute_fn(calls):
    """Deterministic fake reference: logp = -(sum of completion ids) / 100,
    and a log of every sub-list it was asked for."""
    def fn(sub):
        calls.append([tuple(c) for _, c in sub])
        return torch.tensor([-sum(c) / 100.0 for _, c in sub], dtype=torch.float32)
    return fn


def test_row_key_is_content_and_split_sensitive():
    assert row_key([1, 2, 3], [4, 5]) == row_key([1, 2, 3], [4, 5])
    assert row_key([1, 2, 3], [4, 5]) != row_key([1, 2], [3, 4, 5])   # split moves
    assert row_key([1, 2, 3], [4, 5]) != row_key([1, 2, 3], [5, 4])   # order
    assert row_key([], [4, 5]) != row_key([4, 5], [])


def test_lookup_computes_only_missing_rows_and_merges_in_place():
    rows = _rows(6)
    calls = []
    cache = RefLogpCache({"model": "x"})
    fn = _compute_fn(calls)

    out1 = cache.lookup_or_compute(rows[:4], fn)
    assert len(calls) == 1 and len(calls[0]) == 4
    assert cache.misses == 4 and cache.hits == 0 and cache.dirty

    # A batch mixing 2 known + 2 new rows: only the 2 new ones reach compute,
    # in their original relative order, and the output lines up with `rows`.
    mixed = [rows[1], rows[4], rows[3], rows[5]]
    out2 = cache.lookup_or_compute(mixed, fn)
    assert len(calls) == 2
    assert calls[1] == [tuple(rows[4][1]), tuple(rows[5][1])]
    expect = torch.tensor([-sum(c) / 100.0 for _, c in mixed])
    assert torch.equal(out2, expect)
    assert torch.equal(out1, torch.tensor([-sum(c) / 100.0 for _, c in rows[:4]]))
    assert cache.hits == 2 and cache.misses == 6

    # Fully cached batch: no compute call at all.
    cache.lookup_or_compute(list(reversed(rows)), fn)
    assert len(calls) == 2


def test_duplicate_rows_in_one_batch_computed_once():
    r = _rows(1)[0]
    calls = []
    cache = RefLogpCache({"model": "x"})
    out = cache.lookup_or_compute([r, r, r], _compute_fn(calls))
    assert len(calls) == 1 and len(calls[0]) == 1
    assert torch.equal(out, torch.full((3,), -sum(r[1]) / 100.0))


def test_save_load_round_trip_and_fingerprint_invalidation(tmp_path):
    rows = _rows(5, seed=1)
    path = str(tmp_path / "sub" / "ref.pt")
    fp = {"model_dir": "/m", "compute_dtype": "bfloat16"}
    c1 = RefLogpCache(fp, path)
    c1.lookup_or_compute(rows, _compute_fn([]))
    assert c1.save()
    assert not c1.dirty and os.path.exists(path)
    assert not c1.save()            # nothing changed: no rewrite
    assert not [n for n in os.listdir(tmp_path / "sub") if n.endswith(".tmp")]

    # Same fingerprint: every row is a hit, compute never called.
    c2 = RefLogpCache(fp, path)
    assert c2.loaded_entries == 5 and c2.load_note == ""
    calls = []
    out = c2.lookup_or_compute(rows, _compute_fn(calls))
    assert not calls and c2.hits == 5
    assert torch.equal(out, torch.tensor([-sum(c) / 100.0 for _, c in rows]))

    # Different numerics: the file is ignored, not merged.
    c3 = RefLogpCache({**fp, "compute_dtype": "float16"}, path)
    assert len(c3) == 0 and "mismatch" in c3.load_note
    c3.lookup_or_compute(rows[:1], _compute_fn(calls))
    assert len(calls) == 1


def test_garbage_file_starts_empty(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a torch file")
    c = RefLogpCache({"m": 1}, str(path))
    assert len(c) == 0 and c.load_note


def test_model_fingerprint_tracks_shard_content(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["X"]}')
    (d / "output.safetensors").write_bytes(b"\x00" * 1000)
    a = _rc.fingerprint_digest(model_fingerprint(str(d)))
    (d / "output.safetensors").write_bytes(b"\x01" * 1000)   # same size, new content
    b = _rc.fingerprint_digest(model_fingerprint(str(d)))
    assert a != b
    (d / "config.json").write_text('{"architectures": ["Y"]}')
    assert _rc.fingerprint_digest(model_fingerprint(str(d))) not in (a, b)
