"""
Reference log-probability cache for preference training (DPO / KTO).

The reference model in ``qlora_train_pref.py`` is the frozen quantized base
(``NativeLlamaQLoRA.adapters_disabled()``), so a row's reference logp is a
constant of (base weights, token ids, numerics) -- it does not change across
micro-batches, epochs, or training runs on the same model. The trainer used
to recompute it with a no-grad forward on every micro-batch; TRL exposes the
same saving as ``precompute_ref_log_probs``. This module caches those values
by row content and persists them to disk so a later run on the same model
(any dataset that shares rows, any epoch count) never pays for them again.

Keying: each row is ``(prompt_ids, completion_ids)`` and its key is a blake2b
digest of the two id sequences, so the cache is independent of dataset
order, shuffling, ``--max-samples``, and which dataset file the row came from.
KTO's mismatched-pair KL rows are just ``(prompt_i, completion_j)`` and cache
the same way.

Validity: the file carries a fingerprint of everything the reference logp
depends on -- the model directory (config + a content sample of every
safetensors shard, so a requantization into the same directory invalidates
it), the compute dtype and the resolved attention plan (flash vs sdpa vs
eager differ at fp32-sum level). A file whose fingerprint does not match is
ignored, not merged.

Numerics: rows computed through the cache run in a sub-batch containing only
the missing rows, so their padding differs from the policy batch's. Right
padding under a causal mask leaves each row's own logps unaffected; the
remaining difference is kernel-level fp rounding, the same class of noise
as between two runs with different ``--batch``.

CPU-testable; no exllamav3 imports at module import time.
"""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from array import array
from typing import Callable, Optional, Sequence

import torch

CACHE_VERSION = 1
_SHARD_SAMPLE = 1 << 16   # bytes hashed from the head and tail of every shard


def row_key(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> str:
    """Content key for one (prompt, completion) row. Order-sensitive on both
    halves and on the prompt/completion split (the same tokens split
    differently are different rows)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(array("i", prompt_ids).tobytes())
    h.update(b"|")
    h.update(array("i", completion_ids).tobytes())
    return h.hexdigest()


def model_fingerprint(model_dir: str) -> dict:
    """Identity of the frozen base as a dict of stable strings. Cheap: the
    config file and a 64 KiB sample from the head and tail of each safetensors
    shard (name + size + sample), never a full read."""
    model_dir = os.path.realpath(model_dir)
    info = {"model_dir": model_dir}
    cfg = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg):
        with open(cfg, "rb") as f:
            info["config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    shards = []
    try:
        names = sorted(n for n in os.listdir(model_dir) if n.endswith(".safetensors"))
    except OSError:
        names = []
    for name in names:
        path = os.path.join(model_dir, name)
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(_SHARD_SAMPLE))
            if size > 2 * _SHARD_SAMPLE:
                f.seek(size - _SHARD_SAMPLE)
                h.update(f.read(_SHARD_SAMPLE))
        shards.append(f"{name}:{size}:{h.hexdigest()[:16]}")
    info["shards"] = shards
    return info


def fingerprint_digest(info: dict) -> str:
    return hashlib.sha256(
        json.dumps(info, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def default_cache_dir() -> str:
    base = os.environ.get("EXL3_QLORA_CACHE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "exl3_qlora")
    return os.path.join(base, "ref_logps")


class RefLogpCache:
    """In-memory ``row key -> reference logp`` map with a disk file behind it.

    ``fingerprint`` is the dict from :func:`model_fingerprint` plus whatever
    numerics keys the caller adds (compute dtype, attention plan). ``path``
    is the backing file; None keeps the cache in memory only (still saves the
    cross-epoch recompute within one run).
    """

    def __init__(self, fingerprint: dict, path: Optional[str] = None):
        self.fingerprint = dict(fingerprint)
        self.digest = fingerprint_digest(self.fingerprint)
        self.path = path
        self._map: dict[str, float] = {}
        self.dirty = False
        self.hits = 0
        self.misses = 0
        self.loaded_entries = 0
        self.load_note = ""
        if path and os.path.exists(path):
            self._load(path)

    # --- persistence -------------------------------------------------------

    def _load(self, path: str) -> None:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # unreadable / foreign file: start fresh
            self.load_note = f"unreadable ({type(e).__name__}); starting empty"
            return
        if not isinstance(blob, dict) or blob.get("version") != CACHE_VERSION:
            self.load_note = "unknown format; starting empty"
            return
        if blob.get("digest") != self.digest:
            self.load_note = ("fingerprint mismatch (different model / dtype / "
                              "attention plan); starting empty")
            return
        keys = blob.get("keys") or []
        logps = blob.get("logps")
        if logps is None or len(keys) != int(logps.numel()):
            self.load_note = "corrupt entry table; starting empty"
            return
        vals = logps.to(torch.float32).tolist()
        self._map = dict(zip(keys, vals))
        self.loaded_entries = len(self._map)

    def save(self, force: bool = False) -> bool:
        """Write the table if it changed (or ``force``). Atomic: tmp file in
        the same directory, then ``os.replace``. Returns True when written."""
        if not self.path or (not self.dirty and not force):
            return False
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        keys = list(self._map.keys())
        blob = {
            "version": CACHE_VERSION,
            "digest": self.digest,
            "fingerprint": self.fingerprint,
            "keys": keys,
            "logps": torch.tensor([self._map[k] for k in keys], dtype=torch.float32),
        }
        fd, tmp = tempfile.mkstemp(prefix=".ref_logps.", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(blob, f)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.dirty = False
        return True

    # --- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._map)

    def get(self, key: str) -> Optional[float]:
        return self._map.get(key)

    def put(self, key: str, logp: float) -> None:
        self._map[key] = float(logp)
        self.dirty = True

    def lookup_or_compute(
        self,
        rows: Sequence[tuple],
        compute_fn: Callable[[Sequence[tuple]], torch.Tensor],
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Reference logps for ``rows`` (``(prompt_ids, completion_ids)``
        pairs) as a ``[len(rows)]`` tensor. Rows already in the table are
        served from it; the rest go to ``compute_fn`` as ONE sub-list (in
        their original relative order) and its ``[n_missing]`` result is
        stored and merged back in place. Duplicate rows within one call are
        computed once."""
        keys = [row_key(p, c) for p, c in rows]
        out = torch.empty(len(rows), dtype=dtype, device=device)
        missing_idx: list[int] = []
        first_seen: dict[str, int] = {}
        for i, k in enumerate(keys):
            v = self._map.get(k)
            if v is not None:
                out[i] = v
                self.hits += 1
            elif k in first_seen:
                self.hits += 1          # filled below from the duplicate's compute
            else:
                first_seen[k] = i
                missing_idx.append(i)
                self.misses += 1
        if missing_idx:
            sub = [rows[i] for i in missing_idx]
            vals = compute_fn(sub)
            assert vals.numel() == len(sub), \
                f"compute_fn returned {vals.numel()} logps for {len(sub)} rows"
            vals_cpu = vals.detach().to("cpu", torch.float32).tolist()
            for i, v in zip(missing_idx, vals_cpu):
                self._map[keys[i]] = v
            self.dirty = True
            for i, k in enumerate(keys):
                if k in first_seen:
                    out[i] = self._map[k]
        return out

    def stats_line(self) -> str:
        total = self.hits + self.misses
        rate = (100.0 * self.hits / total) if total else 0.0
        where = self.path or "memory only"
        return (f"ref-logp cache: {self.hits} hits / {self.misses} misses "
                f"({rate:.0f}% hit) | {len(self._map)} entries | {where}")
