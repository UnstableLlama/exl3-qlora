"""
CPU tests for exllamav3/training/realtime.py -- the real-time
(inference-time) training coordinator:

  * RWLock semantics: concurrent readers, writer exclusion, writer preference
    (a waiting ingest blocks NEW inference readers while in-flight ones
    drain);
  * sample encoding: prompt/response exact mask boundary, text supervision,
    pre-tokenized passthrough + validation, messages via a render_segments
    callable, BOS normalization (auto-prepending tokenizer), seq_len
    truncation, eot_text;
  * the ingest loop: in-order consumption, batch/grad-accum windowing, step
    counting across ingests, token-weighted mean loss, adapter sync + update
    callbacks (cache invalidation) firing once per ingest, generator drain;
  * checkpoint policy: timestamped names, cadence, pruning to
    keep_checkpoints, optimizer-state save/resume via load();
  * the externally settable constant lr.

The net/tokenizer are stubs implementing only the narrow surface the
coordinator uses. No GPU / compiled extension / real model needed. Run:
    python tests/test_realtime.py
"""

from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_load("fused_ce")
rt_mod = _load("realtime")

RealtimeQLoRA = rt_mod.RealtimeQLoRA
RealtimeConfig = rt_mod.RealtimeConfig
RWLock = rt_mod.RWLock


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Ids:
    def __init__(self, ids): self._ids = ids
    def tolist(self): return list(self._ids)


class StubTokenizer:
    """Char-level tokenizer that auto-prepends BOS on EVERY encode() call even
    with add_bos=False (the HF add_bos_token=true behavior the encoding must
    normalize away). No pad token, so the coordinator falls back to EOS."""
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def encode(self, text, add_bos=False, encode_special_tokens=True):
        return [_Ids([self.bos_token_id] + [ord(c) for c in text])]


class StubNet(nn.Module):
    """The narrow NativeLlamaQLoRA surface RealtimeQLoRA drives. One real
    parameter so AdamW/clipping run for real; compute_loss returns queued
    constants (rides the parameter with a zero coefficient so backward works)
    and records every micro-batch for order/shape assertions."""

    def __init__(self, losses=None):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.losses = list(losses or [])
        self.seen = []            # (input_ids, labels, attn) tensors per call
        self.applied = 0
        self.removed = 0
        self.saved = []           # directories save_adapter wrote to
        self.loaded = []          # directories load_adapter read from
        self.target_modules = ["q_proj"]

    def param_groups(self, weight_decay):
        return [{"params": [self.p], "weight_decay": weight_decay}]

    def trainable_parameters(self):
        return [self.p]

    def num_trainable(self):
        return 1

    def compute_loss(self, input_ids, labels, attention_mask=None, chunk=0,
                     position_ids=None, seg_ids=None):
        self.seen.append((input_ids, labels, attention_mask))
        c = self.losses.pop(0) if self.losses else 1.0
        return self.p.sum() * 0.0 + c

    def apply_to_native(self, scaling=1.0):
        self.applied += 1

    def remove_from_native(self):
        self.removed += 1

    def save_adapter(self, directory, base_model_name_or_path=None):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "adapter_model.safetensors"), "w") as f:
            f.write("stub")

    def load_adapter(self, directory):
        self.loaded.append(directory)
        return 1


class StubGenerator:
    """num_remaining_jobs counts down per poll (simulating in-flight jobs
    draining); pagetable records resets."""

    def __init__(self, jobs=0):
        self.jobs = jobs
        self.pagetable = types.SimpleNamespace(resets=0)
        self.pagetable.reset_page_table = self._reset

    def _reset(self):
        self.pagetable.resets += 1

    def num_remaining_jobs(self):
        n = self.jobs
        if n > 0:
            self.jobs -= 1
        return n


def make_rt(config=None, net=None, **kw):
    return RealtimeQLoRA(None, StubTokenizer(), config or RealtimeConfig(),
                         net=net or StubNet(), **kw)


# ---------------------------------------------------------------------------
# RWLock
# ---------------------------------------------------------------------------

def test_rwlock():
    lock = RWLock()

    # Concurrent readers.
    lock.acquire_read()
    lock.acquire_read()
    lock.release_read()

    # A waiting writer blocks NEW readers (writer preference) but enters only
    # once the in-flight reader drains.
    state = {"writer_in": False, "late_reader_in": False}
    writer_started = threading.Event()

    def writer():
        writer_started.set()
        with lock.write():
            state["writer_in"] = True

    def late_reader():
        with lock.read():
            state["late_reader_in"] = True

    wt = threading.Thread(target=writer)
    wt.start()
    writer_started.wait()
    time.sleep(0.05)                       # writer now queued on the held read
    assert not state["writer_in"]
    rt_thread = threading.Thread(target=late_reader)
    rt_thread.start()
    time.sleep(0.05)
    assert not state["late_reader_in"]     # blocked behind the waiting writer
    lock.release_read()                    # drain the in-flight reader
    wt.join(2)
    rt_thread.join(2)
    assert state["writer_in"] and state["late_reader_in"]

    # Timeout raises instead of hanging.
    lock.acquire_write()
    try:
        try:
            lock.acquire_read(timeout=0.05)
            assert False, "expected TimeoutError"
        except TimeoutError:
            pass
    finally:
        lock.release_write()
    print("rwlock: OK")


# ---------------------------------------------------------------------------
# Sample encoding
# ---------------------------------------------------------------------------

def test_encode_prompt_response():
    rt = make_rt(RealtimeConfig(eot_text="!"))
    ex = rt._encode({"prompt": "ab", "response": "cd"})
    # one BOS survives (auto-prepended twice, deduped once, then add_bos is a
    # no-op); prompt masked, response + eot supervised, boundary exact
    assert ex["input_ids"] == [1, ord("a"), ord("b"), ord("c"), ord("d"), ord("!")]
    assert ex["labels"] == [-100, -100, -100, ord("c"), ord("d"), ord("!")]
    print("encode prompt/response: OK")


def test_encode_text_and_pretokenized():
    rt = make_rt()
    ex = rt._encode({"text": "xy"})
    assert ex["input_ids"] == [1, ord("x"), ord("y")]
    assert ex["labels"] == [-100, ord("x"), ord("y")]       # BOS never supervised

    ex = rt._encode({"input_ids": [7, 8, 9]})
    assert ex["labels"] == [7, 8, 9]                        # default: supervise all
    ex = rt._encode({"input_ids": [7, 8, 9], "labels": [-100, 8, 9]})
    assert ex["labels"] == [-100, 8, 9]
    try:
        rt._encode({"input_ids": [7], "labels": [1, 2]})
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        rt._encode({"bogus": 1})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("encode text/pretokenized: OK")


def test_encode_truncation():
    rt = make_rt(RealtimeConfig(seq_len=3))
    ex = rt._encode({"text": "abcdefg"})
    assert len(ex["input_ids"]) == 3 and len(ex["labels"]) == 3
    ex = rt._encode({"input_ids": list(range(10, 20))})
    assert ex["input_ids"] == [10, 11, 12]
    print("encode truncation: OK")


def test_encode_messages():
    def render(sample):
        # a fake template: user turn masked, assistant turn supervised
        m = sample["messages"]
        return [(m[0]["content"], False), (m[1]["content"], True)]

    rt = make_rt(render_segments=render)
    ex = rt._encode({"messages": [{"role": "user", "content": "ab"},
                                  {"role": "assistant", "content": "c"}]})
    assert ex["input_ids"] == [1, ord("a"), ord("b"), ord("c")]
    assert ex["labels"] == [-100, -100, -100, ord("c")]

    rt = make_rt()   # no renderer
    try:
        rt._encode({"messages": []})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "render_segments" in str(e)
    print("encode messages: OK")


# ---------------------------------------------------------------------------
# Ingest loop
# ---------------------------------------------------------------------------

def test_ingest_batching_and_callbacks():
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=2, grad_accum=2, lr=1e-3), net=net)
    fired = []
    rt.add_update_callback(lambda: fired.append(1))
    gen = StubGenerator(jobs=3)
    rt.attach_generator(gen)

    # 7 samples @ batch 2 -> 4 micro-batches -> 2 accumulation windows.
    samples = [{"text": chr(ord("a") + i)} for i in range(7)]
    stats = rt.ingest(samples)
    assert stats["steps"] == 2 and rt.step == 2
    assert stats["samples"] == 7 and rt.samples_seen == 7
    assert len(net.seen) == 4
    # in-order consumption: first micro-batch holds samples 0..1
    assert net.seen[0][0].shape[0] == 2
    assert net.seen[0][0][0].tolist()[:2] == [1, ord("a")]
    assert net.seen[3][0].shape[0] == 1                     # 7th sample alone
    # one adapter push + one cache nuke + generator drained first
    assert net.applied == 1
    assert gen.pagetable.resets == 1
    assert len(fired) == 1
    assert gen.jobs == 0

    # a second ingest continues the same run
    stats = rt.ingest([{"text": "z"}])
    assert stats["steps"] == 1 and rt.step == 3 and net.applied == 2

    # empty ingest is a no-op (no sync, no callbacks)
    stats = rt.ingest([])
    assert stats["steps"] == 0 and net.applied == 2
    print("ingest batching/callbacks: OK")


def test_ingest_token_weighted_loss():
    # Two micro-batches in one window with 1 and 3 supervised (shifted)
    # tokens and losses 2.0 / 4.0: the token-weighted mean is
    # (2*1 + 4*3) / 4 = 3.5, not the plain mean 3.0.
    net = StubNet(losses=[2.0, 4.0])
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=2), net=net)
    stats = rt.ingest([
        {"input_ids": [5, 6], "labels": [5, 6]},              # 1 shifted sup
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},  # 3 shifted sup
    ])
    assert stats["steps"] == 1
    assert stats["supervised_tokens"] == 4
    assert abs(stats["mean_loss"] - 3.5) < 1e-6
    print("ingest token-weighted loss: OK")


def test_lr_control():
    rt = make_rt(RealtimeConfig(lr=1e-4))
    assert abs(rt.lr - 1e-4) < 1e-12
    rt.lr = 5e-5
    assert all(abs(g["lr"] - 5e-5) < 1e-12 for g in rt.opt.param_groups)
    rt.ingest([{"text": "a"}], lr=3e-5)
    assert abs(rt.lr - 3e-5) < 1e-12       # per-call override sticks
    print("lr control: OK")


def test_ingest_blocks_inference():
    """While an ingest runs, rt.inference() must wait; after it returns the
    adapter has been applied."""
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=1), net=net)
    in_ingest = threading.Event()
    release = threading.Event()
    orig_apply = net.apply_to_native

    def slow_apply(scaling=1.0):
        in_ingest.set()
        release.wait(2)
        orig_apply(scaling)

    net.apply_to_native = slow_apply
    t = threading.Thread(target=rt.ingest, args=([{"text": "a"}],))
    t.start()
    in_ingest.wait(2)
    got_in = {"v": False}

    def reader():
        with rt.inference():
            got_in["v"] = True

    r = threading.Thread(target=reader)
    r.start()
    time.sleep(0.05)
    assert not got_in["v"]                 # blocked while ingest holds write
    release.set()
    t.join(2)
    r.join(2)
    assert got_in["v"] and net.applied == 1
    print("ingest blocks inference: OK")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_naming_and_listing():
    now = time.localtime(1700000000)
    name = rt_mod.checkpoint_name(42, now)
    ts = time.strftime("%Y%m%d-%H%M%S", now)
    assert name == f"ckpt-{ts}-step42"
    print("checkpoint naming: OK")


def test_checkpoint_cadence_prune_resume():
    with tempfile.TemporaryDirectory() as d:
        net = StubNet()
        cfg = RealtimeConfig(batch_size=1, grad_accum=1, checkpoint_dir=d,
                             checkpoint_every=2, keep_checkpoints=2, lr=1e-3)
        rt = make_rt(cfg, net=net)
        stats = rt.ingest([{"text": chr(ord("a") + i)} for i in range(6)])
        # 6 steps @ every-2 -> checkpoints at steps 2, 4, 6; pruned to last 2
        assert len(stats["checkpoints"]) == 3
        names = rt_mod.list_realtime_checkpoints(d)
        assert len(names) == 2
        assert names[-1].endswith("-step6")
        last = os.path.join(d, names[-1])
        assert os.path.exists(os.path.join(last, "realtime_trainer_state.pt"))
        with open(os.path.join(last, "realtime_meta.json")) as f:
            meta = json.load(f)
        assert meta["step"] == 6 and meta["config"]["r"] == cfg.r

        # Resume into a fresh coordinator: step counter and optimizer state
        # come back; the adapter is pushed to inference immediately.
        net2 = StubNet()
        rt2 = make_rt(RealtimeConfig(lr=1e-3), net=net2)
        rt2.load(last)
        assert rt2.step == 6 and rt2.samples_seen == 6
        assert net2.loaded == [last]
        assert net2.applied == 1
        # optimizer state actually restored (AdamW step count is per-param)
        st = rt2.opt.state_dict()["state"]
        assert st and all(int(s["step"]) == 6 for s in st.values())
        print("checkpoint cadence/prune/resume: OK")


def test_manual_checkpoint_requires_dir():
    rt = make_rt()
    try:
        rt.checkpoint()
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("manual checkpoint guard: OK")


class OffloadStubNet(StubNet):
    """StubNet that also implements the idle-offload surface, recording the
    interleaving so tests can assert restore-before-train / offload-after-sync
    ordering, and that training never runs while offloaded."""

    def __init__(self, losses=None):
        super().__init__(losses)
        self.offloaded = False
        self.events = []

    def offload_training_state(self):
        assert not self.offloaded
        self.offloaded = True
        self.events.append("offload")
        return True

    def restore_training_state(self):
        self.offloaded = False
        self.events.append("restore")

    def compute_loss(self, *args, **kwargs):
        assert not self.offloaded, "trained while training state was offloaded"
        self.events.append("loss")
        return super().compute_loss(*args, **kwargs)

    def apply_to_native(self, scaling=1.0):
        self.events.append("apply")
        super().apply_to_native(scaling)

    def load_adapter(self, directory):
        assert not self.offloaded, "load_adapter while training state offloaded"
        self.events.append("load")
        return super().load_adapter(directory)


def test_idle_offload_lifecycle():
    net = OffloadStubNet()
    rt = make_rt(net=net)                   # offload_when_idle defaults on
    # Parked immediately at construction (serving footprint from the start).
    assert net.events == ["offload"] and rt._idle_offloaded

    rt.ingest([{"text": "ab"}])
    # Restored before training, trained, pushed to inference, re-parked.
    assert net.events == ["offload", "restore", "loss", "apply", "offload"]
    assert net.offloaded and rt._idle_offloaded

    # load() un-parks around the state surgery and re-parks after the sync.
    with tempfile.TemporaryDirectory() as d:
        cfg_dir = os.path.join(d, "ck")
        net.save_adapter(cfg_dir)
        net.events.clear()
        rt.load(cfg_dir)
        assert net.events == ["restore", "load", "apply", "offload"]
    print("idle offload lifecycle: OK")


def test_idle_offload_disabled():
    net = OffloadStubNet()
    rt = make_rt(RealtimeConfig(offload_when_idle=False), net=net)
    rt.ingest([{"text": "ab"}])
    assert not rt._idle_offloaded
    assert "offload" not in net.events and "restore" not in net.events
    print("idle offload disabled: OK")


def test_idle_offload_plain_stub():
    # A net without the offload surface (the DI contract) must still work:
    # the coordinator only moves optimizer state, which is a CPU no-op here.
    net = StubNet()
    rt = make_rt(net=net)
    assert rt._idle_offloaded
    stats = rt.ingest([{"text": "ab"}])
    assert stats["steps"] == 1 and rt._idle_offloaded
    assert rt._opt_state_homes == []        # nothing was on an accelerator
    print("idle offload without net surface: OK")


def test_unload_reload():
    net = StubNet()
    rt = make_rt(net=net)
    fired = []
    rt.add_update_callback(lambda: fired.append(1))
    rt.unload_from_inference()             # never synced -> no-op
    assert net.removed == 0 and not fired
    rt.ingest([{"text": "a"}])
    assert net.applied == 1 and len(fired) == 1
    rt.unload_from_inference()
    assert net.removed == 1 and len(fired) == 2
    rt.sync_to_inference()
    assert net.applied == 2 and len(fired) == 3
    print("unload/reload: OK")


if __name__ == "__main__":
    test_rwlock()
    test_encode_prompt_response()
    test_encode_text_and_pretokenized()
    test_encode_truncation()
    test_encode_messages()
    test_ingest_batching_and_callbacks()
    test_ingest_token_weighted_loss()
    test_lr_control()
    test_ingest_blocks_inference()
    test_checkpoint_naming_and_listing()
    test_checkpoint_cadence_prune_resume()
    test_manual_checkpoint_requires_dir()
    test_idle_offload_lifecycle()
    test_idle_offload_disabled()
    test_idle_offload_plain_stub()
    test_unload_reload()
    print("\nALL OK")
