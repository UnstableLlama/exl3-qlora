"""CPU tests for training/lora_capacity.py and its report integration.

No GPU/torch needed. Run: python tests/test_lora_capacity.py
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "training"))
import lora_capacity as lc  # noqa: E402
from run_report import RunLogger, REPORT_SUBDIR, _load_run  # noqa: E402


def test_capacity_matches_published_figures():
    # Figures from the post for Qwen3.5-0.8B: v_proj is 512x1024.
    full = lc.mp_capacity_bits(512, 1024, 512)
    assert abs(full - 512 * 1024 / (2 * lc.LN2)) < 1e-6        # ~378k bits
    assert 377e3 < full < 379e3
    # rank 128: 0.97 bits per LoRA parameter, 23.9 KB.
    c128 = lc.mp_capacity_bits(512, 1024, 128)
    assert abs(c128 / (128 * (512 + 1024)) - 0.97) < 0.01
    assert lc.fmt_bits(c128) == "23.9 KB"
    # Low-rank limit: 1.40 (v_proj) and 1.32 (up_proj, 1024x3584) bits/param.
    assert abs(lc.mp_capacity_bits(512, 1024, 1) / 1536 - 1.40) < 0.03
    assert abs(lc.mp_capacity_bits(1024, 3584, 1) / 4608 - 1.32) < 0.03


def test_capacity_shape_properties():
    # Orientation doesn't matter; monotone and concave in rank; capped at full.
    assert lc.mp_capacity_bits(512, 1024, 16) == lc.mp_capacity_bits(1024, 512, 16)
    vals = [lc.mp_capacity_bits(256, 256, r) for r in (1, 2, 4, 8, 16, 64, 256, 999)]
    assert all(b > a for a, b in zip(vals[:-2], vals[1:-1]))
    assert vals[-1] == vals[-2] == 256 * 256 / (2 * lc.LN2)
    assert vals[1] < 2 * vals[0] and vals[5] < 4 * vals[4]     # diminishing returns
    assert lc.mp_capacity_bits(512, 1024, 0) == 0.0


def test_range_and_rank_hint():
    shapes = [(512, 1024, 4)] * 24
    lo, hi = lc.capacity_range_bits(shapes)
    assert abs(lo - 24 * lc.mp_capacity_bits(512, 1024, 4)) < 1e-6
    assert abs(hi - lo * lc.JOINT_BONUS) < 1e-6
    # A single matrix gets no joint-training bonus.
    one_lo, one_hi = lc.capacity_range_bits([(512, 1024, 4)])
    assert one_lo == one_hi

    target = 100 * lc.BITS_PER_KB
    opt, cons = lc.rank_for_bits(shapes, target)
    assert opt < cons
    # Each hint is the SMALLEST rank that reaches the target.
    assert lc.additive_capacity_bits(shapes, cons) >= target
    assert lc.additive_capacity_bits(shapes, cons - 1) < target
    assert lc.additive_capacity_bits(shapes, opt) * lc.JOINT_BONUS >= target
    assert lc.additive_capacity_bits(shapes, opt - 1) * lc.JOINT_BONUS < target
    # More than full rank can hold -> None; no adapters -> (None, None).
    assert lc.rank_for_bits(shapes, 1e12) == (None, None)
    assert lc.rank_for_bits([], target) == (None, None)


def _curve(b, tau, c, steps):
    return [(t, b * t ** c / (t ** c + tau ** c)) for t in steps]


def test_fit_recovers_asymptote():
    fit = lc.fit_log_logistic(_curve(164.0, 900.0, 0.8, range(10, 4000, 10)))
    assert abs(fit["asymptote"] - 164.0) / 164.0 < 0.03
    assert abs(fit["tau"] - 900.0) / 900.0 < 0.1
    assert abs(fit["c"] - 0.8) < 0.05
    # Too few points, or nothing stored, is not a fit.
    assert lc.fit_log_logistic([(1, 1.0), (2, 2.0)]) is None
    assert lc.fit_log_logistic([(t, 0.0) for t in range(1, 50)]) is None


def test_verdicts():
    cap_bound = lc.fit_log_logistic(_curve(100.0, 300.0, 1.0, range(10, 4000, 10)))
    assert lc.capacity_verdict(cap_bound, 400.0, 4000).startswith("capacity-bound")
    assert lc.capacity_verdict(cap_bound, 100.0, 4000).startswith("step-bound")
    done = lc.fit_log_logistic(_curve(100.0, 30.0, 2.0, range(10, 4000, 10)))
    assert lc.capacity_verdict(done, 100.0, 4000).startswith("dataset fits")
    early = lc.fit_log_logistic(_curve(100.0, 5000.0, 1.0, range(10, 400, 10)))
    assert lc.capacity_verdict(early, 100.0, 400).startswith("too early")
    assert lc.capacity_verdict(None, 100.0, 400) is None
    assert lc.capacity_verdict(cap_bound, None, 4000) is None


def test_report_projects_from_live_run():
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "cap-run", config={"r": 8})
        rep.update_summary({"dataset_kb": 400.0}, flush=True)
        # summary.json exists BEFORE finish, so a live/crashed run can be sized.
        spath = os.path.join(out, REPORT_SUBDIR, "summary.json")
        assert json.load(open(spath))["dataset_kb"] == 400.0
        for t, y in _curve(100.0, 300.0, 1.0, range(10, 4000, 10)):
            rep.log({"capacity/stored_kb": y, "train/loss": 1.0}, step=t)
        run = _load_run(out)
        sm = run["summary"]
        assert abs(sm["stored_projected_kb"] - 100.0) < 5.0
        assert abs(sm["stored_frac"] - sm["stored_kb"] / 400.0) < 1e-9
        assert sm["capacity_verdict"].startswith("capacity-bound")
        rep.finish(exit_code=0)
        html = open(os.path.join(out, REPORT_SUBDIR, "report.html")).read()
        assert "capacity-bound" in html and "capacity/stored_kb" in html


def test_report_without_capacity_metrics_is_unchanged():
    with tempfile.TemporaryDirectory() as out:
        rep = RunLogger(out, "plain", config={})
        for s in range(1, 20):
            rep.log({"train/loss": 2.0 - 0.01 * s}, step=s)
        rep.update_summary({"best_val": 1.5})
        assert _load_run(out)["summary"] == {}          # not flushed, nothing derived
        rep.finish(exit_code=0)
        assert _load_run(out)["summary"] == {"best_val": 1.5}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
