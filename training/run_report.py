"""Self-contained local run logging + HTML report -- the default replacement for
wandb on the exl3-qlora trainers.

Why this exists: the wandb free tier under a team entity can't host public
projects, and sharing training runs publicly is a requirement. So instead of a
third-party platform, each run writes its own data to disk and renders a single
self-contained ``report.html`` (inline CSS + vanilla-JS SVG charts, **no CDN**,
no account) that opens locally or hosts anywhere.

Layout, all under the run's ``out`` dir so the report travels with the adapter::

    <out>/run_report/
        config.json     # run identity + hyperparameters (mirrors the CSV row)
        metrics.jsonl   # one JSON object per log() call: {"step": N, "<metric>": v, ...}
        summary.json    # final scalars (best val, throughput, peak VRAM, ...)
        report.html     # rendered from the three files above

Metric names use wandb-style ``group/name`` keys (``train/loss``,
``eval/held_out``, ``perf/tot_tok_s``); the report groups charts by the prefix.
The HTML is rebuildable from ``metrics.jsonl`` alone, so a crashed run still
renders whatever it logged (``render_report(out_dir)`` regenerates in place).

Comparison: ``compare_reports([outA, outB, ...])`` overlays several runs on one
page -- each metric chart draws one colored line per run with a shared legend --
for A/B runs (e.g. SFT vs EBFT). Same from the CLI::

    python training/run_report.py <out_dir>                 # (re)render one run
    python training/run_report.py <outA> <outB> -o cmp.html # overlay runs

Live monitor: the trainers' ``--live-report`` flag calls
:func:`start_live_monitor`, which serves the SAME template from a localhost
http.server thread with a ``LIVE`` config injected. The live page polls
``/metrics`` (the growing metrics.jsonl) to redraw charts during the run, and
adds a training-batch viewer driven by ``/batch?step=N``: the trainer-side
``batch_fn`` recomputes which examples step N consumes (the data order is
deterministic given the seed) and decodes their text on demand, so the browser
can page through past AND future steps. Nothing about the dataset is written to
disk or into ``report.html`` -- the shareable artifact stays dataset-free; the
live view exists only while the trainer process (and its in-memory examples)
is alive.

Drop-in shape mirrors the wandb calls it replaces::

    rep = RunLogger(out_dir, run_name, config)     # ~ wandb.init(...)
    rep.log({"train/loss": x}, step=step)          # ~ run.log(..., step=step)
    rep.summary.update({"best_val": v})            # ~ run.summary.update(...)
    rep.finish(exit_code=0)                        # ~ run.finish(...) -> writes report.html
"""

import datetime
import html as _html
import json
import math
import os


REPORT_SUBDIR = "run_report"


def _report_dir(out_dir):
    return os.path.join(out_dir, REPORT_SUBDIR)


def _json_safe(v):
    """Best-effort coerce a value into something json.dump can write."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    # torch scalars / numpy floats expose .item(); fall back to str.
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


class RunLogger:
    """Local, self-contained run logger. One instance per training run.

    Streams every ``log()`` call to ``metrics.jsonl`` (flushed each write, so a
    hard kill still leaves the rows already logged) and renders ``report.html``
    on ``finish()``. ``finish()`` is idempotent -- the first caller (normal
    finish, Ctrl-C, or the failure logger) wins -- matching the wandb teardown
    it stands in for.
    """

    def __init__(self, out_dir, run_name, config=None, meta=None):
        self.out_dir = out_dir
        self.dir = _report_dir(out_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.run_name = run_name
        self.summary = {}
        self._closed = False
        self._meta = {
            "run_name": run_name,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        if meta:
            self._meta.update(meta)
        # Fresh metrics stream each run (an --out reuse starts a new report).
        self._metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self._fh = open(self._metrics_path, "w", encoding="utf-8")
        self._write_json("config.json", {
            "meta": self._meta,
            "config": {k: _json_safe(v) for k, v in (config or {}).items()},
        })

    def _write_json(self, name, obj):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False)

    def log(self, metrics, step=None):
        """Append one metrics row. Keys are wandb-style ``group/name`` strings."""
        if self._closed or self._fh is None:
            return
        row = {"step": step}
        for k, v in metrics.items():
            row[k] = _json_safe(v)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def update_summary(self, d):
        self.summary.update({k: _json_safe(v) for k, v in d.items()})

    def finish(self, exit_code=0, status=None):
        if self._closed:
            return
        self._closed = True
        self._meta["status"] = status or ("completed" if exit_code == 0 else "failed")
        self._meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._meta["exit_code"] = exit_code
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        # Re-persist config (meta now has final status) and the summary.
        cfg = _read_json(os.path.join(self.dir, "config.json")) or {}
        cfg["meta"] = self._meta
        self._write_json("config.json", cfg)
        self._write_json("summary.json", self.summary)
        try:
            render_report(self.out_dir)
            print(f"[report] {os.path.join(self.dir, 'report.html')}")
        except Exception as exc:  # a render bug must never mask the run's own exit
            print(f"[report] render failed: {exc}")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_metrics(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # skip a torn final line from a hard kill
    except OSError:
        return []
    return rows


def _load_run(out_dir, label=None):
    """Read one run's config/metrics/summary off disk into the payload shape the
    HTML template consumes: {meta, config, summary, series}. ``series`` maps each
    metric name to its [[step, value], ...] points (non-finite values dropped)."""
    rdir = _report_dir(out_dir)
    cfg = _read_json(os.path.join(rdir, "config.json")) or {}
    meta = dict(cfg.get("meta", {}))
    config = cfg.get("config", {})
    summary = _read_json(os.path.join(rdir, "summary.json")) or {}
    rows = _read_metrics(os.path.join(rdir, "metrics.jsonl"))

    series = {}
    for row in rows:
        step = row.get("step")
        for k, v in row.items():
            if k == "step":
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
                continue
            series.setdefault(k, []).append([step, v])

    meta["label"] = (label if label is not None
                     else meta.get("run_name")
                     or os.path.basename(os.path.normpath(out_dir)))
    return {"meta": meta, "config": config, "summary": summary, "series": series}


# --------------------------------------------------------------------------- #
# Static pre-render. The report is normally drawn by the inline JS at load time,
# but some viewers run the file with JavaScript disabled -- iOS Files/Quick Look,
# Mail/Messages/AirDrop attachment previews, in-app browsers. There the empty
# JS-target containers show nothing. So we ALSO render the header/summary/charts/
# config to static HTML+SVG in Python and inject it into those containers. When
# JS *does* run it clears and redraws them (identical output, plus interactivity)
# -- the static pass is a no-JS fallback, the JS is progressive enhancement. The
# two paths mirror each other on purpose; keep them in sync when either changes.
# --------------------------------------------------------------------------- #

_PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]


def _e(v):
    """HTML-escape (incl. quotes) any value to a str -- mirrors the JS setText."""
    return _html.escape(str(v), quote=True)


def _fmt(v):
    """Number formatting matching the JS ``fmt()`` so static and JS agree."""
    if v is None:
        return "–"  # en dash
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or (0 < a < 1e-3):  # JS toExponential(2), e.g. "1.23e+6"
        mant, exp = ("%.2e" % v).split("e")
        ei = int(exp)
        return "%se%s%d" % (mant, "+" if ei >= 0 else "-", abs(ei))
    if isinstance(v, int) or float(v).is_integer():  # JS Number.isInteger -> toLocaleString
        return "{:,}".format(int(v))
    if a >= 100:
        return "%.1f" % v
    return "%.4f" % v


def _union_keys(objs):
    """Keys across dicts, first-seen order -- mirrors the JS ``unionKeys()``."""
    seen, out = set(), []
    for o in objs:
        for k in (o or {}):
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _xticks(mn, mx, count=5):
    """Port of the JS ``xticks()`` -- round interior ticks, endpoints kept."""
    mn, mx = round(mn), round(mx)
    if mn == mx:
        return [mn]
    span = mx - mn
    raw = span / (count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag
    t = {mn, mx}
    v = math.ceil(mn / step) * step
    while v < mx:
        if v > mn:
            t.add(round(v))
        v += step
    arr = sorted(t)
    tol = span * 0.08
    return [x for i, x in enumerate(arr)
            if i == 0 or i == len(arr) - 1 or (x - mn > tol and mx - x > tol)]


def _sw(color):
    return '<span class="sw" style="background:%s"></span>' % color


def _svg_chart(series_list):
    """Static SVG for one chart: grid, axes, one polyline per series. The
    interactive dots/crosshair/tooltip are added only by the JS on hover."""
    W, H, PL, PR, PT, PB = 360, 158, 44, 12, 10, 26
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for s in series_list:
        for p in s["points"]:
            xmin = min(xmin, p[0]); xmax = max(xmax, p[0])
            ymin = min(ymin, p[1]); ymax = max(ymax, p[1])
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        pad = abs(ymin) or 1
        ymin -= pad * 0.05; ymax += pad * 0.05
    pady = (ymax - ymin) * 0.08
    ymin -= pady; ymax += pady

    def sx(v):
        return PL + (v - xmin) / (xmax - xmin) * (W - PL - PR)

    def sy(v):
        return PT + (1 - (v - ymin) / (ymax - ymin)) * (H - PT - PB)

    out = ['<svg viewBox="0 0 %d %d" width="%d" height="%d" '
           'preserveAspectRatio="xMidYMid meet">' % (W, H, W, H)]
    for i in range(4):
        yv = ymin + (ymax - ymin) * i / 3
        y = sy(yv)
        out.append('<line class="grid-line" x1="%d" y1="%.3f" x2="%d" y2="%.3f"/>'
                   % (PL, y, W - PR, y))
        out.append('<text class="axis-txt" x="%d" y="%.3f" text-anchor="end">%s</text>'
                   % (PL - 5, y + 3, _e(_fmt(yv))))
    for xv in _xticks(xmin, xmax, 5):
        x = sx(xv)
        if xmin < xv < xmax:
            out.append('<line class="grid-line" x1="%.3f" y1="%d" x2="%.3f" y2="%d"/>'
                       % (x, PT, x, H - PB))
        out.append('<text class="axis-txt" x="%.3f" y="%d" text-anchor="middle">%s</text>'
                   % (x, H - 8, _e(str(xv))))
    for s in series_list:
        d = " ".join(("L" if i else "M") + "%.1f %.1f" % (sx(p[0]), sy(p[1]))
                     for i, p in enumerate(s["points"]))
        out.append('<path class="plot" d="%s" stroke="%s"/>' % (d, s["color"]))
    out.append("</svg>")
    return "".join(out)


def _chart_box(name, series_list, multi):
    cur = ("" if multi else
           '<span class="cur">last %s</span>'
           % _e(_fmt(series_list[0]["points"][-1][1])))
    return ('<div class="chart"><div class="title"><span>%s</span>%s</div>'
            '<div class="svg-holder">%s</div></div>'
            % (_e(name), cur, _svg_chart(series_list)))


def _split_scales(series_list):
    """Port of the JS ``splitScales()`` -- true when series sit on incomparable
    y-scales and should each get their own chart rather than be overlaid."""
    if len(series_list) < 2:
        return False
    ulo = float("inf"); uhi = float("-inf"); spans = []
    for s in series_list:
        lo = float("inf"); hi = float("-inf")
        for p in s["points"]:
            lo = min(lo, p[1]); hi = max(hi, p[1])
        ulo = min(ulo, lo); uhi = max(uhi, hi)
        spans.append(hi - lo)
    union = uhi - ulo
    nz = [sp for sp in spans if sp > 0]
    if not (union > 0) or not nz:
        return False
    return min(nz) / union < 0.10


def _charts_html(runs, multi):
    group_order = ["train", "eval", "perf"]
    metric_names = _union_keys([r["series"] for r in runs])
    groups = {}
    for k in metric_names:
        g = k.split("/")[0] if "/" in k else "other"
        groups.setdefault(g, []).append(k)

    def gkey(g):
        return (group_order.index(g) if g in group_order else 99, g)

    out = []
    for g in sorted(groups, key=gkey):
        for metric in sorted(groups[g]):
            series_list = []
            for r in runs:
                pts = [p for p in r["series"].get(metric, []) if p[0] is not None]
                if pts:
                    series_list.append({"label": r["meta"]["label"],
                                        "color": r["color"], "points": pts})
            if not series_list:
                continue
            if multi and _split_scales(series_list):
                for s in series_list:
                    out.append(_chart_box(metric + " — " + s["label"], [s], False))
            else:
                out.append(_chart_box(metric, series_list, multi))
    return "".join(out) or '<div class="sub">no metrics logged</div>'


def _summary_html(runs, multi):
    if not multi:
        sm = runs[0]["summary"] or {}
        if not sm:
            return ""
        cards = "".join(
            '<div class="card"><div class="k">%s</div><div class="v">%s</div></div>'
            % (_e(k), _e(_fmt(sm[k]))) for k in sm)
        return '<div class="cards">%s</div>' % cards
    skeys = _union_keys([r["summary"] for r in runs])
    if not skeys:
        return ""
    head = ('<tr><th class="k"></th>'
            + "".join('<th>%s<span>%s</span></th>' % (_sw(r["color"]), _e(r["meta"]["label"]))
                      for r in runs) + "</tr>")
    rows = "".join(
        '<tr><td class="k">%s</td>%s</tr>'
        % (_e(k), "".join('<td>%s</td>' % _e(_fmt((r["summary"] or {}).get(k)))
                          for r in runs))
        for k in skeys)
    return '<div class="cmp-wrap"><table class="cmp">%s%s</table></div>' % (head, rows)


def _config_html(runs, multi):
    if not multi:
        cfg = runs[0]["config"] or {}
        rows = "".join(
            '<tr><td class="k">%s</td><td class="v">%s</td></tr>'
            % (_e(k), _e(_fmt(cfg[k]))) for k in cfg)
        return '<div class="cfg-wrap"><table class="cfg">%s</table></div>' % rows
    ckeys = _union_keys([r["config"] for r in runs])
    head = ('<tr><th class="k"></th>'
            + "".join('<th>%s<span>%s</span></th>' % (_sw(r["color"]), _e(r["meta"]["label"]))
                      for r in runs) + "</tr>")
    rows = []
    for k in ckeys:
        vals = [(r["config"] or {}).get(k) for r in runs]
        differ = any(json.dumps(v, sort_keys=True) != json.dumps(vals[0], sort_keys=True)
                     for v in vals)
        cells = "".join('<td>%s</td>' % _e(_fmt(v)) for v in vals)
        rows.append('<tr%s><td class="k">%s</td>%s</tr>'
                    % (' class="diff"' if differ else "", _e(k), cells))
    return ('<div class="cmp-wrap"><table class="cmp">%s%s</table></div>'
            % (head, "".join(rows)))


def _header_sections(runs, multi):
    """Return (title, subtitle_html, legend_html) mirroring the JS header code."""
    if multi:
        chips = ""
        for r in runs:
            st = (r["meta"].get("status") or "").lower()
            badge = '<span class="badge b-%s">%s</span>' % (_e(st), _e(st)) if st else ""
            chips += ('<span class="chip">%s<span>%s</span>%s</span>'
                      % (_sw(r["color"]), _e(r["meta"]["label"]), badge))
        return "Comparison — %d runs" % len(runs), "", '<div class="legend">%s</div>' % chips
    m = runs[0]["meta"]
    cfg0 = runs[0]["config"] or {}
    status = (m.get("status") or "running").lower()
    dur = None
    if m.get("started") and m.get("finished"):
        try:
            dur = round((datetime.datetime.fromisoformat(m["finished"])
                         - datetime.datetime.fromisoformat(m["started"])).total_seconds())
        except ValueError:
            dur = None
    parts = [cfg0.get("model") or cfg0.get("model_name") or "",
             ("started " + m["started"].replace("T", " ")) if m.get("started") else "",
             ("%ds wall" % dur) if dur is not None else ""]
    sub = "  ·  ".join(p for p in parts if p) + "  "
    subtitle = _e(sub) + '<span class="badge b-%s">%s</span>' % (_e(status), _e(status))
    return (m.get("run_name") or "training run"), subtitle, ""


def _static_sections(runs_raw):
    """Pre-render every JS-populated container to static HTML. Mirrors the JS
    ``prep()`` (per-run color + duplicate-label disambiguation) then each render
    branch. Returns the {token: html} map injected by ``_write_html``."""
    runs = []
    for i, r in enumerate(runs_raw):
        runs.append({"meta": dict(r.get("meta") or {}), "config": r.get("config") or {},
                     "summary": r.get("summary") or {}, "series": r.get("series") or {},
                     "color": _PALETTE[i % len(_PALETTE)]})
    seen = {}
    for r in runs:
        base = r["meta"].get("label") or r["meta"].get("run_name") or "run"
        seen[base] = seen.get(base, 0) + 1
        r["meta"]["label"] = base + " #" + str(seen[base]) if seen[base] > 1 else base
    if not runs:  # nothing to draw; leave containers empty
        return {k: "" for k in ("__STATIC_TITLE__", "__STATIC_SUBTITLE__",
                                "__STATIC_LEGEND__", "__STATIC_SUMMARY__",
                                "__STATIC_CHARTS__", "__STATIC_CONFIG__")}
    multi = len(runs) > 1
    title, subtitle, legend = _header_sections(runs, multi)
    return {
        "__STATIC_TITLE__": _e(title),
        "__STATIC_SUBTITLE__": subtitle,
        "__STATIC_LEGEND__": legend,
        "__STATIC_SUMMARY__": _summary_html(runs, multi),
        "__STATIC_CHARTS__": _charts_html(runs, multi),
        "__STATIC_CONFIG__": _config_html(runs, multi),
    }


def _render_html(payload, live_info=None):
    html = _HTML_TEMPLATE.replace(
        "/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    if live_info is not None:
        # Static reports keep the template's literal null -> LIVE mode stays off.
        html = html.replace("/*__LIVE__*/null",
                            json.dumps(live_info, separators=(",", ":")))
    for token, frag in _static_sections(payload.get("runs") or []).items():
        html = html.replace(token, frag)
    return html


def _write_html(payload, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_render_html(payload))
    return out_path


def render_report(out_dir):
    """(Re)render ``<out>/run_report/report.html`` from the on-disk config/
    metrics/summary. Usable standalone to regenerate a report (e.g. after a
    crash left only the jsonl)."""
    payload = {"runs": [_load_run(out_dir)]}
    return _write_html(payload, os.path.join(_report_dir(out_dir), "report.html"))


def compare_reports(out_dirs, output_path=None, labels=None):
    """Render one HTML page overlaying several runs for comparison. Each metric
    chart draws a colored line per run with a shared legend; summary and config
    become per-run columns (differing config rows highlighted). Runs with no
    metrics are skipped with a warning."""
    runs = []
    for i, d in enumerate(out_dirs):
        lbl = labels[i] if labels and i < len(labels) else None
        run = _load_run(d, label=lbl)
        if not run["series"]:
            print(f"[report] warning: no metrics under {d}, skipping")
            continue
        runs.append(run)
    if not runs:
        raise SystemExit("no runs with metrics to compare")
    # Disambiguate duplicate labels (e.g. two runs both basename 'exl3_ebft').
    seen = {}
    for r in runs:
        lbl = r["meta"]["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        if seen[lbl] > 1:
            r["meta"]["label"] = f"{lbl} #{seen[lbl]}"
    output_path = output_path or "compare_report.html"
    _write_html({"runs": runs}, output_path)
    print(f"[report] {output_path}  ({len(runs)} runs)")
    return output_path


# --------------------------------------------------------------------------- #
# Live monitor (--live-report). A daemon http.server thread inside the trainer
# process serves the report template in LIVE mode plus two data endpoints. The
# batch texts are never persisted anywhere -- /batch decodes them on demand from
# the trainer's in-memory examples, so the on-disk report.html stays shareable
# without the dataset and the live view dies with the process.
# --------------------------------------------------------------------------- #

def decode_example_docs(tokenizer, example):
    """Decode one training example into per-document prompt/response texts.

    ``example`` is a trainer example dict (``input_ids`` / ``labels``, plus
    ``seg_ids`` for packed blocks, which are split back into their documents
    here). Each document is returned as the alternating run of masked and
    supervised spans it actually is -- a multi-turn conversation has several of
    each, since the follow-up user turns and the assistant headers sit MASKED
    between the supervised replies. Collapsing to a single prompt + response
    would drop every interior masked span and render a multi-turn row as its
    answers concatenated back to back, which reads like corrupted data.
    Trailing pads (label -100 after the last supervised token) are cut, so
    packed-block padding never shows. Returns
    ``[{segments: [{text, sup, n}, ...], n_prompt, n_sup}, ...]``.
    """
    import torch  # deferred: the render/CLI paths stay torch-free
    ids, labels = example["input_ids"], example["labels"]
    seg = example.get("seg_ids")
    if seg is None:
        spans = [(0, len(ids))]
    else:
        spans, start = [], 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or seg[i] != seg[i - 1]:
                spans.append((start, i))
                start = i

    def dec(t):
        if not t:
            return ""
        return tokenizer.decode(torch.tensor(t, dtype=torch.long),
                                decode_special_tokens=True)

    docs = []
    for a, b in spans:
        s_ids, s_labs = ids[a:b], labels[a:b]
        sup = [i for i, l in enumerate(s_labs) if l != -100]
        if not sup:
            continue  # fully-masked span (shouldn't happen; guard anyway)
        # Cut trailing pad so packed-block padding stays off the page.
        s_ids, s_labs = s_ids[:sup[-1] + 1], s_labs[:sup[-1] + 1]
        segments, run, in_sup = [], [], s_labs[0] != -100
        for t, l in zip(s_ids, s_labs):
            if (l != -100) != in_sup:
                segments.append({"text": dec(run), "sup": in_sup, "n": len(run)})
                run, in_sup = [], not in_sup
            run.append(t)
        segments.append({"text": dec(run), "sup": in_sup, "n": len(run)})
        docs.append({"segments": segments, "n_sup": len(sup),
                     "n_prompt": len(s_ids) - len(sup)})
    return docs


def start_live_monitor(out_dir, live_info, batch_fn=None, port=0,
                       open_browser=True):
    """Serve the live run monitor for the run logging into ``out_dir``.

    Endpoints (localhost only):
      * ``/``             -- the report template rendered from current on-disk
                             data with ``live_info`` injected as the JS LIVE
                             config ({total_steps, first_step, run_name, ...}).
      * ``/metrics``      -- raw metrics.jsonl (RunLogger flushes every log(),
                             so polling this tracks the run in real time).
      * ``/batch?step=N`` -- JSON from ``batch_fn(N)``: the decoded batch texts
                             for optimizer step N (past or future). Calls are
                             serialized with a lock (tokenizer decode runs in
                             the server thread while training continues).

    Returns the server object (daemon thread; dies with the trainer). A port of
    0 picks a free one. Failure to open the browser is non-fatal.
    """
    import threading
    import urllib.parse
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    rdir = _report_dir(out_dir)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the training console clean
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path in ("/", "/index.html"):
                    payload = {"runs": [_load_run(out_dir)]}
                    body = _render_html(payload, live_info=live_info)
                    self._send(200, body.encode("utf-8"),
                               "text/html; charset=utf-8")
                elif parsed.path == "/metrics":
                    try:
                        with open(os.path.join(rdir, "metrics.jsonl"), "rb") as f:
                            data = f.read()
                    except OSError:
                        data = b""
                    self._send(200, data, "text/plain; charset=utf-8")
                elif parsed.path == "/batch":
                    if batch_fn is None:
                        self._send(404, b"no batch provider", "text/plain")
                        return
                    q = urllib.parse.parse_qs(parsed.query)
                    try:
                        step = int(q.get("step", [""])[0])
                    except (ValueError, IndexError):
                        self._send(400, b"bad step", "text/plain")
                        return
                    try:
                        with lock:
                            data = batch_fn(step)
                    except Exception as exc:
                        self._send(400, str(exc).encode("utf-8"), "text/plain")
                        return
                    self._send(200, json.dumps(data).encode("utf-8"),
                               "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser navigated away mid-response

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name="live-report").start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"[live] run monitor at {url} (dies with the trainer; report.html "
          f"is the shareable artifact)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return srv


# --------------------------------------------------------------------------- #
# Self-contained HTML template. All CSS/JS inline; the run data is injected as a
# single JSON blob at /*__DATA__*/ in the shape {runs: [{meta,config,summary,
# series}, ...]}. One run -> single-run report (summary cards); several runs ->
# overlay-comparison report (legend + one line per run per chart, per-run
# summary/config columns). Charts are drawn to SVG by the vanilla JS below -- no
# external requests, so the file works offline and hosts anywhere. Always dark
# (single-run and comparison pages share this one template, so they match).
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>training run report</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1216; --panel: #171b21; --border: #262b33; --fg: #e6e9ee;
    --muted: #8a93a0; --grid: #21262e; --accent: #60a5fa;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    overflow-x: hidden;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }
  header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 15px; margin: 28px 0 12px; color: var(--muted);
       text-transform: uppercase; letter-spacing: .04em; }
  .sub { color: var(--muted); font-size: 13px; }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 12px; font-weight: 600; vertical-align: middle; }
  .b-completed { background: #10b98122; color: #10b981; }
  .b-failed, .b-interrupted { background: #ef444422; color: #ef4444; }
  .b-running { background: #f59e0b22; color: #f59e0b; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 4px; }
  .chip { display: inline-flex; align-items: center; gap: 7px; background: var(--panel);
          border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px; font-size: 13px; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(150px,100%),1fr));
           gap: 10px; margin: 8px 0 4px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 11px 13px; }
  .card .k { color: var(--muted); font-size: 12px; }
  .card .v { font-size: 18px; font-weight: 600; margin-top: 2px;
             font-variant-numeric: tabular-nums; word-break: break-word; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(340px,100%),1fr)); gap: 14px; }
  .chart { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: 12px 12px 6px; }
  .chart .title { font-size: 13px; font-weight: 600; margin-bottom: 2px;
                  display: flex; justify-content: space-between; align-items: baseline; }
  .chart .cur { color: var(--muted); font-weight: 400; font-variant-numeric: tabular-nums; }
  .svg-holder { position: relative; }
  svg { width: 100%; height: auto; display: block; }
  .grid-line { stroke: var(--grid); stroke-width: 1; }
  .axis-txt { fill: var(--muted); font-size: 10px; }
  .plot { fill: none; stroke-width: 1.6; }
  .dot { stroke: var(--bg); stroke-width: 1; }
  .crosshair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }
  .tt { position: absolute; pointer-events: none; background: var(--bg);
        border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
        font-size: 11px; line-height: 1.6; opacity: 0; transition: opacity .05s;
        white-space: nowrap; box-shadow: 0 2px 10px rgba(0,0,0,.18); z-index: 2; top: 6px; }
  .tt .sw { margin-right: 5px; vertical-align: baseline; }
  .tt .row { font-variant-numeric: tabular-nums; }
  table.cfg, table.cmp { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.cfg td { border-bottom: 1px solid var(--border); padding: 5px 10px 5px 0; vertical-align: top; }
  table.cfg td.k { color: var(--muted); white-space: nowrap; width: 1%; padding-right: 22px; }
  table.cfg td.v { font-variant-numeric: tabular-nums; word-break: break-word; }
  .cfg-wrap { columns: 2; column-gap: 34px; }
  @media (max-width: 640px) { .cfg-wrap { columns: 1; } }
  .cmp-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table.cmp th, table.cmp td { border-bottom: 1px solid var(--border); padding: 6px 16px 6px 0;
                               text-align: left; font-variant-numeric: tabular-nums; vertical-align: top; }
  table.cmp th { color: var(--fg); font-weight: 600; }
  table.cmp td.k, table.cmp th.k { color: var(--muted); font-weight: 400; white-space: nowrap; }
  table.cmp th .sw { margin-right: 6px; }
  tr.diff td { background: rgba(96,165,250,.12);  /* fallback for iOS < 16.2 (no color-mix) */
               background: color-mix(in srgb, var(--accent) 10%, transparent); }
  .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 12px; }
  .tbtn { font: inherit; font-size: 13px; cursor: pointer; background: var(--panel);
          color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; }
  .tbtn:hover { border-color: var(--accent); }
  #importMsg { margin-left: 2px; }
  .resume-panel { display: flex; flex-direction: column; gap: 8px; margin-top: 10px;
                  padding: 10px 12px; background: var(--panel); border: 1px solid var(--border);
                  border-radius: 8px; }
  .resume-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 13px; }
  .resume-row .rl { font-weight: 600; }
  .resume-panel select, .resume-panel input { font: inherit; font-size: 13px; background: var(--bg);
        color: var(--fg); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
  .resume-panel input[type=number] { width: 96px; }
  /* live-monitor batch viewer (hidden unless served with LIVE injected) */
  #liveBar { margin-top: 8px; font-size: 13px; }
  .pv-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px; }
  .pv-bar input[type=number] { width: 92px; font: inherit; font-size: 13px; background: var(--panel);
        color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; }
  .pv-bar label { display: inline-flex; align-items: center; gap: 5px; }
  .pv-mbh { color: var(--muted); font-size: 12px; text-transform: uppercase;
            letter-spacing: .04em; margin: 14px 0 6px; }
  .pv-seq { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
            padding: 10px 12px; margin-bottom: 8px; max-height: 320px; overflow-y: auto; }
  .pv-seqh { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  .pv-doc { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12.5px; line-height: 1.55;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            border-top: 1px dashed var(--border); padding-top: 6px; margin-top: 6px; }
  .pv-doc:first-of-type { border-top: none; padding-top: 0; margin-top: 0; }
  .pv-prompt { color: var(--muted); }
  .pv-resp { color: var(--fg); background: rgba(16,185,129,.14); border-radius: 3px; }
  .pv-tok { color: var(--muted); font-size: 11px; font-family: inherit; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title">__STATIC_TITLE__</h1>
    <div class="sub" id="subtitle">__STATIC_SUBTITLE__</div>
    <div id="legend">__STATIC_LEGEND__</div>
    <div class="toolbar">
      <button id="importBtn" class="tbtn" type="button">＋ Overlay another report…</button>
      <button id="resetBtn" class="tbtn" type="button" hidden>Reset</button>
      <input id="importInput" type="file" accept=".html,text/html" multiple hidden>
      <span class="sub" id="importMsg"></span>
    </div>
    <div id="resumePanel" class="resume-panel" hidden></div>
    <div id="liveBar" hidden></div>
  </header>
  <div id="summary">__STATIC_SUMMARY__</div>
  <div id="liveViewer" hidden>
    <h2>training data — step viewer</h2>
    <div class="pv-bar">
      <button id="pvPrev" class="tbtn" type="button">◀ prev</button>
      <input id="pvStep" type="number" step="1">
      <span class="sub" id="pvTotal"></span>
      <button id="pvNext" class="tbtn" type="button">next ▶</button>
      <label class="sub"><input id="pvFollow" type="checkbox" checked> follow live</label>
      <span class="sub" id="pvMsg"></span>
    </div>
    <div id="pvBody"></div>
  </div>
  <h2>metrics</h2>
  <div class="charts" id="charts">__STATIC_CHARTS__</div>
  <h2>config</h2>
  <div id="config">__STATIC_CONFIG__</div>
</div>
<script>
const DATA = /*__DATA__*/;
const LIVE = /*__LIVE__*/null;  // injected by the --live-report server; null in static files
const PALETTE = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#ec4899","#14b8a6","#f97316"];
const BASE_RUNS = (DATA.runs || []);
let RUNS = BASE_RUNS.slice();  // working set; overlay import appends, reset restores
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function el(tag, attrs, kids) {
  const svgTags = {svg:1, path:1, line:1, circle:1, text:1, g:1};
  const e = document.createElementNS(
    svgTags[tag] ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs||{})) {
    if (k === "html") e.innerHTML = attrs[k];
    else if (k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  (kids||[]).forEach(c => e.appendChild(c));
  return e;
}
function fmt(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString();
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(4);
}
function swatch(color) { const s = el("span", {class: "sw"}); s.setAttribute("style", "background:" + color); return s; }

// Nice x-axis ticks: endpoints always shown, plus a few interior ticks at round
// step values; interior ticks too close to an endpoint are dropped to avoid
// label overlap.
function xticks(min, max, count) {
  min = Math.round(min); max = Math.round(max);
  if (min === max) return [min];
  const span = max - min, raw = span / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const t = new Set([min, max]);
  for (let v = Math.ceil(min / step) * step; v < max; v += step)
    if (v > min) t.add(Math.round(v));
  const arr = [...t].sort((a, b) => a - b);
  const tol = span * 0.08;
  return arr.filter((v, i) => i === 0 || i === arr.length - 1 || (v - min > tol && max - v > tol));
}

function unionKeys(objs) {
  const seen = [], set = new Set();
  objs.forEach(o => Object.keys(o || {}).forEach(k => { if (!set.has(k)) { set.add(k); seen.push(k); } }));
  return seen;
}

// ---- render the whole page from a runs array; called on load and on every
// overlay import, so all chart scales recompute across the current run set ----
function render(runs) {
  const MULTI = runs.length > 1;

  // header
  const title = document.getElementById("title");
  const subtitle = document.getElementById("subtitle");
  const legendRoot = document.getElementById("legend");
  clear(legendRoot); title.textContent = ""; subtitle.textContent = "";
  if (MULTI) {
    title.textContent = "Comparison — " + runs.length + " runs";
    const leg = el("div", {class: "legend"});
    runs.forEach(r => {
      const st = (r.meta && r.meta.status || "").toLowerCase();
      const chip = el("span", {class: "chip"}, [swatch(r.color), el("span", {text: r.meta.label})]);
      if (st) chip.appendChild(el("span", {class: "badge b-" + st, text: st}));
      leg.appendChild(chip);
    });
    legendRoot.appendChild(leg);
  } else {
    const m = (runs[0] && runs[0].meta) || {};
    const cfg0 = (runs[0] && runs[0].config) || {};
    title.textContent = m.run_name || "training run";
    const status = (m.status || "running").toLowerCase();
    const dur = (m.started && m.finished)
      ? Math.round((new Date(m.finished) - new Date(m.started)) / 1000) : null;
    subtitle.textContent = [
      cfg0.model || cfg0.model_name || "",
      m.started ? ("started " + m.started.replace("T", " ")) : "",
      dur != null ? (dur + "s wall") : "",
    ].filter(Boolean).join("  ·  ") + "  ";
    subtitle.appendChild(el("span", {class: "badge b-" + status, text: status}));
  }

  // summary
  const summaryRoot = document.getElementById("summary"); clear(summaryRoot);
  if (!MULTI) {
    const sum = (runs[0] && runs[0].summary) || {};
    const kk = Object.keys(sum);
    if (kk.length) {
      const cards = el("div", {class: "cards"});
      kk.forEach(k => cards.appendChild(el("div", {class: "card"}, [
        el("div", {class: "k", text: k}), el("div", {class: "v", text: fmt(sum[k])})])));
      summaryRoot.appendChild(cards);
    }
  } else {
    const skeys = unionKeys(runs.map(r => r.summary));
    if (skeys.length) {
      const tbl = el("table", {class: "cmp"});
      const head = el("tr", {}, [el("th", {class: "k", text: ""})]);
      runs.forEach(r => head.appendChild(el("th", {}, [swatch(r.color), el("span", {text: r.meta.label})])));
      tbl.appendChild(head);
      skeys.forEach(k => {
        const tr = el("tr", {}, [el("td", {class: "k", text: k})]);
        runs.forEach(r => tr.appendChild(el("td", {text: fmt((r.summary || {})[k])})));
        tbl.appendChild(tr);
      });
      summaryRoot.appendChild(el("div", {class: "cmp-wrap"}, [tbl]));
    }
  }

  // charts
function chart(name, seriesList, multi) {
  const W = 360, H = 158, PL = 44, PR = 12, PT = 10, PB = 26;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  seriesList.forEach(s => s.points.forEach(p => {
    if (p[0] < xmin) xmin = p[0]; if (p[0] > xmax) xmax = p[0];
    if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1];
  }));
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) { const p = Math.abs(ymin) || 1; ymin -= p * 0.05; ymax += p * 0.05; }
  const padY = (ymax - ymin) * 0.08; ymin -= padY; ymax += padY;
  const sx = v => PL + (v - xmin) / (xmax - xmin) * (W - PL - PR);
  const sy = v => PT + (1 - (v - ymin) / (ymax - ymin)) * (H - PT - PB);

  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, width: W, height: H, preserveAspectRatio: "xMidYMid meet"});
  for (let i = 0; i <= 3; i++) {
    const yv = ymin + (ymax - ymin) * i / 3, y = sy(yv);
    svg.appendChild(el("line", {class: "grid-line", x1: PL, y1: y, x2: W - PR, y2: y}));
    svg.appendChild(el("text", {class: "axis-txt", x: PL - 5, y: y + 3, "text-anchor": "end", text: fmt(yv)}));
  }
  xticks(xmin, xmax, 5).forEach(xv => {
    const x = sx(xv);
    if (xv > xmin && xv < xmax)
      svg.appendChild(el("line", {class: "grid-line", x1: x, y1: PT, x2: x, y2: H - PB}));
    svg.appendChild(el("text", {class: "axis-txt", x: x, y: H - 8, "text-anchor": "middle", text: String(xv)}));
  });

  const dots = [];
  seriesList.forEach(s => {
    const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    const path = el("path", {class: "plot", d}); path.setAttribute("stroke", s.color); svg.appendChild(path);
    const dot = el("circle", {class: "dot", r: 3, cx: -10, cy: -10, opacity: 0}); dot.setAttribute("fill", s.color);
    svg.appendChild(dot); dots.push({s, dot});
  });
  const cross = el("line", {class: "crosshair", x1: 0, y1: PT, x2: 0, y2: H - PB}); svg.appendChild(cross);

  const box = el("div", {class: "chart"});
  const titleRow = el("div", {class: "title"}, [el("span", {text: name})]);
  if (!multi) {
    const pts = seriesList[0].points;
    titleRow.appendChild(el("span", {class: "cur", text: "last " + fmt(pts[pts.length - 1][1])}));
  }
  box.appendChild(titleRow);
  const holder = el("div", {class: "svg-holder"});
  const tip = el("div", {class: "tt"});
  holder.appendChild(svg); holder.appendChild(tip);
  box.appendChild(holder);

  svg.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * W;
    let bx = null, bd = Infinity;
    seriesList.forEach(s => s.points.forEach(p => { const dd = Math.abs(sx(p[0]) - px); if (dd < bd) { bd = dd; bx = p[0]; } }));
    if (bx === null) return;
    cross.setAttribute("x1", sx(bx)); cross.setAttribute("x2", sx(bx)); cross.setAttribute("opacity", 1);
    let rows = '<div class="row" style="color:var(--muted)">step ' + bx + '</div>';
    dots.forEach(({s, dot}) => {
      let pp = null, pd = Infinity;
      s.points.forEach(p => { const dd = Math.abs(p[0] - bx); if (dd < pd) { pd = dd; pp = p; } });
      if (pp) {
        dot.setAttribute("cx", sx(pp[0])); dot.setAttribute("cy", sy(pp[1])); dot.setAttribute("opacity", 1);
        rows += '<div class="row"><span class="sw" style="background:' + s.color + '"></span>'
             + (multi ? s.label + ": " : "") + fmt(pp[1]) + '</div>';
      }
    });
    tip.innerHTML = rows; tip.style.opacity = 1;
    const w = r.width;
    const left = Math.min(sx(bx) / W * w + 12, w - tip.offsetWidth - 6);
    tip.style.left = Math.max(4, left) + "px";
  });
  svg.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", 0); dots.forEach(({dot}) => dot.setAttribute("opacity", 0)); tip.style.opacity = 0;
  });
  return box;
}

  // A metric whose per-run series sit on incomparable scales (e.g. SFT CE loss
  // ~3.4 vs the EBFT composite RL+γCE loss ~0.1) is split into one chart per
  // run instead of overlaid: the union y-range would flatten every line.
  // "Incomparable" = the smallest varying series' own span is under 10% of the
  // union span. All-constant series (flat lr lines etc.) always overlay.
  function splitScales(seriesList) {
    if (seriesList.length < 2) return false;
    let ulo = Infinity, uhi = -Infinity;
    const spans = seriesList.map(s => {
      let lo = Infinity, hi = -Infinity;
      s.points.forEach(p => { if (p[1] < lo) lo = p[1]; if (p[1] > hi) hi = p[1]; });
      if (lo < ulo) ulo = lo; if (hi > uhi) uhi = hi;
      return hi - lo;
    });
    const union = uhi - ulo;
    const nz = spans.filter(sp => sp > 0);
    if (!(union > 0) || !nz.length) return false;
    return Math.min(...nz) / union < 0.10;
  }

  // charts: metric union across runs, grouped by prefix; chart() recomputes
  // its own x/y domains from the union of series handed to it.
  const groupOrder = ["train", "eval", "perf"];
  const metricNames = unionKeys(runs.map(r => r.series));
  const groups = {};
  metricNames.forEach(k => {
    const g = k.includes("/") ? k.split("/")[0] : "other";
    (groups[g] = groups[g] || []).push(k);
  });
  const gnames = Object.keys(groups).sort((a, b) => {
    const ia = groupOrder.indexOf(a), ib = groupOrder.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const chartsRoot = document.getElementById("charts"); clear(chartsRoot);
  gnames.forEach(g => {
    groups[g].sort().forEach(metric => {
      const seriesList = [];
      runs.forEach(r => {
        const pts = (r.series[metric] || []).filter(p => p[0] !== null && p[0] !== undefined);
        if (pts.length) seriesList.push({label: r.meta.label, color: r.color, points: pts});
      });
      if (!seriesList.length) return;
      if (MULTI && splitScales(seriesList))
        seriesList.forEach(s => chartsRoot.appendChild(chart(metric + " — " + s.label, [s], false)));
      else
        chartsRoot.appendChild(chart(metric, seriesList, MULTI));
    });
  });
  if (!chartsRoot.children.length)
    chartsRoot.appendChild(el("div", {class: "sub", text: "no metrics logged"}));

  // config
  const configRoot = document.getElementById("config"); clear(configRoot);
  if (!MULTI) {
    const cfg = (runs[0] && runs[0].config) || {};
    const wrap = el("div", {class: "cfg-wrap"});
    const t = el("table", {class: "cfg"});
    Object.keys(cfg).forEach(k => t.appendChild(el("tr", {}, [
      el("td", {class: "k", text: k}), el("td", {class: "v", text: fmt(cfg[k])})])));
    wrap.appendChild(t); configRoot.appendChild(wrap);
  } else {
    const ckeys = unionKeys(runs.map(r => r.config));
    const tbl = el("table", {class: "cmp"});
    const head = el("tr", {}, [el("th", {class: "k", text: ""})]);
    runs.forEach(r => head.appendChild(el("th", {}, [swatch(r.color), el("span", {text: r.meta.label})])));
    tbl.appendChild(head);
    ckeys.forEach(k => {
      const vals = runs.map(r => (r.config || {})[k]);
      const differ = vals.some(v => JSON.stringify(v) !== JSON.stringify(vals[0]));
      const tr = el("tr", differ ? {class: "diff"} : {}, [el("td", {class: "k", text: k})]);
      vals.forEach(v => tr.appendChild(el("td", {text: fmt(v)})));
      tbl.appendChild(tr);
    });
    configRoot.appendChild(el("div", {class: "cmp-wrap"}, [tbl]));
  }
}

// ---- assign per-run colors + disambiguate duplicate labels, then render ----
function prep(runs) {
  const out = runs.map((r, i) => ({
    meta: Object.assign({}, r.meta || {}),
    config: r.config || {}, summary: r.summary || {}, series: r.series || {},
    color: PALETTE[i % PALETTE.length],
  }));
  const seen = {};
  out.forEach(r => {
    const base = r.meta.label || r.meta.run_name || "run";
    seen[base] = (seen[base] || 0) + 1;
    r.meta.label = seen[base] > 1 ? base + " #" + seen[base] : base;
  });
  return out;
}
function rerender() {
  render(prep(RUNS));
  document.getElementById("resetBtn").hidden = RUNS.length <= BASE_RUNS.length;
}

// ---- overlay import: read another report.html, pull its DATA.runs, append ----
function extractRuns(text) {
  const marker = "const DATA = ";
  const i = text.indexOf(marker);
  if (i < 0) throw new Error("not a run report");
  const rest = text.slice(i + marker.length);
  const end = rest.indexOf(";\n");
  let blob = (end < 0 ? rest : rest.slice(0, end)).trim();
  if (blob.endsWith(";")) blob = blob.slice(0, -1);
  const runs = JSON.parse(blob).runs || [];
  if (!runs.length) throw new Error("no runs in file");
  return runs;
}
// step range actually present in a run's series (across every metric)
function stepBounds(run) {
  let lo = Infinity, hi = -Infinity;
  Object.values(run.series || {}).forEach(pts => pts.forEach(p => {
    if (p[0] < lo) lo = p[0]; if (p[0] > hi) hi = p[0];
  }));
  return [lo, hi];
}
// "(epoch X.XX)" if the base run logged steps_per_epoch, else ""
function epochHint(baseRun, step) {
  const spe = baseRun.config && baseRun.config.steps_per_epoch;
  return (spe && spe > 0) ? " (epoch " + (step / spe).toFixed(2) + ")" : "";
}
// shift every point's x (step) by off, in place
function applyStepOffset(run, off) {
  if (!off) return;
  const s = run.series || {};
  Object.keys(s).forEach(k => { s[k] = s[k].map(p => [p[0] + off, p[1]]); });
}

// Ask, per imported run, whether it CONTINUES a currently-shown run and at which
// step -- then splice it in with its x-axis shifted so its first point lands on
// that join step (robust whether the resumed trainer reset its step counter to 0
// or kept counting). A shared metric like eval/held_out then draws as one
// continuous curve across the SFT->EBFT join.
const resumePanel = document.getElementById("resumePanel");
function showResumeLinker(pending, errs) {
  clear(resumePanel); resumePanel.hidden = false;
  const existing = RUNS.slice();  // candidate base runs, frozen at prompt time
  pending.forEach(pr => {
    const [plo] = stepBounds(pr);
    const row = el("div", {class: "resume-row"});
    row.appendChild(el("span", {class: "rl", text: (pr.meta && pr.meta.label) || "imported run"}));
    const sel = el("select", {});
    sel.appendChild(el("option", {value: "-1", text: "— standalone (no resume) —"}));
    existing.forEach((r, i) => sel.appendChild(el("option", {value: String(i), text: "continues " + r.meta.label})));
    const stepIn = el("input", {type: "number", step: "1", disabled: "", placeholder: "join step"});
    const hint = el("span", {class: "sub"});
    function refresh() {
      const bi = parseInt(sel.value, 10);
      if (bi < 0) { stepIn.disabled = true; stepIn.value = ""; hint.textContent = ""; return; }
      stepIn.disabled = false;
      if (stepIn.value === "") stepIn.value = String(stepBounds(existing[bi])[1]);
      hint.textContent = epochHint(existing[bi], parseFloat(stepIn.value) || 0);
    }
    sel.addEventListener("change", refresh);
    stepIn.addEventListener("input", refresh);
    row.appendChild(sel);
    row.appendChild(el("span", {class: "sub", text: "at step"}));
    row.appendChild(stepIn); row.appendChild(hint);
    row._pr = pr; row._sel = sel; row._stepIn = stepIn; row._plo = plo; row._existing = existing;
    resumePanel.appendChild(row);
  });
  const apply = el("button", {class: "tbtn", type: "button", text: "Add to chart"});
  apply.addEventListener("click", () => {
    resumePanel.querySelectorAll(".resume-row").forEach(row => {
      const bi = parseInt(row._sel.value, 10);
      if (bi >= 0) {
        const join = parseFloat(row._stepIn.value) || 0;
        applyStepOffset(row._pr, join - row._plo);  // land first point on the join step
        const base = row._existing[bi];
        row._pr.meta = row._pr.meta || {};
        row._pr.meta.label = ((row._pr.meta.label) || "run") + " ↳@" + join;
        row._pr.meta.resume_of = base.meta.label; row._pr.meta.resume_at = join;
      }
      RUNS.push(row._pr);
    });
    resumePanel.hidden = true; clear(resumePanel);
    rerender();
    document.getElementById("importMsg").textContent = "";
  });
  const cancel = el("button", {class: "tbtn", type: "button", text: "Cancel"});
  cancel.addEventListener("click", () => { resumePanel.hidden = true; clear(resumePanel); });
  const btnRow = el("div", {class: "resume-row"}, [apply, cancel]);
  if (errs && errs.length) btnRow.appendChild(el("span", {class: "sub", text: "skipped: " + errs.join("; ")}));
  resumePanel.appendChild(btnRow);
}

const importInput = document.getElementById("importInput");
document.getElementById("importBtn").addEventListener("click", () => importInput.click());
document.getElementById("resetBtn").addEventListener("click", () => {
  RUNS = BASE_RUNS.slice(); rerender();
  resumePanel.hidden = true; clear(resumePanel);
  document.getElementById("importMsg").textContent = "";
});
importInput.addEventListener("change", async ev => {
  const files = Array.from(ev.target.files || []);
  const pending = []; const errs = [];
  for (const f of files) {
    try { extractRuns(await f.text()).forEach(r => pending.push(r)); }
    catch (e) { errs.push(f.name + " — " + e.message); }
  }
  ev.target.value = "";  // let the same file be re-picked later
  const msg = document.getElementById("importMsg");
  // With runs already on screen, offer the resume linker; otherwise just add.
  if (pending.length && RUNS.length) {
    showResumeLinker(pending, errs);
  } else {
    pending.forEach(r => RUNS.push(r));
    rerender();
    const parts = [];
    if (pending.length) parts.push("overlaid " + pending.length + " report" + (pending.length > 1 ? "s" : ""));
    if (errs.length) parts.push("skipped: " + errs.join("; "));
    msg.textContent = parts.join("  ·  ");
  }
});

// ---- live-monitor mode: only when served by the trainer's --live-report
// server (LIVE injected). Polls /metrics to redraw the charts as the run logs,
// and drives the batch viewer via /batch?step=N -- texts are decoded on demand
// in the trainer process, so browsing past/future steps costs nothing here and
// nothing about the dataset ever lands in this file. ----
if (LIVE && BASE_RUNS.length) {
  document.getElementById("liveViewer").hidden = false;
  const liveBar = document.getElementById("liveBar"); liveBar.hidden = false;
  const pvStep = document.getElementById("pvStep");
  const pvTotal = document.getElementById("pvTotal");
  const pvFollow = document.getElementById("pvFollow");
  const pvMsg = document.getElementById("pvMsg");
  const pvBody = document.getElementById("pvBody");
  pvStep.min = LIVE.first_step; pvStep.max = LIVE.total_steps;
  pvTotal.textContent = "/ " + LIVE.total_steps;
  let curStep = null;    // newest trained step seen in /metrics
  let shownStep = null;  // step the viewer currently displays
  const cache = new Map();

  function setLive(ok) {
    liveBar.textContent = ok
      ? "● live — polling the trainer every " + ((LIVE.poll_ms || 2500) / 1000) + "s"
      : "◌ monitor disconnected (run over or trainer gone) — report.html next to the adapter is the shareable copy";
    liveBar.style.color = ok ? "#10b981" : "#f59e0b";
  }

  function renderBatches(data) {
    clear(pvBody);
    const mbs = data.micro_batches || [];
    const nMicro = mbs.reduce((a, mb) => Math.max(a, (mb.micro || 0) + 1), 0);
    mbs.forEach(mb => {
      const bits = ["micro-batch " + (mb.micro + 1) + "/" + nMicro,
                    "data pass " + (mb.epoch + 1)];
      if (mb.rank !== undefined && mb.rank !== null) bits.push("rank " + mb.rank);
      pvBody.appendChild(el("div", {class: "pv-mbh", text: bits.join("  ·  ")}));
      (mb.sequences || []).forEach(sq => {
        const card = el("div", {class: "pv-seq"});
        const docs = sq.docs || [];
        card.appendChild(el("div", {class: "pv-seqh", text:
          (docs.length > 1 ? "packed block #" + sq.index + " · " + docs.length + " docs"
                           : "example #" + sq.index)}));
        docs.forEach(doc => {
          const d = el("div", {class: "pv-doc"});
          // One span per masked/supervised run: a multi-turn doc alternates
          // several times, so the interior user turns stay visible.
          (doc.segments || []).forEach(seg => d.appendChild(
            el("span", {class: seg.sup ? "pv-resp" : "pv-prompt", text: seg.text})));
          d.appendChild(el("span", {class: "pv-tok",
            text: "  [" + doc.n_prompt + " prompt + " + doc.n_sup + " supervised tok]"}));
          card.appendChild(d);
        });
        pvBody.appendChild(card);
      });
    });
  }

  async function showStep(s) {
    if (!isFinite(s)) return;
    s = Math.min(Math.max(Math.round(s), LIVE.first_step), LIVE.total_steps);
    shownStep = s; pvStep.value = s;
    let data = cache.get(s);
    if (!data) {
      pvMsg.textContent = "loading…";
      try {
        const r = await fetch("batch?step=" + s, {cache: "no-store"});
        if (!r.ok) throw new Error(await r.text());
        data = await r.json();
      } catch (e) { pvMsg.textContent = "step " + s + ": " + e.message; return; }
      cache.set(s, data);
      if (cache.size > 64) cache.delete(cache.keys().next().value);
    }
    pvMsg.textContent = "";
    if (shownStep === s) renderBatches(data);  // stale response for a superseded step: drop
  }

  document.getElementById("pvPrev").addEventListener("click", () => {
    pvFollow.checked = false; showStep((shownStep ?? LIVE.first_step) - 1); });
  document.getElementById("pvNext").addEventListener("click", () => {
    pvFollow.checked = false; showStep((shownStep ?? LIVE.first_step - 1) + 1); });
  pvStep.addEventListener("change", () => {
    pvFollow.checked = false; showStep(parseInt(pvStep.value, 10)); });

  let lastMetrics = null;
  async function poll() {
    let txt;
    try { txt = await (await fetch("metrics", {cache: "no-store"})).text(); }
    catch (e) { setLive(false); return; }
    setLive(true);
    if (txt !== lastMetrics) {
      lastMetrics = txt;
      // Rebuild run 0's series from the jsonl (mirrors the Python _load_run).
      const series = {};
      let maxStep = null;
      txt.split("\n").forEach(line => {
        line = line.trim(); if (!line) return;
        let row; try { row = JSON.parse(line); } catch (e) { return; }
        Object.keys(row).forEach(k => {
          if (k === "step") return;
          const v = row[k];
          if (typeof v !== "number" || !isFinite(v)) return;
          (series[k] = series[k] || []).push([row.step, v]);
        });
        if (row.step !== null && row.step !== undefined
            && (maxStep === null || row.step > maxStep)) maxStep = row.step;
      });
      BASE_RUNS[0].series = series;
      rerender();
      if (maxStep !== null && maxStep >= LIVE.first_step) {
        curStep = Math.min(maxStep, LIVE.total_steps);
        pvTotal.textContent = "/ " + LIVE.total_steps + " · trained through step " + curStep;
      }
    }
    if (pvFollow.checked && curStep !== null && shownStep !== curStep) showStep(curStep);
  }
  setLive(true);
  showStep(LIVE.first_step);  // future steps are computable before they train
  poll();
  setInterval(poll, LIVE.poll_ms || 2500);
}

// initial paint
rerender();
</script>
</body>
</html>
"""


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render or compare exl3-qlora run reports. One dir -> render "
                    "that run's report.html in place; several dirs -> overlay them "
                    "on one comparison page.")
    ap.add_argument("out_dirs", nargs="+",
                    help="run --out dir(s) (each containing a run_report/ subdir).")
    ap.add_argument("-o", "--output", default=None,
                    help="output HTML path for compare mode (default: compare_report.html).")
    ap.add_argument("--labels", default=None,
                    help="comma-separated legend labels for the runs (compare mode; "
                         "default: each run's name).")
    args = ap.parse_args()
    if len(args.out_dirs) == 1 and not args.labels:
        print(render_report(args.out_dirs[0]))
    else:
        labels = args.labels.split(",") if args.labels else None
        compare_reports(args.out_dirs, args.output, labels)


if __name__ == "__main__":
    _main()
