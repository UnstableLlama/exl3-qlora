"""LoRA capacity accounting: dataset size in bits, adapter capacity by rank, and
the training-curve asymptote.

Follows "Predicting LoRA size for any dataset" (Engram, 2026,
https://engram.com/blog/lora-size). The idea: measure the dataset and the
adapter in the same unit. A dataset's size is its negative log-likelihood under
the frozen base model, in bits; what an adapter has stored is how far it has
pushed that code length down.

Three pieces, all pure Python (no torch/numpy) so the report can use them:

``mp_capacity_bits(m, n, r)``
    Capacity of a rank-r adapter on ONE m x n matrix: the expected sum of the
    top-r squared singular values of an i.i.d. Gaussian m x n matrix, over
    2 ln 2. Evaluated through the Marchenko-Pastur law. Roughly linear at low
    rank (~1.3-1.4 bits per LoRA parameter), flattening to mn / (2 ln 2) at
    full rank.

``capacity_range_bits(shapes)`` / ``rank_for_bits(shapes, bits)``
    Whole-adapter capacity. Summing the per-matrix figure is the FLOOR: the
    post measures adapters trained jointly across many layers storing up to
    ~2.5x that (3.29 bits/param measured against 1.30 predicted, 24 MLP layers
    of a 0.8B model). That bonus has no theory and was measured on one small
    full-precision model, so the result is reported as a range, not a number.

``fit_log_logistic(points)``
    bits(T) = bits_inf * T^c / (T^c + tau^c), fitted to a stored-bits curve to
    project where a run is heading.

Caveats that apply to every number here: "fits" means memorizing the training
set, which is a capacity floor and not a generalization optimum; and during the
first epoch the trainer's loss is measured on unseen examples, so "stored" then
reflects what generalizes rather than what is memorized.
"""

import math

LN2 = math.log(2.0)
BITS_PER_KB = 8000.0        # the post's KB: 1000 bytes

# Joint-training bonus over the additive per-matrix theory, as measured in the
# post (see module docstring). The upper end of every reported range.
JOINT_BONUS = 2.5

_MP_GRID = 2048


def nats_to_bits(nats):
    return nats / LN2


def fmt_bits(bits):
    """Human-readable size, in the post's units (1 KB = 8000 bits)."""
    b = bits / 8.0
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(b) >= div:
            return "%.3g %s" % (b / div, unit)
    return "%.0f B" % b


def mp_capacity_bits(m, n, r):
    """Capacity in bits of a rank-``r`` adapter on one ``m`` x ``n`` matrix."""
    m, n = (m, n) if m <= n else (n, m)
    if m <= 0 or r <= 0:
        return 0.0
    full = m * n / (2.0 * LN2)
    if r >= m:
        return full
    # Eigenvalues of Z Z^T / n follow Marchenko-Pastur with ratio g = m/n.
    # Substituting lam = 1 + g + 2 sqrt(g) cos(phi), phi in [0, pi] (phi = 0 is
    # the top edge) turns the two tail integrals into
    #   fraction of eigenvalues above lam(phi):  int_0^phi 2 sin^2 / (pi lam)
    #   share of total mass above lam(phi):      (phi - sin phi cos phi) / pi
    # Solve the first for phi at fraction r/m (midpoint rule; the integrand is
    # bounded, including at g = 1), then read the second.
    g = m / n
    sg = math.sqrt(g)
    target = r / m
    h = math.pi / _MP_GRID
    acc = 0.0
    phi = math.pi
    for i in range(_MP_GRID):
        p = (i + 0.5) * h
        lam = 1.0 + g + 2.0 * sg * math.cos(p)
        s = math.sin(p)
        step = 2.0 * s * s / (math.pi * max(lam, 1e-12)) * h
        if acc + step >= target:
            phi = i * h + h * (target - acc) / step
            break
        acc += step
    mass = (phi - math.sin(phi) * math.cos(phi)) / math.pi
    return full * min(max(mass, 0.0), 1.0)


def additive_capacity_bits(shapes, rank=None):
    """Sum of per-matrix capacities. ``shapes`` is an iterable of ``(m, n, r)``;
    ``rank`` overrides every adapter's own rank (for a what-if at uniform rank).
    Identical (m, n, r) triples are computed once."""
    counts = {}
    for m, n, r in shapes:
        key = (min(m, n), max(m, n), r if rank is None else rank)
        counts[key] = counts.get(key, 0) + 1
    return sum(c * mp_capacity_bits(m, n, r) for (m, n, r), c in counts.items())


def capacity_range_bits(shapes, rank=None):
    """(floor, ceiling) capacity of the whole adapter set in bits: the additive
    theory, and that times the joint-training bonus (a single adapted matrix
    gets no bonus)."""
    shapes = list(shapes)
    lo = additive_capacity_bits(shapes, rank)
    return lo, lo * (JOINT_BONUS if len(shapes) > 1 else 1.0)


def rank_for_bits(shapes, bits):
    """Smallest uniform rank whose capacity reaches ``bits``, as
    ``(optimistic, conservative)``: with the joint-training bonus, and on the
    additive floor. An entry is None when even full rank falls short."""
    shapes = list(shapes)
    if not shapes or bits <= 0:
        return None, None
    r_max = max(min(m, n) for m, n, _ in shapes)
    bonus = JOINT_BONUS if len(shapes) > 1 else 1.0

    def solve(mult):
        if additive_capacity_bits(shapes, r_max) * mult < bits:
            return None
        lo, hi = 1, r_max
        while lo < hi:
            mid = (lo + hi) // 2
            if additive_capacity_bits(shapes, mid) * mult >= bits:
                hi = mid
            else:
                lo = mid + 1
        return lo

    return solve(bonus), solve(1.0)


def fit_log_logistic(points, max_points=200):
    """Fit ``y = B * T^c / (T^c + tau^c)`` to ``[(T, y), ...]``.

    Grid over (c, tau) with B solved in closed form, then one refinement pass
    around the best cell. Returns ``{"asymptote", "tau", "c", "rmse",
    "last"}`` or None when there is too little signal to fit (fewer than 8
    usable points, or a curve that isn't above zero on average).

    The asymptote is only as good as the curve's bend: while ``tau`` is beyond
    the last observed step the run hasn't reached its half-way point and the
    projection is an extrapolation -- callers should say so.
    """
    pts = [(float(t), float(y)) for t, y in points
           if t is not None and t > 0 and y == y]
    if len(pts) < 8:
        return None
    if len(pts) > max_points:
        stride = len(pts) / max_points
        pts = [pts[int(i * stride)] for i in range(max_points)]
    if sum(y for _, y in pts) <= 0:
        return None
    t_max = pts[-1][0]
    logs = [math.log(t) for t, _ in pts]
    ys = [y for _, y in pts]

    def sse(c, tau):
        lt = math.log(tau)
        sxy = sxx = 0.0
        fs = []
        for lg, y in zip(logs, ys):
            # T^c / (T^c + tau^c) as a logistic in log-space: overflow-safe.
            f = 1.0 / (1.0 + math.exp(min(c * (lt - lg), 700.0)))
            fs.append(f)
            sxy += f * y
            sxx += f * f
        if sxx <= 0:
            return None
        b = sxy / sxx
        if b <= 0:
            return None
        return sum((b * f - y) ** 2 for f, y in zip(fs, ys)), b

    def search(cs, taus):
        best = None
        for c in cs:
            for tau in taus:
                res = sse(c, tau)
                if res is not None and (best is None or res[0] < best[0]):
                    best = (res[0], res[1], c, tau)
        return best

    geo = lambda lo, hi, k: [lo * (hi / lo) ** (i / (k - 1)) for i in range(k)]
    best = search(geo(0.2, 4.0, 20), geo(max(t_max * 1e-3, 0.5), t_max * 100.0, 48))
    if best is None:
        return None
    _, _, c0, tau0 = best
    best = search(geo(c0 / 1.2, c0 * 1.2, 15), geo(tau0 / 1.3, tau0 * 1.3, 25)) or best
    err, b, c, tau = best
    return {"asymptote": b, "tau": tau, "c": c,
            "rmse": math.sqrt(err / len(pts)), "last": ys[-1]}


def capacity_verdict(fit, dataset, steps_done):
    """One line saying what a stored-bits curve implies, given its fit, the
    dataset size (same unit as the curve) and how many steps it covers."""
    if not fit or not dataset or dataset <= 0:
        return None
    share = fit["asymptote"] / dataset
    if fit["tau"] > steps_done:
        return ("too early to project: the curve has not reached its half-way "
                "point (train longer before reading the asymptote)")
    if share >= 0.95:
        if fit["last"] / dataset >= 0.95:
            return "dataset fits: stored bits have reached the dataset size"
        return ("step-bound: the projected asymptote covers the dataset, so more "
                "steps (not more rank) would store the rest")
    return ("capacity-bound: the projected asymptote is %.0f%% of the dataset, so "
            "more steps will not close the gap; raise the rank or widen the targets"
            % (100.0 * share))
