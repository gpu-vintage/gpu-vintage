"""Figures (scripts/build_results.py).

Print figures for the paper: no titles inside the figures (the captions carry
them), sans-serif lettering at 8 to 9 pt, at most three coloured series per
chart, each also told apart by marker or line style and named on the plot.
Colours: the first three slots of the dataviz reference palette (blue, orange,
aqua), documented as distinguishable under colour-vision deficiency.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # no build timestamp: a rebuild is byte-identical

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpuvintage import econ  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
LABEL = {"1080ti": "GTX 1080 Ti", "p100": "P100", "v100": "V100", "t4": "T4", "a100": "A100",
         "a40": "A40", "a6000": "RTX A6000", "ada6000": "RTX 6000 Ada"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
})


def _save(fig, ctx, name):
    for ext in ("pdf", "eps"):
        fig.savefig(ctx["tex"] / "figures" / f"{name}.{ext}")
    plt.close(fig)


def fig_energy(ctx):
    """Energy per output token by release date, B = 1 and the serving batch (Qwen3-1.7B)."""
    S, w, gpus = ctx["settings"], ctx["w"], ctx["gpus"]
    pts = []
    for g in ctx["order"]:
        b1 = S.serving_point(w, g, S.MAIN_MODEL, cap=1)
        sv = S.serving_point(w, g, S.MAIN_MODEL)
        if b1 and sv:
            pts.append((g, S.year(gpus[g]["release_date"]), b1["j_per_out_tok"], sv["j_per_out_tok"], sv["batch"]))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for g, x, e1, es, b in pts:
        ax.plot([x, x], [es, e1], color=GRID, lw=1.0, zorder=1)
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=34, color=BLUE, marker="o", zorder=3,
               edgecolor="white", linewidth=0.8, label="one request at a time ($B=1$)")
    ax.scatter([p[1] for p in pts], [p[3] for p in pts], s=34, color=ORANGE, marker="s", zorder=3,
               edgecolor="white", linewidth=0.8, label=f"batched ($B\\leq{S.SERVE_B}$)")
    for g, x, e1, es, b in pts:
        ax.annotate(LABEL[g], (x, e1), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK)
        if b < S.SERVE_B:
            ax.annotate(f"$B={b}$", (x, es), xytext=(5, -3), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("Release date")
    ax.set_ylabel("Energy per output token (J)")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, ctx, "energy_vintage")


def fig_depreciation(ctx, gpu="a100", country="United States"):
    """Value of an A100 over its life, capacity-constrained data centre, carbon unpriced: the
    measured pace of progress against the FLOP/s-per-dollar pace and six-year straight-line
    depreciation; the point is its second-hand price in March 2026 (DumpsterCluster)."""
    S = ctx["settings"]
    lives = [r for r in ctx.get("lives", []) if r["gpu"] == gpu and r["country"] == country
             and r["carbon_price"] == 0.0 and r["regime"] == "constrained"]
    by = {r["theta_name"]: r for r in lives}
    if "measured" not in by or "spec" not in by:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    horizon = max(10.0, min(by["measured"]["life"], 25.0))
    for name, color, ls, lab in (("measured", BLUE, "-", "measured pace of progress"),
                                 ("spec", ORANGE, "--", "FLOP/s-per-dollar pace")):
        r = by[name]
        v = econ.Vintage(c0=r["c0"], life=r["life"], k_per_mtok=r["c0"] - r["o"])
        ages = [horizon * i / 400 for i in range(401)]
        vals = [econ.value_at_age(a, v, r["Y"], r["o"], r["theta"], S.DISCOUNT) / r["K"] for a in ages]
        ax.plot(ages, vals, color=color, ls=ls, label=lab)
    ax.plot([0, 6], [1, 0], color=AQUA, ls=":", lw=1.5, label="straight line, six years")
    # A100 40 GB: launch May 2020 at 12,000 USD, 1,900 USD second-hand in March 2026 (Cao et al. 2026)
    ax.plot([5.83], [1900 / 12000], marker="D", ms=5, color=INK, ls="none", label="second-hand price, A100")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Value / purchase cost")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, horizon)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    _save(fig, ctx, "depreciation")


def fig_life_vs_theta(ctx, gpus_=("a100", "ada6000"), country="United States"):
    """Economic life of a new card against the pace of progress theta, carbon unpriced, in the
    United States, with spare and with scarce data-centre capacity; the estimates marked."""
    from build_tables import operating_point, real_price, _op
    S, w, gpus = ctx["settings"], ctx["w"], ctx["gpus"]
    c = next(c for c in ctx["countries"] if c["country"] == country)
    th = ctx.get("thetas", {})
    thetas = [0.02 + 0.005 * i for i in range(97)]  # 2% to 50% a year in log points
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    styles = {("a100", "constrained"): (BLUE, "-"), ("a100", "spare"): (BLUE, ":"),
              ("ada6000", "constrained"): (ORANGE, "-"), ("ada6000", "spare"): (ORANGE, ":")}
    for g in gpus_:
        pt = operating_point(S, w, g)
        price = real_price(gpus[g], ctx["cpi"])
        if pt is None or price is None:
            continue
        K = price * (1 + S.INFRA_SHARE)
        for regime, phi in S.REGIMES.items():
            op = _op(S, c, phi=phi)
            Y = econ.mtok_per_year(pt["y"], op.utilization)
            o = econ.operating_cost_per_mtok(pt["y"], pt["p_busy"], pt["p_idle"], op, pt["rated_w"])
            lives = [econ.solve_vintage(K, Y, o, t, S.DISCOUNT).life for t in thetas]
            color, ls = styles[(g, regime)]
            ax.plot([100 * (math.exp(t) - 1) for t in thetas], lives, color=color, ls=ls,
                    label=f"{LABEL[g]}, {'scarce' if regime == 'constrained' else 'spare'} capacity")
    marks = [("spec", "FLOP/s per dollar"), ("measured", "measured, batched"), ("measured_b1", "measured, $B=1$")]
    xs = sorted((100 * (math.exp(th[n]) - 1), lab) for n, lab in marks if th.get(n, 0) > 0)
    prev, level = None, 0
    for x, lab in xs:
        ax.axvline(x, color=MUTED, lw=0.8, ls="--")
        level = level + 1 if prev is not None and x - prev < 8.0 else 0   # stagger close labels
        ax.annotate(lab, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3 + 9 * level),
                    textcoords="offset points", fontsize=7, color=MUTED, ha="center", va="bottom",
                    annotation_clip=False)
        prev = x
    ax.axhline(6, color=AQUA, lw=1.2, ls="-.", label="six-year accounting life")
    ax.set_xlabel("Pace of progress: annual rise in tokens per dollar of full cost (%)")
    ax.set_ylabel("Economic life (years)")
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7, ncol=3)
    _save(fig, ctx, "life_vs_theta")


def build_all(ctx):
    fig_energy(ctx)
    fig_depreciation(ctx)
    fig_life_vs_theta(ctx)
