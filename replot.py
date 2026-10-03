"""Redraw a runtime figure from its .txt log, without rerunning the experiment.

    python3 replot.py runtime_ss.txt --mode xy
    python3 replot.py runtime_ss.txt --mode ratio --yscale log

runtime.py and runtime_ss.py log every measured instance:

    run=       2 k=   7   intervals1=       651 intervals2=       643   runtime1= 0.000773215 runtime2= 0.000594959

so the scatter can be rebuilt from the log alone.  The log carries no family
markers -- the per-instance lines of the families follow each other in FAMILIES
order -- but `run` is nondecreasing within a family, so a drop marks a boundary.

Two views of the same data:

  ratio  x = competitor time per interval, y = ratio BS-DFS / competitor.
         Reads the ratio directly; discards absolute scale.  The right view
         when the ratios are all near 1 (BS-DFS vs BC-DFS).

  xy     x = BS-DFS runtime, y = competitor runtime, log-log, y=x diagonal.
         Ratio becomes distance from the diagonal, but absolute cost stays
         visible -- so an extreme ratio on a cheap instance reads as cheap.
         Uses the raw runtimes, with no `1 + len_paths` normalization.
"""

import argparse
import math
import re
import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm

from graph_generator import CMAP, FAMILIES, K_VALUES, COLS, make_panels

LINE = re.compile(
    r"run=\s*(\d+) k=\s*(\d+)\s+intervals1=\s*(\d+) intervals2=\s*(\d+)"
    r"\s+runtime1=\s*([\d.]+) runtime2=\s*([\d.]+)")


def parse(path):
    """-> [ {k: [(rt1, rt2, iv1, iv2), ...]} ], one dict per family, FAMILIES order."""
    blocks, prev_run = [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = LINE.match(line)
            if not m:
                continue
            run, k, iv1, iv2 = (int(m[i]) for i in (1, 2, 3, 4))
            if prev_run is None or run < prev_run:
                blocks.append({key: [] for key in K_VALUES})
            prev_run = run
            blocks[-1][k].append((float(m[5]), float(m[6]), iv1, iv2))
    return blocks


def points(rows, mode):
    if mode == "xy":
        return [(rt1, rt2) for rt1, rt2, _, _ in rows]
    return [(rt2 / iv2, (rt1 / iv1) / (rt2 / iv2)) for rt1, rt2, iv1, iv2 in rows]


# how to draw the instances that emit no path.  Colour already encodes k, so
# shape is the channel left for the split; an open cross stays separable from a
# filled disc even where the scatter is dense.
EMPTY_STYLES = {
    "x":        dict(marker="x", s=13, lw=.7, alpha=.7),
    "plus":     dict(marker="+", s=16, lw=.7, alpha=.7),
    "triangle": dict(marker="^", s=9, lw=0, alpha=.55),
    "ring":     dict(marker="o", s=9, lw=.5, alpha=.55, facecolors="none"),
}


def _scatter(ax, xs, ys, color, style):
    style = dict(style)
    if style.pop("facecolors", None) is not None:
        ax.scatter(xs, ys, facecolors="none", edgecolors=color, **style)
    else:
        ax.scatter(xs, ys, color=color, **style)


def decade_limits(lo, hi):
    """Widen [lo, hi] to enclosing powers of ten.

    A log axis spanning less than a decade has no major tick inside it, so
    matplotlib labels the minor ticks instead ("2 x 10^-6", "3 x 10^-6", ...),
    which collides in a narrow panel.  Snapping outward guarantees decade
    ticks, and the minor labels can then be dropped.
    """
    return 10.0 ** math.floor(math.log10(lo)), 10.0 ** math.ceil(math.log10(hi))


def tidy_log_axis(axis):
    axis.set_major_locator(mticker.LogLocator(base=10))
    axis.set_minor_formatter(mticker.NullFormatter())


def make_ax(ax, title, data, mode, yscale, split=None):
    lo, hi = None, None
    xlo = xhi = None
    full = dict(s=4, alpha=0.45, lw=0)
    for i, k in enumerate(K_VALUES):
        if not data[k]:
            continue
        # an instance that emits nothing measures the cost of proving no path
        # exists, not a per-output delay -- drawn apart when split is set
        groups = [(data[k], full)]
        if split:
            groups = [([r for r in data[k] if r[2] > 1], full),
                      ([r for r in data[k] if r[2] == 1], EMPTY_STYLES[split])]
        for rows, style in groups:
            if not rows:
                continue
            xs, ys = zip(*points(rows, mode))
            _scatter(ax, xs, ys, CMAP(i), style)
            vals = xs + ys if mode == "xy" else xs
            lo = min(vals) if lo is None else min(lo, min(vals))
            hi = max(vals) if hi is None else max(hi, max(vals))
            xlo = min(xs) if xlo is None else min(xlo, min(xs))
            xhi = max(xs) if xhi is None else max(xhi, max(xs))

    ax.set_xscale("log")
    ax.set_yscale("log" if mode == "xy" else yscale)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if mode == "xy":
        if lo is not None:
            d = np.array([lo, hi])
            ax.plot(d, d, color="gray", lw=1, ls="--", zorder=0)
            for f in (10, 100):                      # order-of-magnitude guides
                ax.plot(d, d * f, color="lightgray", lw=.6, ls=":", zorder=0)
                ax.plot(d, d / f, color="lightgray", lw=.6, ls=":", zorder=0)
            d0, d1 = decade_limits(lo, hi)
            ax.set_xlim(d0, d1); ax.set_ylim(d0, d1)
            tidy_log_axis(ax.xaxis); tidy_log_axis(ax.yaxis)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.axhline(1, color="lightgray", lw=1, ls="--")
        if xlo is not None:
            ax.set_xlim(*decade_limits(xlo, xhi))
        tidy_log_axis(ax.xaxis)
        if yscale == "log":
            tidy_log_axis(ax.yaxis)


def print_medians(title, data, mode):
    label = "median t_bs/t_other" if mode == "xy" else "median inst. ratio"
    print(f"\n--- {title}: {label} (from the log) ---")
    print(f"{'k':>3} {'n':>6} {'median':>9} {'p1':>9} {'p99':>9}")
    for k in K_VALUES:
        if not data[k]:
            continue
        if mode == "xy":
            r = sorted(rt1 / rt2 for rt1, rt2, _, _ in data[k])
        else:
            r = sorted(y for _, y in points(data[k], mode))
        n = len(r)
        print(f"{k:>3} {n:6,} {statistics.median(r):9.3f} "
              f"{r[n//100]:9.3f} {r[min(n-1, 99*n//100)]:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--mode", default="ratio", choices=("ratio", "xy"))
    ap.add_argument("--yscale", default="log", choices=("log", "linear"),
                    help="ratio mode only; xy is always log-log")
    ap.add_argument("--split", nargs="?", const="x", default=None,
                    choices=tuple(EMPTY_STYLES),
                    help="draw instances that emit no path with this marker")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    blocks = parse(args.log)
    if len(blocks) != len(FAMILIES):
        raise SystemExit(f"{args.log}: found {len(blocks)} family blocks, "
                         f"expected {len(FAMILIES)} -- log truncated?")

    stem = args.log.rsplit(".", 1)[0]
    other = "SimpleSearch" if stem.endswith("_ss") else "BC-DFS"

    plt.rcParams.update({"pdf.fonttype": 42})
    fig, axes = make_panels(sharey=(args.mode != "xy"))
    if args.mode == "xy":
        fig.supxlabel("BS-DFS runtime [s]")
        ylabel = f"{other} runtime [s]"
    else:
        fig.supxlabel(f"{other} runtime per interval [s]")
        ylabel = f"runtime ratio BS-DFS / {other}"
    for ax in axes[::COLS]:
        ax.set_ylabel(ylabel)

    for ax, f, data in zip(axes, FAMILIES, blocks):
        make_ax(ax, f.name, data, args.mode, args.yscale, args.split)

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    if args.split:
        _scatter(axes[0], [], [], "dimgray", EMPTY_STYLES[args.split])
        axes[0].scatter([], [], s=4, lw=0, color="dimgray")
        axes[0].legend(["no path found", "paths enumerated"], loc="best",
                       fontsize=6, framealpha=.8)

    out = args.out or f"{stem}_{args.mode}{'_' + args.split if args.split else ''}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"{args.log}: {sum(len(d[k]) for d in blocks for k in K_VALUES):,} "
          f"instances in {len(blocks)} families -> {out} (mode={args.mode})")
    for f, data in zip(FAMILIES, blocks):
        print_medians(f.name, data, args.mode)


if __name__ == "__main__":
    main()
