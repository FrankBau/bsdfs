""" Experiment accompanying the paper, generates log and .pdf figure

How many k-bounded simple s-t paths does an instance actually have?

The count is a property of the instance, not of the algorithm: every
complete enumerator returns the same number, so BS-DFS serves only as the
oracle.  The histogram is therefore a description of the graph families --
it says what enumeration workload each family poses, and how that workload
spreads over orders of magnitude as the hop bound k grows.

Two populations are drawn explicitly, because both are informative:

  x = 0        instances with no k-bounded path at all (t out of reach
               within k hops); the left-most bin, in the linear part of
               the symlog axis
  x >= ISLICE  instances right-censored by the enumeration cap; they land
               in the top bin, marked by the dotted line, and are counted
               separately in the log
"""

import statistics
from collections import Counter
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from bsdfs import bsdfs
from graph_generator import (CMAP, FAMILIES, ISLICE, K_VALUES, RUNS,
                             COLS, make_panels, print_families)

print_families("path_counts")

# one bin for the zero-path instances, then logarithmic bins up to the cap
BINS = np.concatenate(([-0.5, 0.5], np.logspace(0, np.log10(ISLICE), 25)))


def print_totals(title, totals):
    print(f"\n--- {title}: k-bounded simple s-t paths per instance ---")
    print(f"{'k':>3} {'n':>6} {'no path':>9} {'censored':>9} {'median':>10}"
          f" {'p90':>12} {'max':>12} {'total':>14}")
    for k in K_VALUES:
        c = totals[k]
        print(f"{k:>3} {c['n']:6,} {100*c['zero']/c['n']:8.1f}% {c['trunc']:9,}"
              f" {c['median']:10,.0f} {c['p90']:12,.0f} {c['max']:12,} {c['paths']:14,}")


def make_ax(ax, title, graph_generator):
    ks = list(K_VALUES)

    data = {k: [] for k in ks}
    totals = {k: Counter() for k in ks}
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in ks:
            p = sum(1 for _ in islice(bsdfs(G, s, t, k), ISLICE))
            data[k].append(p)
            totals[k].update(n=1, zero=(p == 0), trunc=(p >= ISLICE), paths=p)
            print(f"{run=:8} {n=:4} {m=:4} {k=:4} {p=:10}")

    for i, k in enumerate(ks):
        ax.hist(data[k], bins=BINS, histtype="step", lw=1, color=CMAP(i), label=f"k={k}")

    ax.set_xscale("symlog", linthresh=1)
    ax.axvline(ISLICE, color="gray", lw=1, ls=":")   # enumeration cap: right-censored
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    for k in ks:
        totals[k]["median"] = statistics.median(data[k])
        totals[k]["p90"] = np.percentile(data[k], 90)
        totals[k]["max"] = max(data[k])
    return totals


def main():
    plt.rcParams.update({"pdf.fonttype": 42}) # Type 42 (TrueType) makes the figure text searchable and selectable
    # every panel holds the same number of instances, so the counts stay
    # comparable without a shared y axis -- and not sharing it keeps the tall
    # zero-path bin of one family from flattening the shape of another
    fig, axes = make_panels(sharey=False)

    fig.supxlabel("number of $k$-bounded simple $s$-$t$ paths")
    for ax in axes[::COLS]:                      # leftmost panel of each row
        ax.set_ylabel("instances")

    totals = [make_ax(ax, f.name, f.generate) for ax, f in zip(axes, FAMILIES)]

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("path_counts.pdf", bbox_inches="tight")

    for f, t in zip(FAMILIES, totals):
        print_totals(f.name, t)


if __name__ == "__main__":
    main()
