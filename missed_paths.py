""" Experiment accompanying the paper, generates log and .pdf figure"""


import math
import random
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from collections import Counter

from bsdfs import bsdfs
from bcdfs import bcdfs

import sys

if any("pydevd" in m for m in sys.modules):
    print("### debug mode - reduced data set, for preview only ###")
    ISLICE = 10_000
    RUNS = 100
else:
    ISLICE = 100_000
    RUNS = 1_000
print(f"{RUNS=} {ISLICE=}")

K_VALUES = range(3, 11)
CMAP = plt.get_cmap("plasma", len(K_VALUES))   # one discrete color per k, shared by scatter and colorbar


def gen_er(run):
    rng = random.Random(42 + run)
    n = rng.randint(6, 30)
    m = int(n * math.exp(rng.uniform(0, math.log(n-1))))
    G = nx.gnm_random_graph(n, m, directed=True, seed=rng)
    s, t = rng.sample(list(G.nodes), 2)
    return G, s, t


def gen_ws(run):
    rng = random.Random(73 + run)
    n = 1000
    d = 6
    p = 0.2
    H = nx.watts_strogatz_graph(n, d, p, seed=rng)
    G = nx.DiGraph(H)
    s, t = rng.sample(list(G.nodes), 2)
    return G, s, t


def print_totals(title, totals):
    print(f"\n--- {title}: totals ---")
    print(f"{'k':>3} {'n':>6} {'trunc':>6} {'BS paths':>12} {'BC paths':>12} {'missed':>8} {'worst':>8} {'lossy':>7}")
    for k in K_VALUES:
        c = totals[k]
        if not c["n"]:
            print(f"{k:>3} {0:6,} {c['truncated']:6,} -- all instances truncated --")
            continue
        print(f"{k:>3} {c['n']:6,} {c['truncated']:6,} {c['bs']:12,} {c['bc']:12,}"
            f" {100*(1-c['bc']/c['bs']):7.2f}% {100*c['worst']:7.2f}% {100*c['lossy']/c['n']:6.1f}%")


def make_ax(ax, title, graph_generator):
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in ks:
            paths1 = list(islice(bsdfs(G, s, t, k), ISLICE))
            if len(paths1) >= ISLICE:
                totals[k].update(truncated=1)
                continue
            paths2 = list(islice(bcdfs(G, s, t, k), ISLICE))
            p1 = len(paths1)   # number of paths
            p2 = len(paths2)   # number paths
            print(f"{run=:8} {n=:4} {m=:4} {k=:4} {len(paths1)=:10} {len(paths2)=:10}")
            if p1 == 0:
                assert p2 == 0
                x = 0
                y = 0
            else:
                x = p1
                y = 1 - p2 / p1
            data[k].append((x, y))
            totals[k].update(n=1, lossy=(p2 < p1), bs=p1, bc=p2, truncated=0)

    for i, k in enumerate(ks):
        if not data[k]:
            continue
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.5, lw=0, color=CMAP(i), label=f"k={k}")

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="lightgray", lw=1, ls="--")

    for k in ks:
        if data[k]:
            totals[k]["worst"] = max(y for _, y in data[k])
    return totals


def main():
    plt.rcParams.update({"pdf.fonttype": 42}) # Type 42 (TrueType) makes the figure text searchable and selectable
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)

    fig.supxlabel("BS-DFS number of paths")
    axes[0].set_ylabel("BC-DFS missed paths")

    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("missed_paths.pdf", bbox_inches="tight")


    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


if __name__ == "__main__":
    main()
