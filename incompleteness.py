import math
import random
import networkx as nx
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from collections import Counter

from bsdfs import bsdfs
from bcdfs import bcdfs


ISLICE = 100_000
RUNS = 1_000

K_VALUES = range(3, 11)


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
    print(f"{'k':>3} {'n':>6} {'BS paths':>12} {'BC paths':>12} {'missed':>8} {'worst':>8} {'lossy':>7}")
    for k in K_VALUES:
        c = totals[k]
        print(f"{k:>3} {c['n']:6,} {c['bs']:12,} {c['bc']:12,}"
                f" {100*(1-c['bc']/c['bs']):7.2f}% {c['worst']:8.3f} {100*c['lossy']/c['n']:6.1f}%")


def make_ax(ax, title, graph_generator):
    cmap = plt.get_cmap("plasma")
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in ks:
            paths1 = list(islice(bsdfs(G, s, t, k), ISLICE))
            paths2 = list(islice(bcdfs(G, s, t, k), ISLICE))
            iv1 = len(paths1) + 1   # number of intervals
            iv2 = len(paths2) + 1   # number intervals
            print(f"{run=:8} {n=:4} {m=:4} {k=:4} {len(paths1)=:10} {len(paths2)=:10}")
            x = iv1
            y = iv2 / iv1
            data[k].append((x, y))
            totals[k].update(n=1, lossy=(iv2 < iv1), bs=iv1, bc=iv2)

    for i, k in enumerate(ks):
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.5, lw=0, color=cmap(i / (len(ks) - 1)), label=f"k={k}")

    ax.set_xscale("log")
    ax.set_xlim(1, 1.5e5)
    ax.set_xlabel("BS-DFS intervals")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="lightgray", lw=1, ls="--")

    for k in ks:
        if data[k]:
            totals[k]["worst"] = min(y for _, y in data[k])
    return totals


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), constrained_layout=True)
    axes[0].set_ylabel("BC-DFS intervals (fraction)")
    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    cmap = plt.get_cmap("plasma", len(K_VALUES))
    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("incompleteness.pdf", bbox_inches="tight")


    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


if __name__ == "__main__":
    main()
