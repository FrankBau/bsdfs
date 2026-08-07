""" Experiment accompanying the paper, generates log and .pdf figure"""


import math
import random
import networkx as nx
from collections import deque
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


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


def bsdfs_delays(G, s, t, k):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []
    steps = 0
    
    def fruitful(v, sd):
        nonlocal steps
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            steps += 1
            for p in G.predecessors(q):
                steps += 1
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        nonlocal steps
        steps += 1
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            steps += 1
            if b[w] + h < k:
                if w == t:
                    steps += h + 2          # <-- output cost, paper sec:work-attribution
                    yield steps
                    steps = 0
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)
    yield steps


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


def make_ax(ax, title, graph_generator):
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in K_VALUES:
            delays = list(islice(bsdfs_delays(G, s, t, k), ISLICE))
            if len(delays) >= ISLICE:           # do not use truncated 
                continue
            x = len(delays)                     # intervals = outputs + 1
            y = max(delays) / ((k+1)*(n+m))
            data[k].append((x, y))
            print(f"{run=:8} {n=:4} {m=:4} {k=:4}   {(k+1)*(n+m)=:10} {max(delays)=:10} {len(delays)=:10}   {x=:10} {y=:6.4f}")

    for i, k in enumerate(ks):
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.45, lw=0, color=CMAP(i), label=f"k={k}")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.axhline(2, color="gray", lw=1, ls="-.")
    ax.axhline(3, color="gray", lw=1, ls="--")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    maxima = {k: max(y for _, y in data[k]) for k in ks if data[k]}
    samples = {k: len(data[k]) for k in ks}
    return maxima, samples


def main():
    plt.rcParams.update({"pdf.fonttype": 42}) # Type 42 (TrueType) makes the figure text searchable and selectable
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)

    fig.supxlabel("BS-DFS number of intervals")
    axes[0].set_ylabel("max. delay / $(k+1)(n+m)$")

    maxima_er, n_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    maxima_ws, n_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("delay_bounds.pdf", bbox_inches="tight")

    print(f"\n{'k':>3}   {'n ER':>6} {'max ER':>10}   {'n WS':>6} {'max WS':>10}")
    for k in K_VALUES:
        print(f"{k:>3}   {n_er.get(k, float('nan')):6} {maxima_er.get(k, float('nan')):10.4f}   {n_ws.get(k, float('nan')):6} {maxima_ws.get(k, float('nan')):10.4f}")
    print(f"{'all':>3} {max(maxima_er.values()):19.4f} {max(maxima_ws.values()):19.4f}")


if __name__ == "__main__":
    main()
