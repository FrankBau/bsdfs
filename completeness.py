"""
The extra cost of completeness
"""

import math
import random
import networkx as nx
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

ISLICE = 100_000
RUNS = 1_000

K_VALUES = range(3, 11)


from collections import Counter, deque, defaultdict


def bsdfs(G, s, t, k, steps):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    steps["drop"] += 1
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            steps["fruitful"] += 1
            fruitful(v, sd)
        else:
            steps["raise"] += 1
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)


def bcdfs(G, s, t, k, steps):
    """original control-flow, NO completeness"""
    S = []
    bar = {v: 0 for v in G.nodes}

    def length(S):
        return len(S) - 1

    def UpdateBarrier(u, l, root_call=False):
        if bar[u] > l:
            steps["root" if root_call else "drop"] += 1
            bar[u] = l
            for v in G.predecessors(u):
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        F = k + 1
        S.append(u)
        if u == t:
            yield S.copy()
            S.pop()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                if v not in S:
                    if length(S) + 1 + bar[v] <= k:
                        f = yield from search(v)
                        if f != k + 1:
                            F = min(F, f + 1)
        if F == k + 1:
            steps["raise"] += 1
            bar[u] = k - length(S) + 1
        else:
            steps["fruitful"] += 1
            UpdateBarrier(u, F, root_call=True)
        S.pop()
        return F

    yield from search(s)
    
    
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
    print(f"\n--- {title} ---")
    print(f"{'k':>3} {'n':>6} {'BS origin':>12} {'BC origin':>12} {'ratio':>8}"
          f" {'BC fruitful':>12} {'fired':>7} {'y=0':>7}")
    for k in K_VALUES:
        c = totals[k]
        if not c["n"] or not c["bs_origin"] or not c["bc_fruitful"]:
            print(f"{k:>3} {c['n']:6,} -- no fruitful calls --")
            continue
        print(f"{k:>3} {c['n']:6,} {c['bs_origin']:12,} {c['bc_origin']:12,}"
              f" {c['bc_origin']/c['bs_origin']:8.4f} {c['bc_fruitful']:12,}"
              f" {c['bc_origin']/c['bc_fruitful']:7.4f} {100*c['zeros']/c['n']:6.1f}%")


def make_ax(ax, title, graph_generator):
    cmap = plt.get_cmap("plasma")
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}

    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in ks:
            steps1 = defaultdict(int)
            paths1 = list(islice(bsdfs(G, s, t, k, steps1), ISLICE))
            steps2 = defaultdict(int)
            paths2 = list(islice(bcdfs(G, s, t, k, steps2), ISLICE))
            print(f"{run=:8} {n=:4} {m=:4} {k=:4} {steps1["fruitful"]=:8} {steps1["drop"]=:8} {steps1["raise"]=:8}   {steps2["fruitful"]=:8} {steps2["root"]=:8} {steps2["drop"]=:8} {steps2["raise"]=:8}")
            x = steps1["fruitful"]
            if x==0:
                assert steps2["root"]==0
                y = 0
            else:
                y = steps2["root"] / x
            data[k].append((x, y))
            totals[k].update(n=1, zeros=(steps2["root"] == 0),
                             bs_origin=x, bs_drop=steps1["drop"], bs_raise=steps1["raise"],
                             bc_origin=steps2["root"], bc_drop=steps2["drop"],
                             bc_raise=steps2["raise"], bc_fruitful=steps2["fruitful"])

    for i, k in enumerate(ks):
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.45, lw=0, color=cmap(i / (len(ks) - 1)), label=f"k={k}")

    ax.set_xscale("log")
    ax.set_xlim(1, 1.1e5)
    ax.set_xlabel("BS-DFS origin writes")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return totals


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)
    axes[0].set_ylabel("BC-DFS origin writes (fraction)")
    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    cmap = plt.get_cmap("plasma", len(K_VALUES))
    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("completeness.pdf", bbox_inches="tight")

    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


if __name__ == "__main__":
    main()
