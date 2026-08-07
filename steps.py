"""
The cost of replacing BC-DFS by BS-DFS

Both algorithms are counted in the elementary-step model of the paper
(sec:work-attribution), decomposed into three accounts:

    scan     1 (call bookkeeping) + 1 per scanned successor
    cascade  1 per firing barrier update + 1 per scanned predecessor
    output   |path| <= k+1

A non-firing UpdateBarrier costs nothing extra: examining it was already
paid for by the caller's predecessor scan.

The two algorithms differ in bookkeeping, and this is counted honestly:
BC-DFS enters t as a search call and skips the successor loop at depth k;
BS-DFS scans successors in every call and handles t inside the parent's
scan.  These are differences in the algorithms, not in the measurement.

Calibration: BS-DFS reproduces the exact step counts of lem:clique-trap,
c*N_{c,k} for G_c and c*N_{c,k}+c^2-1 for the terminal interval of G_c^+.
"""

import math
import random
import networkx as nx
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from collections import Counter, deque, defaultdict

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


def bsdfs(G, s, t, k, steps):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            steps["cascade"] += 1
            for p in G.predecessors(q):
                steps["cascade"] += 1
                if p not in S and b[p] > d + 1:
                    steps["drop"] += 1
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1
        steps["scan"] += 1
        sd = k + 1
        for w in G.successors(v):
            steps["scan"] += 1
            if b[w] + h < k:
                if w == t:
                    steps["output"] += h + 2
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
            steps["cascade"] += 1
            steps["root" if root_call else "drop"] += 1
            bar[u] = l
            for v in G.predecessors(u):
                steps["cascade"] += 1
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        F = k + 1
        S.append(u)
        steps["scan"] += 1
        if u == t:
            steps["output"] += len(S)
            yield S.copy()
            S.pop()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                steps["scan"] += 1
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


ACCOUNTS = ("scan", "cascade", "output")


def total_steps(steps):
    return sum(steps[a] for a in ACCOUNTS)


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
    print(f"\n--- {title}: elementary steps, BS-DFS relative to BC-DFS ---")
    print(f"{'k':>3} {'n':>6} {'BS steps':>14} {'BC steps':>14} {'bs/bc':>7}"
          f" {'scan':>7} {'cascade':>8} {'output':>7} {'worst':>7} {'fired':>7}")
    for k in K_VALUES:
        c = totals[k]
        if not c["n"] or not c["bs_total"]:
            print(f"{k:>3} {c['n']:6,} -- no work recorded --")
            continue
        rat = lambda a: c[f"bs_{a}"] / c[f"bc_{a}"] if c[f"bc_{a}"] else float("nan")
        fired = c["bc_root"] / c["bc_fruitful"] if c["bc_fruitful"] else float("nan")
        per = (c["bc_total"]/c["bc_int"]) / (c["bs_total"]/c["bs_int"]) # steps per interval for each algorithm, then their ratio
        print(f"{k:>3} {c['n']:6,} {c['bs_total']:14,} {c['bc_total']:14,}"
              f" {c['bs_total']/c['bc_total']:7.3f}"
              f" {rat('scan'):7.3f} {rat('cascade'):8.3f} {rat('output'):7.3f}"
              f" {c['worst']:7.3f} {fired:7.4f} {per:7.3f}")


def make_ax(ax, title, graph_generator):
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}

    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in ks:
            steps1 = defaultdict(int)
            paths1 = list(islice(bsdfs(G, s, t, k, steps1), ISLICE))
            if len(paths1) >= ISLICE:              # truncated: excluded
                totals[k].update(truncated=1)
                continue
            steps2 = defaultdict(int)
            paths2 = list(islice(bcdfs(G, s, t, k, steps2), ISLICE))
            iv1 = 1 + len(paths1)   # intervals
            iv2 = 1 + len(paths2)   # intervals
            s1 = total_steps(steps1)
            s2 = total_steps(steps2)
            x = s1 / iv1
            y = s2 / iv2
            print(f"{run=:8} {n=:4} {m=:4} {k=:4}   bs_steps={x:10.2f} bc_steps={y:10.2f}"
                  f"   bs_scan={steps1['scan']:9} bs_casc={steps1['cascade']:9} bs_out={steps1['output']:9}"
                  f"   bc_scan={steps2['scan']:9} bc_casc={steps2['cascade']:9} bc_out={steps2['output']:9}")
            data[k].append((y, x / y))
            totals[k].update(n=1, bs_total=s1, bc_total=s2,
                             bs_int=iv1, bc_int=iv2,
                             bc_fruitful=steps2["fruitful"], bc_root=steps2["root"],
                             **{f"bs_{a}": steps1[a] for a in ACCOUNTS},
                             **{f"bc_{a}": steps2[a] for a in ACCOUNTS})

    for i, k in enumerate(ks):
        if not data[k]:
            continue
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.45, lw=0, color=CMAP(i), label=f"k={k}")
        totals[k]["worst"] = max(ys)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="gray", lw=1, ls="--", zorder=0)

    return totals


def main():
    plt.rcParams.update({"pdf.fonttype": 42}) # Type 42 (TrueType) makes the figure text searchable and selectable
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)

    fig.supxlabel("BC-DFS elementary steps per interval")
    axes[0].set_ylabel("step ratio BS-DFS / BC-DFS")

    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig("steps.pdf", bbox_inches="tight")

    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


if __name__ == "__main__":
    main()