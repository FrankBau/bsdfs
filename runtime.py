""" Experiment accompanying the paper, generates log and .pdf figure"""


from collections import Counter
import gc
import math
import statistics
import time
import random
import networkx as nx
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from bsdfs import bsdfs
from bcdfs import bcdfs

PASSES = 5

import sys

if any("pydevd" in m for m in sys.modules):
    print("### debug mode - reduced data set, for preview only ###")
    ISLICE = 10_000
    RUNS = 100
else:
    ISLICE = 100_000
    RUNS = 1_000
print(f"{RUNS=} {ISLICE=} {PASSES=}")

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


def check_perf_counter_resolution():
    try:
        info = time.get_clock_info('perf_counter')
        print(info)
        if info.resolution > 1e-07:
            print("### check_perf_counter_resolution: resolution may be insufficient ###")
    except Exception as e:
        print(f"Error retrieving clock info: {e}")


def get_runtime(G, s, t, k, algo):
    gc.disable()
    tick = time.perf_counter()
    n_paths = sum(1 for _ in islice(algo(G, s, t, k), ISLICE))
    dt = time.perf_counter() - tick
    gc.enable()
    return dt, n_paths


def get_runtimes(graph_generator):
    best = {}   # (run, k) -> [t_bs, t_bc, p_bs, p_bc]
    truncated = {k: 0 for k in K_VALUES}

    for p in range(PASSES):
        for run in range(RUNS):
            G, s, t = graph_generator(run)
            for k in K_VALUES:
                key = (run, k)
                if p > 0 and key not in best:
                    continue # truncated on pass 0
                tb, pb = get_runtime(G, s, t, k, bsdfs)
                if p == 0 and pb >= ISLICE:
                    truncated[k] += 1
                    continue
                tc, pc = get_runtime(G, s, t, k, bcdfs)
                if key in best:
                    best[key][0] = min(best[key][0], tb)
                    best[key][1] = min(best[key][1], tc)
                else:
                    best[key] = [tb, tc, pb, pc]
    return best, truncated


def print_totals(title, totals):
    print(f"\n--- {title}: totals ---")
    print(f"{'k':>3} {'n':>6} {'total bs/bc':>12} {'per-output':>11} {'median inst.':>13}")
    for k in K_VALUES:
        c = totals[k]
        if not c["n"]:
            continue
        tot = c["t_bs"]/c["t_bc"]
        per = (c["t_bs"]/c["out_bs"]) / (c["t_bc"]/c["out_bc"])
        print(f"{k:>3} {c['n']:6,} {tot:12.3f} {per:11.3f} {c['median']:13.3f}")


def make_ax(ax, title, graph_generator):
    ks = list(K_VALUES)

    best, truncated = get_runtimes(graph_generator)
    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}
    for run in range(RUNS):
        for k in K_VALUES:
            key = (run, k)
            if key not in best:
                continue
            runtime1, runtime2, len_paths1, len_paths2 =  best[key]
            intervals1 = 1 + len_paths1
            intervals2 = 1 + len_paths2
            x = runtime2 / intervals2
            y = runtime1 / intervals1
            data[k].append((x, y/x))
            totals[k].update(n=1, t_bs=runtime1, t_bc=runtime2, out_bs=intervals1, out_bc=intervals2)
            print(f"{run=:8} {k=:4}   {intervals1=:10} {intervals2=:10}   {runtime1=:12.9f} {runtime2=:12.9f}")

    for i, k in enumerate(ks):
        if not data[k]:
            continue
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.45, lw=0, color=CMAP(i), label=f"k={k}")

    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="lightgray", lw=1, ls="--")

    for k in ks:
        if data[k]:
            totals[k]["median"] = statistics.median(y for _, y in data[k])
    return totals


def make_figure():
    plt.rcParams.update({"pdf.fonttype": 42}) # Type 42 (TrueType) makes the figure text searchable and selectable
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)
    
    fig.supxlabel("BC-DFS runtime per interval [s]")
    axes[0].set_ylabel("runtime ratio BS-DFS / BC-DFS")
    
    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws)

    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), CMAP.N)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig(f"runtime.pdf", bbox_inches="tight")

    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


def main():
    check_perf_counter_resolution()
    estimate_overhead(gen_er)
    make_figure()


def estimate_overhead(graph_generator):

    from statistics import median, quantiles
    def null(G, s, t, k):
        return []

    timings = []
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in K_VALUES:
            dt, n_paths = get_runtime(G, s, t, k, null)
            timings += [dt]
    q = quantiles(timings, n=100)
    print(f"estimate_overhead:  min {min(timings):.9f}, median {median(timings):.9f} s, p95 {q[94]:.9f}, p99 {q[98]:.9f}, max {max(timings):.9f}")
    hiccups = sum(1 for t in timings if t > 1e-6)
    print(f"  measurements above 1 us: {hiccups / len(timings):%}")
    return median(timings)


if __name__ == "__main__":
    main()
