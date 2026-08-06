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

REPEATS = 5

import sys

if any("pydevd" in m for m in sys.modules):
    print("### debug mode - reduced data set, for preview only ###")
    ISLICE = 10_000
    RUNS = 100
else:
    ISLICE = 100_000
    RUNS = 1_000
print(f"{RUNS=} {ISLICE=} {REPEATS=}")

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


def check_perf_counter_resolution():
    try:
        info = time.get_clock_info('perf_counter')
        print(info)
        if info.resolution > 1e-07:
            print("### check_perf_counter_resolution: resolution may be insufficient ###")
    except Exception as e:
        print(f"Error retrieving clock info: {e}")


def get_runtime(G, s, t, k, algo):
    dt = []
    for _ in range(REPEATS):
        gc.disable()
        tick = time.perf_counter()
        n_paths = sum(1 for _ in islice(algo(G, s, t, k), ISLICE))
        dt += [time.perf_counter() - tick]
        gc.enable()
    return min(dt), n_paths


def print_totals(title, totals):
    print(f"\n--- {title}: totals ---")
    print(f"{'k':>3} {'n':>6} {'total bc/bs':>12} {'per-output':>11} {'median inst.':>13}")
    for k in K_VALUES:
        c = totals[k]
        tot = c["t_bc"]/c["t_bs"]
        per = (c["t_bc"]/c["out_bc"]) / (c["t_bs"]/c["out_bs"])
        print(f"{k:>3} {c['n']:6,} {tot:12.3f} {per:11.3f} {c['median']:13.3f}")


def make_ax(ax, title, graph_generator, relative=False):
    cmap = plt.get_cmap("plasma")
    ks = list(K_VALUES)

    data = {k: [] for k in K_VALUES}
    totals = {k: Counter() for k in K_VALUES}
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in K_VALUES:
            runtimeX, len_pathsX = get_runtime(G, s, t, k, bsdfs)
            if len_pathsX >= ISLICE:
                totals[k].update(truncated=1)
                continue                          # skip before timing BC-DFS
            runtimeY, len_pathsY = get_runtime(G, s, t, k, bcdfs)
            intervalsX = 1 + len_pathsX
            intervalsY = 1 + len_pathsY
            x = runtimeX / intervalsX if relative else runtimeX
            y = runtimeY / intervalsY if relative else runtimeY
            data[k].append((x, y/x))
            totals[k].update(n=1, t_bs=runtimeX, t_bc=runtimeY, out_bs=intervalsX, out_bc=intervalsY)
            print(f"{run=:8} {n=:4} {m=:4} {k=:4}   {intervalsX=:10} {intervalsY=:10}   {runtimeX=:12.9f} runtimeY={y:12.9f}")


    for i, k in enumerate(ks):
        xs, ys = zip(*data[k])
        ax.scatter(xs, ys, s=4, alpha=0.45, lw=0, color=cmap(i / (len(ks) - 1)), label=f"k={k}")

    ax.set_xscale("log")
    ax.set_xlabel("BS-DFS runtime per interval" if relative else "BS-DFS runtime (absolute)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(1, color="lightgray", lw=1, ls="--")

    for k in ks:
        if data[k]:
            totals[k]["median"] = statistics.median(y for _, y in data[k])
    return totals


def make_figure(title, relative):
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3), sharey=True, constrained_layout=True)
    axes[0].set_ylabel("runtime ratio BC-DFS / BS-DFS")

    totals_er = make_ax(axes[0], "Erdős–Rényi", gen_er, relative=relative)
    totals_ws = make_ax(axes[1], "Watts–Strogatz", gen_ws, relative=relative)

    cmap = plt.get_cmap("plasma", len(K_VALUES))
    norm = BoundaryNorm(np.arange(min(K_VALUES)-.5, max(K_VALUES)+1.5, 1), cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=K_VALUES, pad=.02, fraction=.035)
    cb.set_label("hop bound $k$")

    fig.savefig(f"{title}.pdf", bbox_inches="tight")

    print_totals("Erdős–Rényi", totals_er)
    print_totals("Watts–Strogatz", totals_ws)


def main():
    check_perf_counter_resolution()
    make_figure("runtime", relative=False)


def estimate_overhead(graph_generator):
    
    from statistics import median, stdev
    def null(G, s, t, k):
        return []

    timings = []
    for run in range(RUNS):
        G, s, t = graph_generator(run)
        n, m = G.number_of_nodes(), G.number_of_edges()
        for k in K_VALUES:
            dt, n_paths = get_runtime(G, s, t, k, null)
            timings += [dt]
    print(f"estimate_overhead: {median(timings)=:12.9f} s; {stdev(timings)=:12.9f} s.")
    return median(timings)


if __name__ == "__main__":
    estimate_overhead(gen_er)
    main()
