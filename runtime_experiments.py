# coarse compare of bcdfs vs. bsdfs total runtime for the experiments
import math
import random
import time
import gc
from collections import deque

import networkx as nx
import numpy as np
from tqdm import tqdm

from bsdfs import bsdfs
from bcdfs import bcdfs
from bcdfs_kickstart import bcdfs

RUNS = 100_000          # overnight run, start with a smaller pre-test
BATCH_SIZE = 1000
K_VALUES = range(3, 11)

# ER params
NMAX = 30

# WS params
WS_N = 1000
WS_D = 6
WS_P = 0.2

# null algorithm for estimating overhead
def null(G, s, t, k):
    return []

# MODE = [null, bsdfs, bcdfs]
MODE = [bcdfs]


# ------------------------------------------------------------
# Reachability (unchanged)
# ------------------------------------------------------------

def limited_descendants(G, s, k=math.inf):
    """All nodes within BFS distance <= k from s (s itself excluded)."""
    dist = {s: 0}
    q = deque([s])
    result = []
    succ = G.successors

    while q:
        u = q.popleft()
        if dist[u] == k:
            continue
        for v in succ(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                result.append(v)
                q.append(v)
    return result


def pick_target(G, s_pool):
    t = None
    while t is None:
        s = random.choice(s_pool)
        reachable = limited_descendants(G, s)
        if reachable:
            t = random.choice(reachable)
    return s, t


# ------------------------------------------------------------
# Graph factories (unchanged)
# ------------------------------------------------------------

def make_watts_strogatz(n, d, p, seed):
    random.seed(seed)
    np.random.seed(seed)
    H = nx.watts_strogatz_graph(n, d, p, seed=seed)
    return nx.DiGraph(H)


def make_erdos_renyi(n, p_edge, seed):
    random.seed(seed)
    np.random.seed(seed)
    return nx.gnp_random_graph(n, p_edge, seed=seed, directed=True)


# ------------------------------------------------------------
# Sample generators -- deterministic in `run` alone
# ------------------------------------------------------------

def er_sample(run, nmax=NMAX):
    seed = 42 + run
    random.seed(seed)
    np.random.seed(seed)
    n = random.randint(6, nmax)
    p_edge = random.uniform(2, 5) / (n - 1)
    G = make_erdos_renyi(n, p_edge, seed)
    s_pool = sorted(G.nodes)
    s, t = pick_target(G, s_pool)
    return G, s, t


def ws_sample(run, n=WS_N, d=WS_D, p=WS_P):
    seed = 42 + run
    G = make_watts_strogatz(n, d, p, seed)
    s_pool = sorted(G.nodes)
    s, t = pick_target(G, s_pool)
    return G, s, t


# ------------------------------------------------------------
# Coarse timing, batched: generate BATCH_SIZE samples up front
# (untimed), time the whole batch's algorithm work in one bracket,
# then do GC housekeeping at the batch boundary before moving on.
# This keeps to a handful of numbers per k (not per-sample sums) while
# bounding both peak memory (one batch of graphs, not the whole
# corpus) and how long garbage can accumulate before a collection
# pass gets a chance to run (one batch, not the whole sweep -- the
# whole-sweep-disabled version came within ~400MB of the commit
# ceiling on WS at RUNS=100_000).
# ------------------------------------------------------------

def run_sweep(algo_name, algo_call, sample_fn, runs, k_values, desc, batch_size=BATCH_SIZE):
    print(f"\n=== {desc}: {algo_name} ===")

    total_algo_time = 0.0
    total_paths = 0
    total_samples = 0
    per_k = []
    n_batches = math.ceil(runs / batch_size)
    for k in k_values:
        k_algo_time = 0.0
        k_paths = 0
        for bi in tqdm(range(n_batches), desc=f"{algo_name} k={k}", leave=False):
            run_start = bi * batch_size
            run_end = min(run_start + batch_size, runs)
            batch = [sample_fn(run) for run in range(run_start, run_end)]  # untimed

            gc.disable()
            try:
                t0 = time.perf_counter()
                for G, s, t in batch:
                    k_paths += sum(1 for _ in algo_call(G, s, t, k))
                k_algo_time += time.perf_counter() - t0
            finally:
                gc.enable()
                gc.collect()
            # batch goes out of scope here; garbage from it is reclaimed
            # by the gc.collect() above before the next batch starts

        per_k.append((k, k_algo_time, k_paths))
        total_algo_time += k_algo_time
        total_paths += k_paths
        total_samples += runs

        per_path_live = 1e6 * k_algo_time / k_paths if k_paths else float("nan")
        print(f"  k={k:2}: {k_algo_time:10.3f} s  {k_paths:12,} paths  {per_path_live:8.3f} us/path")

    print(f"{algo_name}: {total_algo_time:.3f} s total, algorithm only "
          f"({total_paths:,} paths, {total_samples:,} samples)")
    for k, k_algo_time, k_paths in per_k:
        share = 100 * k_algo_time / total_algo_time if total_algo_time > 0 else float("nan")
        per_path = 1e6 * k_algo_time / k_paths if k_paths else float("nan")
        print(f"  k={k:2}: {k_algo_time:10.3f} s ({share:5.1f}% of total)  "
              f"{k_paths:12,} paths  {per_path:8.3f} us/path")
    return total_algo_time


def run_experiment(sample_fn, runs, k_values, desc, mode=MODE):
    print(f"\n{desc}: run order = {mode}")
    results = {}
    for algo in mode:
        algo_name = algo.__name__
        results[algo_name] = run_sweep(algo_name, algo, sample_fn, runs, k_values, desc)

    if "bsdfs" in results and "bcdfs" in results:
        r = results["bsdfs"] / results["bcdfs"]
        print(f"{desc}: bsdfs {results['bsdfs']:.3f}s, bcdfs {results['bcdfs']:.3f}s, "
              f"ratio(bsdfs/bcdfs) = {r:.3f}")
    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(er_sample, RUNS, K_VALUES, f"Erdos-Renyi (nmax={NMAX})")
    run_experiment(ws_sample, RUNS, K_VALUES, f"Watts-Strogatz (n={WS_N}, d={WS_D}, p={WS_P})")

# Erdos-Renyi (nmax=30): run order = ['bsdfs', 'bcdfs']

# === Erdos-Renyi (nmax=30): bsdfs ===
#   k= 3:      0.150 s        40,674 paths     3.684 us/path                                                                                                                                                                                               
#   k= 4:      0.306 s       125,125 paths     2.444 us/path                                                                                                                                                                                               
#   k= 5:      0.733 s       361,389 paths     2.029 us/path                                                                                                                                                                                               
#   k= 6:      1.969 s     1,011,346 paths     1.947 us/path                                                                                                                                                                                               
#   k= 7:      5.582 s     2,803,115 paths     1.991 us/path                                                                                                                                                                                               
#   k= 8:     15.951 s     7,695,213 paths     2.073 us/path                                                                                                                                                                                               
#   k= 9:     45.736 s    20,799,271 paths     2.199 us/path                                                                                                                                                                                               
#   k=10:    127.467 s    55,036,331 paths     2.316 us/path                                                                                                                                                                                               
# bsdfs: 197.893 s total, algorithm only (87,872,464 paths, 80,000 samples)
#   k= 3:      0.150 s (  0.1% of total)        40,674 paths     3.684 us/path
#   k= 4:      0.306 s (  0.2% of total)       125,125 paths     2.444 us/path
#   k= 5:      0.733 s (  0.4% of total)       361,389 paths     2.029 us/path
#   k= 6:      1.969 s (  1.0% of total)     1,011,346 paths     1.947 us/path
#   k= 7:      5.582 s (  2.8% of total)     2,803,115 paths     1.991 us/path
#   k= 8:     15.951 s (  8.1% of total)     7,695,213 paths     2.073 us/path
#   k= 9:     45.736 s ( 23.1% of total)    20,799,271 paths     2.199 us/path
#   k=10:    127.467 s ( 64.4% of total)    55,036,331 paths     2.316 us/path

# === Erdos-Renyi (nmax=30): bcdfs ===
#   k= 3:      0.131 s        40,674 paths     3.216 us/path                                                                                                                                                                                               
#   k= 4:      0.273 s       124,504 paths     2.194 us/path                                                                                                                                                                                               
#   k= 5:      0.616 s       353,119 paths     1.745 us/path                                                                                                                                                                                               
#   k= 6:      1.535 s       964,481 paths     1.592 us/path                                                                                                                                                                                               
#   k= 7:      4.149 s     2,614,957 paths     1.587 us/path                                                                                                                                                                                               
#   k= 8:     11.681 s     7,064,190 paths     1.654 us/path                                                                                                                                                                                               
#   k= 9:     32.221 s    18,850,751 paths     1.709 us/path                                                                                                                                                                                               
#   k=10:     87.889 s    49,344,604 paths     1.781 us/path                                                                                                                                                                                               
# bcdfs: 138.495 s total, algorithm only (79,357,280 paths, 80,000 samples)
#   k= 3:      0.131 s (  0.1% of total)        40,674 paths     3.216 us/path
#   k= 4:      0.273 s (  0.2% of total)       124,504 paths     2.194 us/path
#   k= 5:      0.616 s (  0.4% of total)       353,119 paths     1.745 us/path
#   k= 6:      1.535 s (  1.1% of total)       964,481 paths     1.592 us/path
#   k= 7:      4.149 s (  3.0% of total)     2,614,957 paths     1.587 us/path
#   k= 8:     11.681 s (  8.4% of total)     7,064,190 paths     1.654 us/path
#   k= 9:     32.221 s ( 23.3% of total)    18,850,751 paths     1.709 us/path
#   k=10:     87.889 s ( 63.5% of total)    49,344,604 paths     1.781 us/path
# Erdos-Renyi (nmax=30): bsdfs 197.893s, bcdfs 138.495s, ratio(bsdfs/bcdfs) = 1.429

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = ['bsdfs', 'bcdfs']

# === Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs ===
#   k= 3:      1.041 s         1,745 paths   596.625 us/path                                                                                                                                                                                               
#   k= 4:      2.523 s         8,713 paths   289.620 us/path                                                                                                                                                                                               
#   k= 5:      6.111 s        42,528 paths   143.685 us/path                                                                                                                                                                                               
#   k= 6:     11.880 s       204,276 paths    58.158 us/path                                                                                                                                                                                               
#   k= 7:     19.878 s       973,065 paths    20.428 us/path                                                                                                                                                                                               
#   k= 8:     39.119 s     4,611,508 paths     8.483 us/path                                                                                                                                                                                               
#   k= 9:    113.428 s    21,788,877 paths     5.206 us/path                                                                                                                                                                                               
#   k=10:    441.414 s   102,818,606 paths     4.293 us/path                                                                                                                                                                                               
# bsdfs: 635.395 s total, algorithm only (130,449,318 paths, 80,000 samples)
#   k= 3:      1.041 s (  0.2% of total)         1,745 paths   596.625 us/path
#   k= 4:      2.523 s (  0.4% of total)         8,713 paths   289.620 us/path
#   k= 5:      6.111 s (  1.0% of total)        42,528 paths   143.685 us/path
#   k= 6:     11.880 s (  1.9% of total)       204,276 paths    58.158 us/path
#   k= 7:     19.878 s (  3.1% of total)       973,065 paths    20.428 us/path
#   k= 8:     39.119 s (  6.2% of total)     4,611,508 paths     8.483 us/path
#   k= 9:    113.428 s ( 17.9% of total)    21,788,877 paths     5.206 us/path
#   k=10:    441.414 s ( 69.5% of total)   102,818,606 paths     4.293 us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bcdfs ===
#   k= 3:      0.757 s         1,745 paths   433.860 us/path                                                                                                                                                                                               
#   k= 4:      1.832 s         8,572 paths   213.664 us/path                                                                                                                                                                                               
#   k= 5:      4.933 s        40,843 paths   120.773 us/path                                                                                                                                                                                               
#   k= 6:     11.415 s       189,770 paths    60.154 us/path                                                                                                                                                                                               
#   k= 7:     21.587 s       860,382 paths    25.091 us/path                                                                                                                                                                                               
#   k= 8:     37.980 s     3,763,511 paths    10.092 us/path                                                                                                                                                                                               
#   k= 9:     81.433 s    15,946,807 paths     5.107 us/path                                                                                                                                                                                               
#   k=10:    243.509 s    68,092,558 paths     3.576 us/path                                                                                                                                                                                               
# bcdfs: 403.446 s total, algorithm only (88,904,188 paths, 80,000 samples)
#   k= 3:      0.757 s (  0.2% of total)         1,745 paths   433.860 us/path
#   k= 4:      1.832 s (  0.5% of total)         8,572 paths   213.664 us/path
#   k= 5:      4.933 s (  1.2% of total)        40,843 paths   120.773 us/path
#   k= 6:     11.415 s (  2.8% of total)       189,770 paths    60.154 us/path
#   k= 7:     21.587 s (  5.4% of total)       860,382 paths    25.091 us/path
#   k= 8:     37.980 s (  9.4% of total)     3,763,511 paths    10.092 us/path
#   k= 9:     81.433 s ( 20.2% of total)    15,946,807 paths     5.107 us/path
#   k=10:    243.509 s ( 60.4% of total)    68,092,558 paths     3.576 us/path
# Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs 635.395s, bcdfs 403.446s, ratio(bsdfs/bcdfs) = 1.575

# Erdos-Renyi (nmax=30): run order = ['null']

# === Erdos-Renyi (nmax=30): null ===
#   k= 3:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 4:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 5:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 6:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 7:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 8:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 9:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
#   k=10:      0.002 s             0 paths       nan us/path                                                                                                                                                                                               
# null: 0.013 s total, algorithm only (0 paths, 80,000 samples)
#   k= 3:      0.002 s ( 12.4% of total)             0 paths       nan us/path
#   k= 4:      0.002 s ( 12.4% of total)             0 paths       nan us/path
#   k= 5:      0.002 s ( 12.4% of total)             0 paths       nan us/path
#   k= 6:      0.002 s ( 12.4% of total)             0 paths       nan us/path
#   k= 7:      0.002 s ( 12.4% of total)             0 paths       nan us/path
#   k= 8:      0.002 s ( 12.6% of total)             0 paths       nan us/path
#   k= 9:      0.002 s ( 12.9% of total)             0 paths       nan us/path
#   k=10:      0.002 s ( 12.6% of total)             0 paths       nan us/path

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = ['null']

# === Watts-Strogatz (n=1000, d=6, p=0.2): null ===
#   k= 3:      0.006 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 4:      0.008 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 5:      0.008 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 6:      0.008 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 7:      0.009 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 8:      0.009 s             0 paths       nan us/path                                                                                                                                                                                               
#   k= 9:      0.009 s             0 paths       nan us/path                                                                                                                                                                                               
#   k=10:      0.009 s             0 paths       nan us/path                                                                                                                                                                                               
# null: 0.065 s total, algorithm only (0 paths, 80,000 samples)
#   k= 3:      0.006 s (  9.9% of total)             0 paths       nan us/path
#   k= 4:      0.008 s ( 12.1% of total)             0 paths       nan us/path
#   k= 5:      0.008 s ( 12.5% of total)             0 paths       nan us/path
#   k= 6:      0.008 s ( 12.3% of total)             0 paths       nan us/path
#   k= 7:      0.009 s ( 13.2% of total)             0 paths       nan us/path
#   k= 8:      0.009 s ( 13.6% of total)             0 paths       nan us/path
#   k= 9:      0.009 s ( 13.5% of total)             0 paths       nan us/path
#   k=10:      0.009 s ( 13.0% of total)             0 paths       nan us/path

# 2nd run
# Erdos-Renyi (nmax=30): run order = ['bsdfs', 'bcdfs']

# === Erdos-Renyi (nmax=30): bsdfs ===
#   k= 3:      0.150 s        40,674 paths     3.698 us/path                                                                                                                                                                                               
#   k= 4:      0.304 s       125,125 paths     2.431 us/path                                                                                                                                                                                               
#   k= 5:      0.730 s       361,389 paths     2.021 us/path                                                                                                                                                                                               
#   k= 6:      2.132 s     1,011,346 paths     2.108 us/path                                                                                                                                                                                               
#   k= 7:      5.666 s     2,803,115 paths     2.021 us/path                                                                                                                                                                                               
#   k= 8:     16.182 s     7,695,213 paths     2.103 us/path                                                                                                                                                                                               
#   k= 9:     46.074 s    20,799,271 paths     2.215 us/path                                                                                                                                                                                               
#   k=10:    127.596 s    55,036,331 paths     2.318 us/path                                                                                                                                                                                               
# bsdfs: 198.836 s total, algorithm only (87,872,464 paths, 80,000 samples)
#   k= 3:      0.150 s (  0.1% of total)        40,674 paths     3.698 us/path
#   k= 4:      0.304 s (  0.2% of total)       125,125 paths     2.431 us/path
#   k= 5:      0.730 s (  0.4% of total)       361,389 paths     2.021 us/path
#   k= 6:      2.132 s (  1.1% of total)     1,011,346 paths     2.108 us/path
#   k= 7:      5.666 s (  2.8% of total)     2,803,115 paths     2.021 us/path
#   k= 8:     16.182 s (  8.1% of total)     7,695,213 paths     2.103 us/path
#   k= 9:     46.074 s ( 23.2% of total)    20,799,271 paths     2.215 us/path
#   k=10:    127.596 s ( 64.2% of total)    55,036,331 paths     2.318 us/path

# === Erdos-Renyi (nmax=30): bcdfs ===
#   k= 3:      0.136 s        40,674 paths     3.334 us/path                                                                                                                                                                                               
#   k= 4:      0.276 s       124,504 paths     2.218 us/path                                                                                                                                                                                               
#   k= 5:      0.628 s       353,119 paths     1.779 us/path                                                                                                                                                                                               
#   k= 6:      1.555 s       964,481 paths     1.613 us/path                                                                                                                                                                                               
#   k= 7:      4.200 s     2,614,957 paths     1.606 us/path                                                                                                                                                                                               
#   k= 8:     11.732 s     7,064,190 paths     1.661 us/path                                                                                                                                                                                               
#   k= 9:     32.106 s    18,850,751 paths     1.703 us/path                                                                                                                                                                                               
#   k=10:     87.169 s    49,344,604 paths     1.767 us/path                                                                                                                                                                                               
# bcdfs: 137.802 s total, algorithm only (79,357,280 paths, 80,000 samples)
#   k= 3:      0.136 s (  0.1% of total)        40,674 paths     3.334 us/path
#   k= 4:      0.276 s (  0.2% of total)       124,504 paths     2.218 us/path
#   k= 5:      0.628 s (  0.5% of total)       353,119 paths     1.779 us/path
#   k= 6:      1.555 s (  1.1% of total)       964,481 paths     1.613 us/path
#   k= 7:      4.200 s (  3.0% of total)     2,614,957 paths     1.606 us/path
#   k= 8:     11.732 s (  8.5% of total)     7,064,190 paths     1.661 us/path
#   k= 9:     32.106 s ( 23.3% of total)    18,850,751 paths     1.703 us/path
#   k=10:     87.169 s ( 63.3% of total)    49,344,604 paths     1.767 us/path
# Erdos-Renyi (nmax=30): bsdfs 198.836s, bcdfs 137.802s, ratio(bsdfs/bcdfs) = 1.443

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = ['bsdfs', 'bcdfs']

# === Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs ===
#   k= 3:      1.038 s         1,745 paths   594.632 us/path                                                                                                                                                                                               
#   k= 4:      2.635 s         8,713 paths   302.452 us/path                                                                                                                                                                                               
#   k= 5:      6.122 s        42,528 paths   143.954 us/path                                                                                                                                                                                               
#   k= 6:     11.795 s       204,276 paths    57.742 us/path                                                                                                                                                                                               
#   k= 7:     19.766 s       973,065 paths    20.313 us/path                                                                                                                                                                                               
#   k= 8:     38.668 s     4,611,508 paths     8.385 us/path                                                                                                                                                                                               
#   k= 9:    111.738 s    21,788,877 paths     5.128 us/path                                                                                                                                                                                               
#   k=10:    441.986 s   102,818,606 paths     4.299 us/path                                                                                                                                                                                               
# bsdfs: 633.749 s total, algorithm only (130,449,318 paths, 80,000 samples)
#   k= 3:      1.038 s (  0.2% of total)         1,745 paths   594.632 us/path
#   k= 4:      2.635 s (  0.4% of total)         8,713 paths   302.452 us/path
#   k= 5:      6.122 s (  1.0% of total)        42,528 paths   143.954 us/path
#   k= 6:     11.795 s (  1.9% of total)       204,276 paths    57.742 us/path
#   k= 7:     19.766 s (  3.1% of total)       973,065 paths    20.313 us/path
#   k= 8:     38.668 s (  6.1% of total)     4,611,508 paths     8.385 us/path
#   k= 9:    111.738 s ( 17.6% of total)    21,788,877 paths     5.128 us/path
#   k=10:    441.986 s ( 69.7% of total)   102,818,606 paths     4.299 us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bcdfs ===
#   k= 3:      0.761 s         1,745 paths   436.291 us/path                                                                                                                                                                                               
#   k= 4:      1.908 s         8,572 paths   222.606 us/path                                                                                                                                                                                               
#   k= 5:      4.995 s        40,843 paths   122.294 us/path                                                                                                                                                                                               
#   k= 6:     11.385 s       189,770 paths    59.995 us/path                                                                                                                                                                                               
#   k= 7:     21.177 s       860,382 paths    24.614 us/path                                                                                                                                                                                               
#   k= 8:     37.502 s     3,763,511 paths     9.965 us/path                                                                                                                                                                                               
#   k= 9:     80.438 s    15,946,807 paths     5.044 us/path                                                                                                                                                                                               
#   k=10:    246.273 s    68,092,558 paths     3.617 us/path                                                                                                                                                                                               
# bcdfs: 404.439 s total, algorithm only (88,904,188 paths, 80,000 samples)
#   k= 3:      0.761 s (  0.2% of total)         1,745 paths   436.291 us/path
#   k= 4:      1.908 s (  0.5% of total)         8,572 paths   222.606 us/path
#   k= 5:      4.995 s (  1.2% of total)        40,843 paths   122.294 us/path
#   k= 6:     11.385 s (  2.8% of total)       189,770 paths    59.995 us/path
#   k= 7:     21.177 s (  5.2% of total)       860,382 paths    24.614 us/path
#   k= 8:     37.502 s (  9.3% of total)     3,763,511 paths     9.965 us/path
#   k= 9:     80.438 s ( 19.9% of total)    15,946,807 paths     5.044 us/path
#   k=10:    246.273 s ( 60.9% of total)    68,092,558 paths     3.617 us/path
# Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs 633.749s, bcdfs 404.439s, ratio(bsdfs/bcdfs) = 1.567

############################# 100k ###########################
# C:\Users\frank\OneDrive\2026-06 path finding\code>python -u -OO runtime_comparison.py

# Erdos-Renyi (nmax=30): run order = ['null', 'bsdfs', 'bcdfs']

# === Erdos-Renyi (nmax=30): null ===
#   k= 3:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 4:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 5:      0.017 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 6:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 7:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 8:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 9:      0.017 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k=10:      0.018 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
# null: 0.142 s total, algorithm only (0 paths, 800,000 samples)
#   k= 3:      0.018 s ( 12.6% of total)             0 paths       nan us/path
#   k= 4:      0.018 s ( 12.8% of total)             0 paths       nan us/path
#   k= 5:      0.017 s ( 12.2% of total)             0 paths       nan us/path
#   k= 6:      0.018 s ( 12.3% of total)             0 paths       nan us/path
#   k= 7:      0.018 s ( 12.8% of total)             0 paths       nan us/path
#   k= 8:      0.018 s ( 12.7% of total)             0 paths       nan us/path
#   k= 9:      0.017 s ( 12.3% of total)             0 paths       nan us/path
#   k=10:      0.018 s ( 12.4% of total)             0 paths       nan us/path

# === Erdos-Renyi (nmax=30): bsdfs ===
#   k= 3:      1.515 s       404,477 paths     3.745 us/path                                                                                                                                                                                                                                                              
#   k= 4:      3.122 s     1,247,526 paths     2.503 us/path                                                                                                                                                                                                                                                              
#   k= 5:      7.628 s     3,612,521 paths     2.112 us/path                                                                                                                                                                                                                                                              
#   k= 6:     20.489 s    10,159,470 paths     2.017 us/path                                                                                                                                                                                                                                                              
#   k= 7:     57.970 s    28,215,982 paths     2.055 us/path                                                                                                                                                                                                                                                              
#   k= 8:    165.391 s    77,350,381 paths     2.138 us/path                                                                                                                                                                                                                                                              
#   k= 9:    462.680 s   208,314,173 paths     2.221 us/path                                                                                                                                                                                                                                                              
#   k=10:   1278.138 s   548,756,432 paths     2.329 us/path                                                                                                                                                                                                                                                              
# bsdfs: 1996.933 s total, algorithm only (878,060,962 paths, 800,000 samples)
#   k= 3:      1.515 s (  0.1% of total)       404,477 paths     3.745 us/path
#   k= 4:      3.122 s (  0.2% of total)     1,247,526 paths     2.503 us/path
#   k= 5:      7.628 s (  0.4% of total)     3,612,521 paths     2.112 us/path
#   k= 6:     20.489 s (  1.0% of total)    10,159,470 paths     2.017 us/path
#   k= 7:     57.970 s (  2.9% of total)    28,215,982 paths     2.055 us/path
#   k= 8:    165.391 s (  8.3% of total)    77,350,381 paths     2.138 us/path
#   k= 9:    462.680 s ( 23.2% of total)   208,314,173 paths     2.221 us/path
#   k=10:   1278.138 s ( 64.0% of total)   548,756,432 paths     2.329 us/path

# === Erdos-Renyi (nmax=30): bcdfs ===
#   k= 3:      1.282 s       404,477 paths     3.168 us/path                                                                                                                                                                                                                                                              
#   k= 4:      2.675 s     1,241,188 paths     2.155 us/path                                                                                                                                                                                                                                                              
#   k= 5:      6.065 s     3,529,691 paths     1.718 us/path                                                                                                                                                                                                                                                              
#   k= 6:     15.420 s     9,682,049 paths     1.593 us/path                                                                                                                                                                                                                                                              
#   k= 7:     41.672 s    26,308,384 paths     1.584 us/path                                                                                                                                                                                                                                                              
#   k= 8:    116.954 s    70,892,691 paths     1.650 us/path                                                                                                                                                                                                                                                              
#   k= 9:    318.621 s   188,366,292 paths     1.691 us/path                                                                                                                                                                                                                                                              
#   k=10:    859.872 s   490,434,617 paths     1.753 us/path                                                                                                                                                                                                                                                              
# bcdfs: 1362.560 s total, algorithm only (790,859,389 paths, 800,000 samples)
#   k= 3:      1.282 s (  0.1% of total)       404,477 paths     3.168 us/path
#   k= 4:      2.675 s (  0.2% of total)     1,241,188 paths     2.155 us/path
#   k= 5:      6.065 s (  0.4% of total)     3,529,691 paths     1.718 us/path
#   k= 6:     15.420 s (  1.1% of total)     9,682,049 paths     1.593 us/path
#   k= 7:     41.672 s (  3.1% of total)    26,308,384 paths     1.584 us/path
#   k= 8:    116.954 s (  8.6% of total)    70,892,691 paths     1.650 us/path
#   k= 9:    318.621 s ( 23.4% of total)   188,366,292 paths     1.691 us/path
#   k=10:    859.872 s ( 63.1% of total)   490,434,617 paths     1.753 us/path
# Erdos-Renyi (nmax=30): bsdfs 1996.933s, bcdfs 1362.560s, ratio(bsdfs/bcdfs) = 1.466

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = ['null', 'bsdfs', 'bcdfs']

# === Watts-Strogatz (n=1000, d=6, p=0.2): null ===
#   k= 3:      0.076 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 4:      0.084 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 5:      0.088 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 6:      0.088 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 7:      0.082 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 8:      0.084 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k= 9:      0.086 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
#   k=10:      0.085 s             0 paths       nan us/path                                                                                                                                                                                                                                                              
# null: 0.674 s total, algorithm only (0 paths, 800,000 samples)
#   k= 3:      0.076 s ( 11.3% of total)             0 paths       nan us/path
#   k= 4:      0.084 s ( 12.5% of total)             0 paths       nan us/path
#   k= 5:      0.088 s ( 13.1% of total)             0 paths       nan us/path
#   k= 6:      0.088 s ( 13.1% of total)             0 paths       nan us/path
#   k= 7:      0.082 s ( 12.2% of total)             0 paths       nan us/path
#   k= 8:      0.084 s ( 12.5% of total)             0 paths       nan us/path
#   k= 9:      0.086 s ( 12.8% of total)             0 paths       nan us/path
#   k=10:      0.085 s ( 12.6% of total)             0 paths       nan us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs ===
#   k= 3:      9.957 s        18,865 paths   527.822 us/path                                                                                                                                                                                                                                                              
#   k= 4:     25.240 s        91,501 paths   275.841 us/path                                                                                                                                                                                                                                                              
#   k= 5:     60.537 s       436,091 paths   138.817 us/path                                                                                                                                                                                                                                                              
#   k= 6:    116.291 s     2,065,053 paths    56.314 us/path                                                                                                                                                                                                                                                              
#   k= 7:    195.044 s     9,756,847 paths    19.990 us/path                                                                                                                                                                                                                                                              
#   k= 8:    382.352 s    46,074,726 paths     8.299 us/path                                                                                                                                                                                                                                                              
#   k= 9:   1107.668 s   217,538,216 paths     5.092 us/path                                                                                                                                                                                                                                                              
#   k=10:   4419.422 s  1,026,766,923 paths     4.304 us/path                                                                                                                                                                                                                                                             
# bsdfs: 6316.511 s total, algorithm only (1,302,748,222 paths, 800,000 samples)
#   k= 3:      9.957 s (  0.2% of total)        18,865 paths   527.822 us/path
#   k= 4:     25.240 s (  0.4% of total)        91,501 paths   275.841 us/path
#   k= 5:     60.537 s (  1.0% of total)       436,091 paths   138.817 us/path
#   k= 6:    116.291 s (  1.8% of total)     2,065,053 paths    56.314 us/path
#   k= 7:    195.044 s (  3.1% of total)     9,756,847 paths    19.990 us/path
#   k= 8:    382.352 s (  6.1% of total)    46,074,726 paths     8.299 us/path
#   k= 9:   1107.668 s ( 17.5% of total)   217,538,216 paths     5.092 us/path
#   k=10:   4419.422 s ( 70.0% of total)  1,026,766,923 paths     4.304 us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bcdfs ===
#   k= 3:      7.333 s        18,865 paths   388.698 us/path                                                                                                                                                                                                                                                              
#   k= 4:     18.563 s        89,884 paths   206.518 us/path                                                                                                                                                                                                                                                              
#   k= 5:     49.546 s       417,642 paths   118.632 us/path                                                                                                                                                                                                                                                              
#   k= 6:    112.876 s     1,914,909 paths    58.946 us/path                                                                                                                                                                                                                                                              
#   k= 7:    208.583 s     8,611,408 paths    24.222 us/path                                                                                                                                                                                                                                                              
#   k= 8:    370.030 s    37,521,567 paths     9.862 us/path                                                                                                                                                                                                                                                              
#   k= 9:    815.500 s   158,913,709 paths     5.132 us/path                                                                                                                                                                                                                                                              
#   k=10:   2441.160 s   679,046,608 paths     3.595 us/path                                                                                                                                                                                                                                                              
# bcdfs: 4023.589 s total, algorithm only (886,534,592 paths, 800,000 samples)
#   k= 3:      7.333 s (  0.2% of total)        18,865 paths   388.698 us/path
#   k= 4:     18.563 s (  0.5% of total)        89,884 paths   206.518 us/path
#   k= 5:     49.546 s (  1.2% of total)       417,642 paths   118.632 us/path
#   k= 6:    112.876 s (  2.8% of total)     1,914,909 paths    58.946 us/path
#   k= 7:    208.583 s (  5.2% of total)     8,611,408 paths    24.222 us/path
#   k= 8:    370.030 s (  9.2% of total)    37,521,567 paths     9.862 us/path
#   k= 9:    815.500 s ( 20.3% of total)   158,913,709 paths     5.132 us/path
#   k=10:   2441.160 s ( 60.7% of total)   679,046,608 paths     3.595 us/path
# Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs 6316.511s, bcdfs 4023.589s, ratio(bsdfs/bcdfs) = 1.570
    
    
    
#################################### kick-start fixed bcdfs ####################################
# Erdos-Renyi (nmax=30): run order = [<function bcdfs at 0x0000024BB016C0F0>]

# === Erdos-Renyi (nmax=30): bcdfs ===
#   k= 3:      1.581 s       404,477 paths     3.910 us/path                                                                                                                                                                                                                                             
#   k= 4:      3.487 s     1,247,526 paths     2.795 us/path                                                                                                                                                                                                                                             
#   k= 5:      8.710 s     3,612,521 paths     2.411 us/path                                                                                                                                                                                                                                             
#   k= 6:     23.772 s    10,159,470 paths     2.340 us/path                                                                                                                                                                                                                                             
#   k= 7:     67.678 s    28,215,982 paths     2.399 us/path                                                                                                                                                                                                                                             
#   k= 8:    194.660 s    77,350,381 paths     2.517 us/path                                                                                                                                                                                                                                             
#   k= 9:    543.100 s   208,314,173 paths     2.607 us/path                                                                                                                                                                                                                                             
#   k=10:   1517.799 s   548,756,432 paths     2.766 us/path                                                                                                                                                                                                                                             
# bcdfs: 2360.787 s total, algorithm only (878,060,962 paths, 800,000 samples)
#   k= 3:      1.581 s (  0.1% of total)       404,477 paths     3.910 us/path
#   k= 4:      3.487 s (  0.1% of total)     1,247,526 paths     2.795 us/path
#   k= 5:      8.710 s (  0.4% of total)     3,612,521 paths     2.411 us/path
#   k= 6:     23.772 s (  1.0% of total)    10,159,470 paths     2.340 us/path
#   k= 7:     67.678 s (  2.9% of total)    28,215,982 paths     2.399 us/path
#   k= 8:    194.660 s (  8.2% of total)    77,350,381 paths     2.517 us/path
#   k= 9:    543.100 s ( 23.0% of total)   208,314,173 paths     2.607 us/path
#   k=10:   1517.799 s ( 64.3% of total)   548,756,432 paths     2.766 us/path

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = [<function bcdfs at 0x0000024BB016C0F0>]

# === Watts-Strogatz (n=1000, d=6, p=0.2): bcdfs ===
#   k= 3:      7.393 s        18,865 paths   391.874 us/path                                                                                                                                                                                                                                             
#   k= 4:     18.635 s        91,501 paths   203.663 us/path                                                                                                                                                                                                                                             
#   k= 5:     50.157 s       436,091 paths   115.015 us/path                                                                                                                                                                                                                                             
#   k= 6:    117.147 s     2,065,053 paths    56.728 us/path                                                                                                                                                                                                                                             
#   k= 7:    229.974 s     9,756,847 paths    23.570 us/path                                                                                                                                                                                                                                             
#   k= 8:    489.974 s    46,074,726 paths    10.634 us/path                                                                                                                                                                                                                                             
#   k= 9:   1434.766 s   217,538,216 paths     6.595 us/path  
  