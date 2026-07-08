import networkx as nx
import random
# from bcdfs import bcdfs
# from dfs import all_simple_paths as bcdfs
from bsdfs_trivial import bsdfs as bcdfs
from bsdfs import bsdfs
import math
import statistics
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from collections import deque

RUNS = 100_000
K_VALUES = range(3, 11)

def limited_descendants(G, s, k = math.inf):
    """All nodes within BFS distance <= k from s (s itself excluded)."""
    dist = {s: 0}
    q = deque([s])
    result = []
    succ = G.successors

    while q:
        u = q.popleft()
        if dist[u] == k:
            continue  # frontier reached; don't expand further
        for v in succ(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                result.append(v)
                q.append(v)
    return result


# ------------------------------------------------------------
# Graph factories
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
# Workers
# ------------------------------------------------------------


def worker_ws(args):
    """Watts-Strogatz worker: fixed n, d, p, k."""
    try:
        n, d, p, k, run = args
        seed = 42 + run
        G = make_watts_strogatz(n, d, p, seed)

        t = None
        while t is None:
            s = random.choice(sorted(G.nodes))
            reachable = limited_descendants(G, s)
            if reachable:
                t = random.choice(reachable)

        P_bs = set(map(tuple, bsdfs(G, s, t, k)))
        P_bc = set(map(tuple,  bcdfs(G, s, t, k)))
        missed = P_bs - P_bc  # the real Δ
        spurious = P_bc - P_bs  # must be empty if BC-DFS is sound
        assert not spurious, (G.edges, s, t, k, spurious)

        return len(P_bs), len(missed)

    except AssertionError:
        raise
    except Exception as e:
        print(f"Error in worker_ws: {e}")
        return None


def worker_er(args):
    """Erdos-Renyi worker: randomized n and p_edge, fixed k."""
    try:
        nmax, k, run = args
        seed = 42 + run
        random.seed(seed)
        np.random.seed(seed)

        n = random.randint(6, nmax)
        p_edge = random.uniform(2, 5) / (n - 1)

        G = make_erdos_renyi(n, p_edge, seed)

        t = None
        while t is None:
            s = random.choice(sorted(G.nodes))
            reachable = limited_descendants(G, s)
            if reachable:
                t = random.choice(reachable)

        P_bs = set(map(tuple, bsdfs(G, s, t, k)))
        P_bc = set(map(tuple, bcdfs(G, s, t, k)))
        missed = P_bs - P_bc  # the real Δ
        spurious = P_bc - P_bs  # must be empty if BC-DFS is sound
        assert not spurious, (G.edges, s, t, k, spurious)

        return len(P_bs), len(missed)

    except AssertionError:
        raise
    except Exception as e:
        print(f"Error in worker_er: {e}")
        return None


# ------------------------------------------------------------
# Statistics aggregator
# ------------------------------------------------------------


def aggregate(results):
    counts = [r[0] for r in results if r is not None]
    diffs = [r[1] for r in results if r is not None]

    if not counts:
        return None

    mean_count = sum(counts) / len(counts)
    mean_diff = sum(diffs) / len(diffs)
    std_count = math.sqrt(sum((x - mean_count) ** 2 for x in counts) / len(counts))
    std_diff = math.sqrt(sum((x - mean_diff) ** 2 for x in diffs) / len(diffs))

    return dict(
        n=len(counts),
        mean_count=mean_count,
        std_count=std_count,
        min_count=min(counts),
        median_count=statistics.median(counts),
        max_count=max(counts),
        mean_diff=mean_diff,
        std_diff=std_diff,
        min_diff=min(diffs),
        median_diff=statistics.median(diffs),
        max_diff=max(diffs),
    )


def print_header():
    print(
        f"{'n':>6} {'d':>4} {'p':>6} {'k':>4} "
        f"{'mean':>10} {'std':>10} {'min':>6} {'median':>8} {'max':>6} "
        f"{'meanΔ':>10} {'stdΔ':>10} {'minΔ':>6} {'medΔ':>8} {'maxΔ':>6}"
    )


def print_row(label_n, label_d, label_p, label_k, s):
    if s is None:
        print(
            f"{label_n:>6} {label_d:>4} {label_p:>6} {label_k:>4} -- no valid runs --"
        )
        return
    print(
        f"{label_n:6} {label_d:4} {label_p:6} {label_k:4} "
        f"{s['mean_count']:10.2f} {s['std_count']:10.2f} {s['min_count']:6} "
        f"{s['median_count']:8.2f} {s['max_count']:6} "
        f"{s['mean_diff']:10.2f} {s['std_diff']:10.2f} {s['min_diff']:6} "
        f"{s['median_diff']:8.2f} {s['max_diff']:6}"
    )


# ------------------------------------------------------------
# Watts-Strogatz experiment
# ------------------------------------------------------------


def run_watts_strogatz(
    runs=RUNS, n=1000, d=6, p=0.2, k_values=range(3, 10), processes=None
):
    print("\n=== Watts-Strogatz ===")
    print_header()

    for k in k_values:
        tasks = [(n, d, p, k, run) for run in range(runs)]

        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_ws, tasks, chunksize=200),
                    total=runs,
                    desc=f"WS k={k}",
                    leave=False,
                )
            )

        print_row(n, d, p, k, aggregate(results))


# ------------------------------------------------------------
# Erdos-Renyi experiment
# ------------------------------------------------------------


def run_erdos_renyi(runs=RUNS, nmax=30, k_values=range(3, 10), processes=None):
    print(f"\n=== Erdos-Renyi (nmax={nmax}) ===")
    print_header()

    for k in k_values:
        tasks = [(nmax, k, run) for run in range(runs)]

        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_er, tasks, chunksize=200),
                    total=runs,
                    desc=f"ER k={k}",
                    leave=False,
                )
            )

        print_row(f"<={nmax}", "-", "rand", k, aggregate(results))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":


    if True:
        run_erdos_renyi(runs=RUNS, nmax=30, k_values=K_VALUES, processes=None)
        
    if True:
        run_watts_strogatz(
            runs=RUNS, n=1000, d=6, p=0.2, k_values=K_VALUES, processes=None
        )


# 2026-05-22:
#
# === Erdos-Renyi (nmax=30) ===
#      n    d      p    k       mean        std    min   median    max      meanΔ       stdΔ   minΔ     medΔ   maxΔ
# <=30   -    rand      3       4.04       4.05      0     3.00     36       0.00       0.00      0     0.00      0                                                                                                                                                                                                                                             
# <=30   -    rand      4      12.48      13.90      0     7.00    162       0.06       0.34      0     0.00      9                                                                                                                                                                                                                                             
# <=30   -    rand      5      36.13      47.32      0    18.00    628       0.83       2.11      0     0.00     48                                                                                                                                                                                                                                             
# <=30   -    rand      6     101.59     161.38      0    39.00   2686       4.77       9.37      0     1.00    159                                                                                                                                                                                                                                             
# <=30   -    rand      7     282.16     544.40      0    73.00  11818      19.08      36.68      0     4.00    730                                                                                                                                                                                                                                             
# <=30   -    rand      8     773.50    1804.09      0   123.00  49929      64.58     135.73      0    11.00   3324                                                                                                                                                                                                                                             
# <=30   -    rand      9    2083.14    5850.74      0   187.00 196630     199.48     479.25      0    22.00  14341                                                                                                                                                                                                                                             
# <=30   -    rand     10    5487.56   18478.87      0   260.00 716992     583.22    1633.01      0    37.00  58740                
#
# === Watts-Strogatz ===
#      n    d      p    k       mean        std    min   median    max      meanΔ       stdΔ   minΔ     medΔ   maxΔ
#   1000    6    0.2    3       0.19       0.93      0     0.00     15       0.00       0.00      0     0.00      0                                                                                                                                                                                                                                             
#   1000    6    0.2    4       0.92       2.97      0     0.00     47       0.02       0.21      0     0.00      6                                                                                                                                                                                                                                             
#   1000    6    0.2    5       4.36       9.35      0     1.00    165       0.18       1.04      0     0.00     26                                                                                                                                                                                                                                             
#   1000    6    0.2    6      20.65      29.72      0    11.00    483       1.50       4.48      0     0.00     73                                                                                                                                                                                                                                             
#   1000    6    0.2    7      97.57      96.61      0    71.00   1610      11.45      19.14      0     3.00    251                                                                                                                                                                                                                                             
#   1000    6    0.2    8     460.75     325.43      0   389.00   5001      85.53      85.38      0    61.00   1017                                                                                                                                                                                                                                             
#   1000    6    0.2    9    2175.38    1154.83     29  1974.00  18218     586.25     391.52      0   512.00   4254                                                                                                                                                                                                                                             
#   1000    6    0.2   10   10267.67    4394.89    413  9642.00  57968    3477.20    1781.92      0  3218.00  16536                                                                                                                                                                                                                                             
