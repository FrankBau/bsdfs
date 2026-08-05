import statistics
from multiprocessing import Pool
from tqdm import tqdm

import bsdfs
import bsdfs_trivial
import bcdfs
import experiments_base as base


def worker_ws(args):
    k, run = args
    G, s, t = base.make_watts_strogatz(run)
    P_gt = set(map(tuple, bsdfs_trivial.bsdfs(G, s, t, k)))
    P_bs = set(map(tuple, bsdfs.bsdfs(G, s, t, k)))
    P_bc = set(map(tuple, bcdfs.bcdfs(G, s, t, k)))
    complete = not (P_bs ^ P_gt)
    assert complete, (G.edges, s, t, k, P_bs ^ P_gt)
    missed = P_gt - P_bc  # the real Δ
    spurious = P_bc - P_bs  # must be empty if BC-DFS is sound
    assert not spurious, (G.edges, s, t, k, spurious)
    return len(P_bs), len(missed)


def worker_er(args):
    k, run = args
    G, s, t = base.make_erdos_renyi(run)
    P_gt = set(map(tuple, bsdfs_trivial.bsdfs(G, s, t, k)))
    P_bs = set(map(tuple, bsdfs.bsdfs(G, s, t, k)))
    P_bc = set(map(tuple, bcdfs.bcdfs(G, s, t, k)))
    complete = not (P_bs ^ P_gt)
    assert complete, (G.edges, s, t, k, P_bs ^ P_gt)
    missed = P_gt - P_bc    # the real Δ
    spurious = P_bc - P_bs  # must be empty if BC-DFS is sound
    assert not spurious, (G.edges, s, t, k, spurious)
    return len(P_bs), len(missed)


# ------------------------------------------------------------
# Statistics aggregator
# ------------------------------------------------------------


def aggregate(results):
    counts  = [r[0] for r in results if r is not None]
    diffs   = [r[1] for r in results if r is not None]

    if not counts:
        return None

    return dict(
        n   =   len(counts),

        mean_count  = statistics.mean(counts),
        std_count   = statistics.stdev(counts),
        min_count   = min(counts),
        median_count= statistics.median(counts),
        max_count   = max(counts),

        mean_diff   = statistics.mean(diffs),
        std_diff    = statistics.stdev(diffs),
        min_diff    = min(diffs),
        median_diff = statistics.median(diffs),
        max_diff    = max(diffs),
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


def run_watts_strogatz(runs, k_values, processes=None):
    print("\n=== Watts-Strogatz ===")
    print_header()

    n = base.WS_N
    d = base.WS_D
    p = base.WS_P
    for k in k_values:
        tasks = [(k, run) for run in range(runs)]

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

import signal

def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_erdos_renyi(runs, k_values, processes=None):
    print(f"\n=== Erdos-Renyi ===")
    print_header()

    for k in k_values:
        tasks = [(k, run) for run in range(runs)]

        with Pool(processes=processes, initializer=ignore_sigint) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_er, tasks, chunksize=200),
                    total=runs,
                    desc=f"ER k={k}",
                    leave=False,
                )
            )

        print_row(f"{min(base.ER_N_VALUES)}-{max(base.ER_N_VALUES)}", "-", "rand", k, aggregate(results))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_erdos_renyi(runs=base.RUNS, k_values=base.K_VALUES, processes=None)
        run_watts_strogatz(runs=base.RUNS, k_values=base.K_VALUES, processes=None)
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)


# 2026-08-04
#
# === Erdos-Renyi ===
#      n    d      p    k       mean        std    min   median    max      meanΔ       stdΔ   minΔ     medΔ   maxΔ
# 6-30   -    rand      3       4.03       4.07      0     3.00     40       0.00       0.00      0     0.00      0                                                                                                       
# 6-30   -    rand      4      12.37      13.99      0     7.00    166       0.07       0.35      0     0.00      8                                                                                                       
# 6-30   -    rand      5      36.03      47.99      0    18.00    757       0.85       2.14      0     0.00     39                                                                                                       
# 6-30   -    rand      6     102.11     165.00      0    37.00   3391       4.73       9.44      0     1.00    177                                                                                                       
# 6-30   -    rand      7     285.99     560.59      0    67.00  13997      18.76      37.01      0     4.00    798                                                                                                       
# 6-30   -    rand      8     790.57    1864.89      0   114.00  54778      63.29     135.94      0    10.00   3546                                                                                                       
# 6-30   -    rand      9    2144.73    6040.03      0   172.00 199650     196.49     479.62      0    20.00  13886                                                                                                       
# 6-30   -    rand     10    5677.76   18963.36      0   240.00 679686     578.17    1629.47      0    33.00  49643                                                                                                       

# === Watts-Strogatz ===
#      n    d      p    k       mean        std    min   median    max      meanΔ       stdΔ   minΔ     medΔ   maxΔ
#   1000    6    0.2    3       0.19       0.93      0     0.00     15       0.00       0.00      0     0.00      0                                                                                                       
#   1000    6    0.2    4       0.91       2.92      0     0.00     50       0.02       0.20      0     0.00      6                                                                                                       
#   1000    6    0.2    5       4.32       9.16      0     1.00    157       0.19       1.04      0     0.00     26                                                                                                       
#   1000    6    0.2    6      20.52      29.30      0    11.00    505       1.51       4.48      0     0.00     80                                                                                                       
#   1000    6    0.2    7      97.19      96.05      0    71.00   1499      11.45      19.13      0     3.00    300 
  