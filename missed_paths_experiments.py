import math
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


def run_erdos_renyi(runs, k_values, processes=None):
    print(f"\n=== Erdos-Renyi ===")
    print_header()

    for k in k_values:
        tasks = [(k, run) for run in range(runs)]

        with Pool(processes=processes) as pool:
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
    run_erdos_renyi(runs=base.RUNS, k_values=base.K_VALUES, processes=None)
    run_watts_strogatz(runs=base.RUNS, k_values=base.K_VALUES, processes=None)


