"""
coarse compare of bcdfs vs. bsdfs total runtime for the experiments.
single-core, long running
"""

import math
import time
import gc
from tqdm import tqdm

import experiments_base as base
import bcdfs
import bsdfs

RUNS = base.RUNS    # reduce for pre-tests
BATCH_SIZE = 1000


# null algorithm for estimating overhead
def null(G, s, t, k):
    return []

MODE = [null, bsdfs.bsdfs, bcdfs.bcdfs]


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
        print(f"  k={k:2}: {k_algo_time:10.3f} s ({share:5.1f}% of total)  {k_paths:12,} paths  {per_path:8.3f} us/path")
    return total_algo_time


def run_experiment(sample_fn, runs, k_values, desc, mode=MODE):
    print(f"\n{desc}: run order = {mode}")
    results = {}
    for algo in mode:
        algo_name = algo.__name__
        results[algo_name] = run_sweep(algo_name, algo, sample_fn, runs, k_values, desc)

    # subtract overhead
    results["bsdfs"] -= results["null"]
    results["bcdfs"] -= results["null"]

    if "bsdfs" in results and "bcdfs" in results:
        r = results["bsdfs"] / results["bcdfs"]
        print(f"{desc}: bsdfs {results['bsdfs']:.3f}s, bcdfs {results['bcdfs']:.3f}s, ratio(bsdfs/bcdfs) = {r:.3f}")
    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_experiment(base.make_erdos_renyi, RUNS, base.K_VALUES, f"Erdos-Renyi ({min(base.ER_N_VALUES)}..{max(base.ER_N_VALUES)})")
        run_experiment(base.make_watts_strogatz, RUNS, base.K_VALUES, f"Watts-Strogatz (n={base.WS_N}, d={base.WS_D}, p={base.WS_P})")
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)

# 2026-08-04
#

# Erdos-Renyi (6..30): run order = [<function null at 0x0000016B8973FAB0>, <function bsdfs at 0x0000016B8CADF7F0>, <function bcdfs at 0x0000016B8CADD9B0>]

# === Erdos-Renyi (6..30): null ===
#   k= 3:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k= 4:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k= 5:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k= 6:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k= 7:      0.018 s             0 paths       nan us/path                                                                                                                                                              
#   k= 8:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k= 9:      0.017 s             0 paths       nan us/path                                                                                                                                                              
#   k=10:      0.018 s             0 paths       nan us/path                                                                                                                                                              
# null: 0.139 s total, algorithm only (0 paths, 800,000 samples)
#   k= 3:      0.017 s ( 12.3% of total)             0 paths       nan us/path
#   k= 4:      0.017 s ( 12.3% of total)             0 paths       nan us/path
#   k= 5:      0.017 s ( 12.5% of total)             0 paths       nan us/path
#   k= 6:      0.017 s ( 12.6% of total)             0 paths       nan us/path
#   k= 7:      0.018 s ( 12.7% of total)             0 paths       nan us/path
#   k= 8:      0.017 s ( 12.4% of total)             0 paths       nan us/path
#   k= 9:      0.017 s ( 12.5% of total)             0 paths       nan us/path
#   k=10:      0.018 s ( 12.8% of total)             0 paths       nan us/path

# === Erdos-Renyi (6..30): bsdfs ===
#   k= 3:      1.469 s       402,585 paths     3.649 us/path                                                                                                                                                              
#   k= 4:      3.050 s     1,237,409 paths     2.465 us/path                                                                                                                                                              
#   k= 5:      7.337 s     3,602,698 paths     2.036 us/path                                                                                                                                                              
#   k= 6:     19.906 s    10,211,016 paths     1.949 us/path                                                                                                                                                              
#   k= 7:     56.796 s    28,598,970 paths     1.986 us/path                                                                                                                                                              
#   k= 8:    164.853 s    79,057,076 paths     2.085 us/path                                                                                                                                                              
#   k= 9:    467.489 s   214,472,635 paths     2.180 us/path                                                                                                                                                              
#   k=10:   1305.637 s   567,775,657 paths     2.300 us/path                                                                                                                                                              
# bsdfs: 2026.536 s total, algorithm only (905,358,046 paths, 800,000 samples)
#   k= 3:      1.469 s (  0.1% of total)       402,585 paths     3.649 us/path
#   k= 4:      3.050 s (  0.2% of total)     1,237,409 paths     2.465 us/path
#   k= 5:      7.337 s (  0.4% of total)     3,602,698 paths     2.036 us/path
#   k= 6:     19.906 s (  1.0% of total)    10,211,016 paths     1.949 us/path
#   k= 7:     56.796 s (  2.8% of total)    28,598,970 paths     1.986 us/path
#   k= 8:    164.853 s (  8.1% of total)    79,057,076 paths     2.085 us/path
#   k= 9:    467.489 s ( 23.1% of total)   214,472,635 paths     2.180 us/path
#   k=10:   1305.637 s ( 64.4% of total)   567,775,657 paths     2.300 us/path

# === Erdos-Renyi (6..30): bcdfs ===
#   k= 3:      1.630 s       402,585 paths     4.049 us/path                                                                                                                                                              
#   k= 4:      3.022 s     1,230,475 paths     2.456 us/path                                                                                                                                                              
#   k= 5:      6.233 s     3,517,764 paths     1.772 us/path                                                                                                                                                              
#   k= 6:     15.410 s     9,737,563 paths     1.583 us/path                                                                                                                                                              
#   k= 7:     41.849 s    26,723,015 paths     1.566 us/path                                                                                                                                                              
#   k= 8:    117.553 s    72,728,544 paths     1.616 us/path                                                                                                                                                              
#   k= 9:    322.684 s   194,823,481 paths     1.656 us/path                                                                                                                                                              
#   k=10:    882.786 s   509,958,546 paths     1.731 us/path                                                                                                                                                              
# bcdfs: 1391.167 s total, algorithm only (819,121,973 paths, 800,000 samples)
#   k= 3:      1.630 s (  0.1% of total)       402,585 paths     4.049 us/path
#   k= 4:      3.022 s (  0.2% of total)     1,230,475 paths     2.456 us/path
#   k= 5:      6.233 s (  0.4% of total)     3,517,764 paths     1.772 us/path
#   k= 6:     15.410 s (  1.1% of total)     9,737,563 paths     1.583 us/path
#   k= 7:     41.849 s (  3.0% of total)    26,723,015 paths     1.566 us/path
#   k= 8:    117.553 s (  8.4% of total)    72,728,544 paths     1.616 us/path
#   k= 9:    322.684 s ( 23.2% of total)   194,823,481 paths     1.656 us/path
#   k=10:    882.786 s ( 63.5% of total)   509,958,546 paths     1.731 us/path
# Erdos-Renyi (6..30): bsdfs 2026.397s, bcdfs 1391.029s, ratio(bsdfs/bcdfs) = 1.457

# Watts-Strogatz (n=1000, d=6, p=0.2): run order = [<function null at 0x0000016B8973FAB0>, <function bsdfs at 0x0000016B8CADF7F0>, <function bcdfs at 0x0000016B8CADD9B0>]

# === Watts-Strogatz (n=1000, d=6, p=0.2): null ===
#   k= 3:      0.082 s             0 paths       nan us/path                                                                                                                                                              
#   k= 4:      0.078 s             0 paths       nan us/path                                                                                                                                                              
#   k= 5:      0.078 s             0 paths       nan us/path                                                                                                                                                              
#   k= 6:      0.079 s             0 paths       nan us/path                                                                                                                                                              
#   k= 7:      0.081 s             0 paths       nan us/path                                                                                                                                                              
#   k= 8:      0.080 s             0 paths       nan us/path                                                                                                                                                              
#   k= 9:      0.080 s             0 paths       nan us/path                                                                                                                                                              
#   k=10:      0.080 s             0 paths       nan us/path                                                                                                                                                              
# null: 0.640 s total, algorithm only (0 paths, 800,000 samples)
#   k= 3:      0.082 s ( 12.9% of total)             0 paths       nan us/path
#   k= 4:      0.078 s ( 12.2% of total)             0 paths       nan us/path
#   k= 5:      0.078 s ( 12.2% of total)             0 paths       nan us/path
#   k= 6:      0.079 s ( 12.4% of total)             0 paths       nan us/path
#   k= 7:      0.081 s ( 12.7% of total)             0 paths       nan us/path
#   k= 8:      0.080 s ( 12.6% of total)             0 paths       nan us/path
#   k= 9:      0.080 s ( 12.5% of total)             0 paths       nan us/path
#   k=10:      0.080 s ( 12.5% of total)             0 paths       nan us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs ===
#   k= 3:      9.903 s        18,754 paths   528.040 us/path                                                                                                                                                              
#   k= 4:     24.938 s        90,541 paths   275.432 us/path                                                                                                                                                              
#   k= 5:     59.581 s       431,964 paths   137.931 us/path                                                                                                                                                              
#   k= 6:    114.313 s     2,051,942 paths    55.710 us/path                                                                                                                                                              
#   k= 7:    192.267 s     9,718,746 paths    19.783 us/path                                                                                                                                                              
#   k= 8:    379.508 s    45,964,706 paths     8.257 us/path                                                                                                                                                              
#   k= 9:   1079.841 s   217,177,858 paths     4.972 us/path                                                                                                                                                              
#   k=10:   4315.114 s  1,025,602,442 paths     4.207 us/path                                                                                                                                                             
# bsdfs: 6175.465 s total, algorithm only (1,301,056,953 paths, 800,000 samples)
#   k= 3:      9.903 s (  0.2% of total)        18,754 paths   528.040 us/path
#   k= 4:     24.938 s (  0.4% of total)        90,541 paths   275.432 us/path
#   k= 5:     59.581 s (  1.0% of total)       431,964 paths   137.931 us/path
#   k= 6:    114.313 s (  1.9% of total)     2,051,942 paths    55.710 us/path
#   k= 7:    192.267 s (  3.1% of total)     9,718,746 paths    19.783 us/path
#   k= 8:    379.508 s (  6.1% of total)    45,964,706 paths     8.257 us/path
#   k= 9:   1079.841 s ( 17.5% of total)   217,177,858 paths     4.972 us/path
#   k=10:   4315.114 s ( 69.9% of total)  1,025,602,442 paths     4.207 us/path

# === Watts-Strogatz (n=1000, d=6, p=0.2): bcdfs ===
#   k= 3:      7.258 s        18,754 paths   386.996 us/path                                                                                                                                                              
#   k= 4:     18.297 s        88,931 paths   205.738 us/path                                                                                                                                                              
#   k= 5:     48.718 s       413,425 paths   117.839 us/path                                                                                                                                                              
#   k= 6:    111.029 s     1,900,887 paths    58.409 us/path                                                                                                                                                              
#   k= 7:    206.485 s     8,573,864 paths    24.083 us/path                                                                                                                                                              
#   k= 8:    370.771 s    37,432,977 paths     9.905 us/path                                                                                                                                                              
#   k= 9:    797.497 s   158,620,043 paths     5.028 us/path                                                                                                                                                              
#   k=10:   2444.630 s   677,715,621 paths     3.607 us/path                                                                                                                                                              
# bcdfs: 4004.683 s total, algorithm only (884,764,502 paths, 800,000 samples)
#   k= 3:      7.258 s (  0.2% of total)        18,754 paths   386.996 us/path
#   k= 4:     18.297 s (  0.5% of total)        88,931 paths   205.738 us/path
#   k= 5:     48.718 s (  1.2% of total)       413,425 paths   117.839 us/path
#   k= 6:    111.029 s (  2.8% of total)     1,900,887 paths    58.409 us/path
#   k= 7:    206.485 s (  5.2% of total)     8,573,864 paths    24.083 us/path
#   k= 8:    370.771 s (  9.3% of total)    37,432,977 paths     9.905 us/path
#   k= 9:    797.497 s ( 19.9% of total)   158,620,043 paths     5.028 us/path
#   k=10:   2444.630 s ( 61.0% of total)   677,715,621 paths     3.607 us/path
# Watts-Strogatz (n=1000, d=6, p=0.2): bsdfs 6174.825s, bcdfs 4004.043s, ratio(bsdfs/bcdfs) = 1.542
