# coarse compare of bcdfs vs. bsdfs total runtime for the experiments
import math
import time
import gc
from tqdm import tqdm

import experiments_base as base
import bcdfs
import bsdfs


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
    run_experiment(base.make_erdos_renyi, base.RUNS, base.K_VALUES, f"Erdos-Renyi ({min(base.ER_N_VALUES)}..{max(base.ER_N_VALUES)})")
    run_experiment(base.make_watts_strogatz, base.RUNS, base.K_VALUES, f"Watts-Strogatz (n={base.WS_N}, d={base.WS_D}, p={base.WS_P})")

