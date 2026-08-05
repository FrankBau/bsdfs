"""Delay-bound measurements for the paper -- no command-line parameters:

    python delay_bound_experiments.py

Re-runs the samplers of Tables 1 and 2 (missed_paths_experiments.py:
Erdos-Renyi with n = 6..30, p in [2/(n-1), 5/(n-1)]; directed
symmetrizations of Watts-Strogatz with n = 1000, d = 6, p = 0.2; 100 000
trials per value of k, k = 3..10, identical seeds and s/t selection) and
measures, per trial, in units of (k+1)(n+m).

Intervals are indexed tau = 0..T with T = p_out, so a run has p_out + 1
intervals.  The proven bound depends on which class the interval falls in,
because the boundary intervals consist of a single phase:

    tau = 0     initial interval, ascent 1.  One phase, so the entry
                account is k(n+m) rather than 2k(n+m); the spine account
                is empty (lem:descent-stack(1)); the drop account is empty
                since an ascent contains no cascade (lem:phase-a-purity)
                -- before the first output no call has returned fruitfully.
                    k(n+m) + (k+1) <= (k+1)(n+m)        bound 1
                (For T = 0 the whole execution is interval 0 and pays no
                output either, so the same bound applies a fortiori.)

    1<=tau<T    interior interval, descent tau followed by ascent tau+1.
                All four accounts of thm:worst-case-delay are live:
                    2k(n+m) + (k+m) + k(n+m) + (k+1) <= 3(k+1)(n+m)
                                                        bound 3

    tau = T     terminal interval, descent T.  One phase; the spine
                account is non-empty and the descent carries its dequeues,
                but no output is produced:
                    k(n+m) + (k+m) + k(n+m) <= 2(k+1)(n+m)
                                                        bound 2

The amortized statistic steps(o_p) / (p (k+1)(n+m)) ranges over the
events o_2, ..., o_T, o_{T+1}, the last being the termination
(thm:amortized-delay).  It needs no separate entry for p = 1: steps(o_1)
is exactly the delay of interval 0, so that case is wc0 and is bounded
by 1 rather than by the 2 of the theorem; likewise a run with no output
has its single event covered by wc0.

Reported per k, and as a maximum over all k, per graph family:

    wc0   max over trials of interval 0's delay / unit           [bound 1]
    wcIn  max over trials and interior intervals / unit          [bound 3]
    wcT   max over trials of interval T's delay / unit           [bound 2]
    amP   max over trials and p >= 2 of steps(o_p) / (p unit)    [bound 2]

Each is accompanied by the run index attaining it and by that run's number
of intervals, so the extremal instance can be replayed with replay().
NB: a run index identifies an instance only relative to the seed offsets
and sampler code in experiments_base.

Steps follow the paper's cost model (sec:work-attribution):
1 + |suc(v)| per search call,
1 + |pre(q)| per cascade dequeue,
1 per node of the output path (<= k+1 per output).

Dependencies: networkx, tqdm (same as missed_paths_experiments.py).
"""

from multiprocessing import Pool
import signal

from tqdm import tqdm

import experiments_base as base
from trace_eval import measure_delays_traced

CHUNK = 200

# (key, column label, proven bound) -- order fixes the column order
STATS = (
    ("wc0",  "wc0",  1),
    ("wcIn", "wcIn", 3),
    ("wcT",  "wcT",  2),
    ("amP",  "amP",  2),
)


# ------------------------------------------------------------
# Workers -- instance construction verbatim, measurement swapped in
# ------------------------------------------------------------


def worker_er(args):
    k, run = args
    G, s, t = base.make_erdos_renyi(run)
    return measure_delays_traced(G, s, t, k) + (run,)


def worker_ws(args):
    k, run = args
    G, s, t = base.make_watts_strogatz(run)
    return measure_delays_traced(G, s, t, k) + (run,)


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Best:
    """Running argmax of one statistic, remembering where it was attained."""

    __slots__ = ("val", "k", "run", "n_int")

    def __init__(self):
        self.val, self.k, self.run, self.n_int = 0.0, None, None, None

    def offer(self, val, k, run, n_int):
        if val > self.val:
            self.val, self.k, self.run, self.n_int = val, k, run, n_int

    def merge(self, other):
        if other.val > self.val:
            self.val, self.k = other.val, other.k
            self.run, self.n_int = other.run, other.n_int


def print_header():
    head = f"{'n':>6} {'d':>4} {'p':>6} {'k':>4} {'outputs':>14}"
    for _, label, bound in STATS:
        head += f" {label + '[' + str(bound) + ']':>9} {'@run':>8}"
    print(head)


def run_family(name, worker, label, runs, processes=None):
    print(f"\n=== {name} ===")
    print_header()
    fam = {key: Best() for key, _, _ in STATS}
    fam_out = 0
    for k in base.K_VALUES:
        cur = {key: Best() for key, _, _ in STATS}
        out_k = 0
        jobs = ((k, run) for run in range(runs))
        with Pool(processes=processes, initializer=ignore_sigint) as pool:
            for res in tqdm(pool.imap_unordered(worker, jobs, chunksize=CHUNK),
                            total=runs, desc=f"{name} k={k}", leave=False):
                p_out, n_int, run = res[0], res[-2], res[-1]
                out_k += p_out
                for (key, _, _), val in zip(STATS, res[1:5]):
                    cur[key].offer(val, k, run, n_int)
        ln, ld, lp = label
        row = f"{ln:>6} {ld:>4} {lp:>6} {k:>4} {out_k:>14,}"
        for key, _, _ in STATS:
            row += f" {cur[key].val:9.3f} {str(cur[key].run):>8}"
        print(row, flush=True)
        fam_out += out_k
        for key, _, _ in STATS:
            fam[key].merge(cur[key])
    return fam, fam_out


# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------


def main(runs=base.RUNS, processes=None):
    results = {}

    results["Erdos-Renyi"] = run_family(
        f"Erdos-Renyi (n={min(base.ER_N_VALUES)}..{max(base.ER_N_VALUES)})",
        worker_er,
        (f"{min(base.ER_N_VALUES)}-{max(base.ER_N_VALUES)}", "-", "rand"),
        runs, processes)

    results["Watts-Strogatz"] = run_family(
        f"Watts-Strogatz (n={base.WS_N}, d={base.WS_D}, p={base.WS_P})",
        worker_ws,
        (base.WS_N, base.WS_D, base.WS_P),
        runs, processes)

    print("\n=== summary (max over all trials and all k, "
          "in units of (k+1)(n+m)) ===")
    print(f"{'family':<16} {'stat':>5} {'bound':>6} {'measured':>9} "
          f"{'ratio':>6} {'at k':>5} {'at run':>9} {'intervals':>10}")
    for fam_name, (fam, _) in results.items():
        for key, _, bound in STATS:
            r = fam[key]
            print(f"{fam_name:<16} {key:>5} {bound:>6} {r.val:9.3f} "
                  f"{r.val / bound:6.3f} {str(r.k):>5} {str(r.run):>9} "
                  f"{str(r.n_int):>10}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)

# 2026-08-04
#
# === Erdos-Renyi (n=6..30) ===
#      n    d      p    k        outputs    wc0[1]     @run   wcIn[3]     @run    wcT[2]     @run    amP[2]     @run
#   6-30    -   rand    3        402,585     0.480    23173     0.458    81995     0.557    38587     0.315    60115                                                                                                      
#   6-30    -   rand    4      1,237,409     0.464    73017     0.474    84867     0.600    38587     0.310    30400                                                                                                      
#   6-30    -   rand    5      3,602,698     0.467    90329     0.487    57468     0.637    30622     0.320    30622                                                                                                      
#   6-30    -   rand    6     10,211,016     0.471    90329     0.516    77446     0.663    30622     0.333    30622                                                                                                      
#   6-30    -   rand    7     28,598,970     0.472    90329     0.523    77446     0.658    30622     0.330    30622                                                                                                      
#   6-30    -   rand    8     79,057,076     0.472    90329     0.517    53572     0.645    30622     0.330    10772                                                                                                      
#   6-30    -   rand    9    214,472,635     0.471    90329     0.535    53572     0.623    30622     0.340    10772                                                                                                      
#   6-30    -   rand   10    567,775,657     0.471    90329     0.526    53572     0.607    30622     0.335    10772                                                                                                      

# === Watts-Strogatz (n=1000, d=6, p=0.2) ===
#      n    d      p    k        outputs    wc0[1]     @run   wcIn[3]     @run    wcT[2]     @run    amP[2]     @run
#   1000    6    0.2    3         18,754     0.065    87270     0.039    90447     0.053    39448     0.030    54399                                                                                                      
#   1000    6    0.2    4         90,541     0.148    53117     0.096    27458     0.131    18150     0.077    18150                                                                                                      
#   1000    6    0.2    5        431,964     0.264    53117     0.192    13495     0.217    41578     0.132    54010                                                                                                      
#   1000    6    0.2    6      2,051,942     0.333    14412     0.242    18837     0.259    86159     0.180     1166                                                                                                      
#   1000    6    0.2    7      9,718,746     0.365    83314     0.264    58526     0.230    57946     0.206    75649                                                                                                      
#   1000    6    0.2    8     45,964,706     0.376    83037     0.260    58851     0.057    89302     0.200    61187                                                                                                      
#   1000    6    0.2    9    217,177,858     0.340    51079     0.226    33099     0.045    89302     0.183    11796                                                                                                      
#   1000    6    0.2   10  1,025,602,442     0.322    71590     0.274    11393     0.031    56736     0.161    71590                                                                                                      

# === summary (max over all trials and all k, in units of (k+1)(n+m)) ===
# family            stat  bound  measured  ratio  at k    at run  intervals
# Erdos-Renyi        wc0      1     0.480  0.480     3     23173          2
# Erdos-Renyi       wcIn      3     0.535  0.178     9     53572         36
# Erdos-Renyi        wcT      2     0.663  0.331     6     30622          2
# Erdos-Renyi        amP      2     0.340  0.170     9     10772        514
# Watts-Strogatz     wc0      1     0.376  0.376     8     83037        228
# Watts-Strogatz    wcIn      3     0.274  0.091    10     11393       4985
# Watts-Strogatz     wcT      2     0.259  0.129     6     86159          2
# Watts-Strogatz     amP      2     0.206  0.103     7     75649          2
