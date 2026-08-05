"""Cost of completeness -- no command-line parameters:

    python cost_of_completeness.py

Compares BC-DFS (Peng et al. 2021) against its kick-started variant, the
one-line completeness fix `bar[u] <- k+1` before the root UpdateBarrier
call.  Both are measured in the paper's cost model (sec:work-attribution),
decomposed into the three accounts:

    scan    -- 1 + #scanned successors, per search call
    cascade -- 1 + |pre(q)|, per firing UpdateBarrier
    output  -- |path| (<= k+1), per output

Per trial we record, for each account, the kick-started count and the
*difference* to plain BC-DFS (kick minus plain), i.e. the extra work the
fix costs on that instance.  Reported per k: mean, stdev, min, median and
max of both series, plus the ratio of sums (kick / plain) -- the ratio of
means, not the mean of ratios, since per-trial ratios are undefined when
plain BC-DFS does no cascade work at all.

Samplers, seeds and s/t selection are those of experiments_base, hence
identical to missed_paths_experiments.py and delay_bound_experiments.py.

The paper's claim that kick-start and BS-DFS produce identical output
sequences is checked by verify() below, which compares the two output
streams by rolling hash on a reduced number of trials; it is not part of
the main campaign, since running BS-DFS as well costs another ~50%.

Dependencies: networkx, tqdm.
"""

import statistics
import signal
from multiprocessing import Pool

from tqdm import tqdm

import experiments_base as base
from bsdfs_traced import bsdfs_traced
from bcdfs_traced import bcdfs_traced

CHUNK = 200
VERIFY_RUNS = 2000        # trials per k for the BS-DFS identity check

ACCOUNTS = ("scan", "cascade", "output")


# ------------------------------------------------------------
# Streaming account measurement -- no trace is stored
# ------------------------------------------------------------


class Accounts:
    """Emit-target accumulating the three accounts of one run.

    `bc=True` applies BC-DFS's scan rule: calls at t and calls at depth k
    skip the successor loop.  BS-DFS scans successors in every call.
    """

    __slots__ = ("G", "t", "k", "bc", "scan", "cascade", "output", "ohash", "p_out")

    def __init__(self, G, t, k, bc):
        self.G, self.t, self.k, self.bc = G, t, k, bc
        self.scan = self.cascade = self.output = 0
        self.ohash = 0
        self.p_out = 0

    def __call__(self, ev):
        kind = ev[0]
        if kind == "enter":
            v, h = ev[2], ev[3]
            if self.bc and (v == self.t or h == self.k):
                self.scan += 1
            else:
                self.scan += 1 + self.G.out_degree(v)
        elif kind == "dequeue":
            self.cascade += 1 + self.G.in_degree(ev[2])
        elif kind == "output":
            self.output += len(ev[2])
            self.p_out += 1
            self.ohash = hash((self.ohash, ev[2]))

    def triple(self):
        return self.scan, self.cascade, self.output


def measure(G, s, t, k, algo, bc):
    a = Accounts(G, t, k, bc)
    total = algo(G, s, t, k, a)
    assert a.scan + a.cascade + a.output == total, "cost model mismatch"
    return a


# ------------------------------------------------------------
# Workers -- return flat tuples of ints only
# ------------------------------------------------------------


def _sample(which, run):
    return (base.make_erdos_renyi if which == "er" else base.make_watts_strogatz)(run)


def worker(args):
    """(kick_scan, kick_cascade, kick_output, d_scan, d_cascade, d_output)."""
    which, k, run = args
    G, s, t = _sample(which, run)
    kick = measure(G, s, t, k, lambda *a: bcdfs_traced(*a, kickstart=True), True)
    plain = measure(G, s, t, k, lambda *a: bcdfs_traced(*a, kickstart=False), True)
    return kick.triple() + tuple(x - y for x, y in zip(kick.triple(), plain.triple()))


def worker_verify(args):
    """(outputs_equal, p_out) -- BS-DFS vs kick-started BC-DFS."""
    which, k, run = args
    G, s, t = _sample(which, run)
    kick = measure(G, s, t, k, lambda *a: bcdfs_traced(*a, kickstart=True), True)
    bs = measure(G, s, t, k, bsdfs_traced, False)
    return int(kick.ohash == bs.ohash and kick.p_out == bs.p_out), bs.p_out


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def summarize(series):
    """mean, std, min, median, max of one integer series."""
    return dict(
        mean=statistics.mean(series),
        std=statistics.stdev(series) if len(series) > 1 else 0.0,
        min=min(series),
        median=statistics.median(series),
        max=max(series),
        total=sum(series),
    )


def aggregate(results):
    """results: list of 6-tuples -> {account: (kick_summary, diff_summary)}."""
    if not results:
        return None
    cols = list(zip(*results))
    out = dict(n=len(results))
    for i, acc in enumerate(ACCOUNTS):
        out[acc] = (summarize(cols[i]), summarize(cols[i + 3]))
    return out


# ------------------------------------------------------------
# Reporting -- one table per account, rows over k
# ------------------------------------------------------------


def print_account_header(acc):
    print(f"\n--- {acc} account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---")
    print(f"{'k':>4} {'mean':>12} {'std':>12} {'min':>10} {'median':>10} {'max':>12}"
          f" | {'d mean':>10} {'d std':>10} {'d min':>8} {'d median':>9} {'d max':>10} {'kick/plain':>11}")


def print_account_row(k, a, acc):
    if a is None:
        print(f"{k:>4} -- no valid runs --")
        return
    c, d = a[acc]
    plain_total = c["total"] - d["total"]
    ratio = c["total"] / plain_total if plain_total else float("nan")
    print(f"{k:>4} {c['mean']:12.1f} {c['std']:12.1f} {c['min']:10,} {c['median']:10,.0f} {c['max']:12,}"
          f" | {d['mean']:10.1f} {d['std']:10.1f} {d['min']:8,} {d['median']:9,.0f} {d['max']:10,} {ratio:11.3f}")


def run_family(name, which, label, runs, processes=None):
    print(f"\n=== {name} ===")
    aggs = {}
    for k in base.K_VALUES:
        jobs = ((which, k, run) for run in range(runs))
        with Pool(processes=processes, initializer=ignore_sigint) as pool:
            results = list(tqdm(pool.imap_unordered(worker, jobs, chunksize=CHUNK),
                                total=runs, desc=f"{name} k={k}", leave=False))
        aggs[k] = aggregate(results)
    for acc in ACCOUNTS:
        print_account_header(acc)
        for k in base.K_VALUES:
            print_account_row(k, aggs[k], acc)
        print(flush=True)
    return aggs


def print_totals(name, aggs):
    print(f"\n--- {name}: totals, all accounts (ratio of sums) ---")
    print(f"{'k':>4} {'kick steps':>16} {'plain steps':>16} {'extra':>14} {'kick/plain':>11}")
    for k in base.K_VALUES:
        a = aggs[k]
        if a is None:
            continue
        kick = sum(a[acc][0]["total"] for acc in ACCOUNTS)
        diff = sum(a[acc][1]["total"] for acc in ACCOUNTS)
        plain = kick - diff
        ratio = kick / plain if plain else float("nan")
        print(f"{k:>4} {kick:16,} {plain:16,} {diff:14,} {ratio:11.3f}")


# ------------------------------------------------------------
# BS-DFS identity check (separate, reduced sweep)
# ------------------------------------------------------------


def verify(runs=VERIFY_RUNS, processes=None):
    print(f"\n=== output identity: kick-started BC-DFS vs BS-DFS "
          f"({runs:,} trials per k and family) ===")
    print(f"{'family':<16} {'k':>4} {'trials':>10} {'equal':>10} {'outputs':>16}")
    for which, fam in (("er", "Erdos-Renyi"), ("ws", "Watts-Strogatz")):
        for k in base.K_VALUES:
            jobs = ((which, k, run) for run in range(runs))
            with Pool(processes=processes, initializer=ignore_sigint) as pool:
                res = list(tqdm(pool.imap_unordered(worker_verify, jobs, chunksize=CHUNK),
                                total=runs, desc=f"{fam} k={k}", leave=False))
            eq = sum(r[0] for r in res)
            outs = sum(r[1] for r in res)
            flag = "" if eq == len(res) else "   *** MISMATCH ***"
            print(f"{fam:<16} {k:>4} {len(res):10,} {eq:10,} {outs:16,}{flag}", flush=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main(runs=base.RUNS, processes=None):
    er = run_family(f"Erdos-Renyi (n={min(base.ER_N_VALUES)}..{max(base.ER_N_VALUES)})", "er", None, runs, processes)
    print_totals("Erdos-Renyi", er)

    ws = run_family(f"Watts-Strogatz (n={base.WS_N}, d={base.WS_D}, p={base.WS_P})", "ws", None, runs, processes)
    print_totals("Watts-Strogatz", ws)


if __name__ == "__main__":
    try:
        verify()
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)


# 2026-08-04
#
# === output identity: kick-started BC-DFS vs BS-DFS (2,000 trials per k and family) ===
# family              k     trials      equal          outputs
# Erdos-Renyi         3      2,000      2,000            8,279                                                                                                                                                            
# Erdos-Renyi         4      2,000      2,000           24,879                                                                                                                                                            
# Erdos-Renyi         5      2,000      2,000           71,057                                                                                                                                                            
# Erdos-Renyi         6      2,000      2,000          197,035                                                                                                                                                            
# Erdos-Renyi         7      2,000      2,000          540,650                                                                                                                                                            
# Erdos-Renyi         8      2,000      2,000        1,465,256                                                                                                                                                            
# Erdos-Renyi         9      2,000      2,000        3,891,867                                                                                                                                                            
# Erdos-Renyi        10      2,000      2,000       10,040,126                                                                                                                                                            
# Watts-Strogatz      3      2,000      2,000              400                                                                                                                                                            
# Watts-Strogatz      4      2,000      2,000            1,812                                                                                                                                                            
# Watts-Strogatz      5      2,000      2,000            8,444                                                                                                                                                            
# Watts-Strogatz      6      2,000      2,000           40,503                                                                                                                                                            
# Watts-Strogatz      7      2,000      2,000          194,133                                                                                                                                                            
# Watts-Strogatz      8      2,000      2,000          924,659                                                                                                                                                            
# Watts-Strogatz      9      2,000      2,000        4,374,363                                                                                                                                                            
# Watts-Strogatz     10      2,000      2,000       20,616,022                                                                                                                                                            

# === Erdos-Renyi (n=6..30) ===
                                                                                                                                                                                                                        
# --- scan account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3         70.4         45.7          3         62          368 |       -2.3        3.5      -36        -1          0       0.968
#    4        163.8        138.7          3        125        1,487 |       -6.9        9.1      -67        -5        116       0.960
#    5        412.7        485.3          3        238        7,283 |       -2.7       29.5      -85        -5        605       0.994
#    6       1132.3       1748.2          3        443       31,948 |       45.7      130.6     -100         0      3,110       1.042
#    7       3218.2       6193.1          3        800      132,240 |      237.7      532.8     -104        27     14,838       1.080
#    8       9149.9      21349.6          3      1,349      520,488 |      866.0     1999.4     -108       110     66,933       1.105
#    9      25576.9      71335.3          3      2,119    1,979,593 |     2784.1     7165.6      -96       262    266,898       1.122
#   10      69688.1     230409.8          3      3,061    8,049,508 |     8385.1    24669.1     -106       490    977,065       1.137


# --- cascade account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3         32.2         30.5          0         23          285 |       32.2       30.5        0        23        285         nan
#    4        109.0        118.8          0         67        1,438 |      108.8      118.7        0        67      1,438     628.427
#    5        341.2        450.1          0        171        7,597 |      338.4      447.6        0       169      7,597     120.672
#    6       1027.8       1657.3          0        371       33,889 |     1010.9     1640.3        0       359     33,889      60.832
#    7       3026.5       5916.4          0        718      140,801 |     2959.8     5837.0        0       686    140,740      45.385
#    8       8724.8      20462.4          0      1,259      554,255 |     8501.9    20140.6        0     1,188    553,956      39.146
#    9      24547.6      68490.5          0      2,026    2,035,062 |    23857.9    67274.3        0     1,888  2,032,754      35.592
#   10      67149.1     221444.7          0      2,954    6,987,999 |    65111.6   217067.7        0     2,714  6,976,214      32.956


# --- output account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3         14.6         15.2          0         10          152 |        0.0        0.0        0         0          0       1.000
#    4         56.4         65.5          0         33          788 |        0.3        1.8        0         0         40       1.006
#    5        198.3        271.2          0         94        4,370 |        5.1       12.8        0         0        231       1.026
#    6        660.9       1094.7          0        226       22,808 |       32.8       65.5        0         7      1,237       1.052
#    7       2131.9       4270.4          0        467      107,656 |      148.2      293.4        0        27      6,358       1.075
#    8       6673.1      16037.3          0        864      474,685 |      561.0     1211.2        0        81     31,729       1.092
#    9      20214.7      57864.9          0      1,426    1,923,405 |     1932.1     4745.1        0       184    137,681       1.106
#   10      59078.0     200232.1          0      2,146    7,203,801 |     6245.7    17721.2        0       332    540,226       1.118


# --- Erdos-Renyi: totals, all accounts (ratio of sums) ---
#    k       kick steps      plain steps          extra  kick/plain
#    3       11,728,619        8,738,990      2,989,629       1.342
#    4       32,914,824       22,686,161     10,228,663       1.451
#    5       95,223,020       61,143,110     34,079,910       1.557
#    6      282,092,823      173,153,522    108,939,301       1.629
#    7      837,664,092      503,086,666    334,577,426       1.665
#    8    2,454,782,741    1,461,890,308    992,892,433       1.679
#    9    7,033,920,310    4,176,512,204  2,857,408,106       1.684
#   10   19,591,523,215   11,617,288,409  7,974,234,806       1.686

# === Watts-Strogatz (n=1000, d=6, p=0.2) ===
                                                                                                                                                                                                                        
# --- scan account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3        276.0         66.0         87        270          687 |       -0.2        1.1      -19         0          0       0.999
#    4        970.4        254.9        226        952        2,371 |       -0.9        4.9      -53         0        101       0.999
#    5       3003.2        742.0        672      2,976        6,602 |       -1.5       22.4      -98         0        537       0.999
#    6       7553.6       1490.7      1,988      7,579       14,082 |       16.4      103.7     -230        -6      1,850       1.002
#    7      15042.4       2397.5      5,296     14,988       33,456 |      228.5      462.3     -275        12      5,460       1.015
#    8      28521.7       6074.2     11,153     27,662       91,717 |     1998.6     2080.0     -371     1,375     22,128       1.075
#    9      70264.6      21908.1     18,453     67,150      279,102 |    14340.0     9635.2     -301    12,485     91,723       1.256
#   10     249035.1      89689.9     34,043    236,942      881,806 |    86015.8    44621.7      177    79,207    388,542       1.528


# --- cascade account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3          2.7         10.8          0          0          140 |        2.7       10.8        0         0        140         nan
#    4         16.0         39.1          0          0          479 |       15.9       38.8        0         0        479     248.142
#    5         84.5        134.3          0         38        1,589 |       83.4      130.5        0        38      1,576      77.649
#    6        418.7        452.5          0        274        5,794 |      410.2      433.2        0       274      5,646      49.044
#    7       2012.5       1545.5          0      1,635       18,222 |     1954.1     1462.8        0     1,612     17,362      34.423
#    8       9562.9       5509.8         55      8,583       68,001 |     9163.5     5156.3       55     8,278     65,738      23.944
#    9      45241.6      21044.8        985     42,191      236,863 |    42711.5    19561.4      985    39,887    227,693      17.881
#   10     213727.7      86670.2     10,505    202,070      807,193 |   200366.6    80666.8   10,406   189,522    767,496      15.996


# --- output account: kick-started BC-DFS, and the extra cost over plain BC-DFS ---
#    k         mean          std        min     median          max |     d mean      d std    d min  d median      d max  kick/plain
#    3          0.7          3.4          0          0           57 |        0.0        0.0        0         0          0       1.000
#    4          4.3         13.6          0          0          232 |        0.1        1.0        0         0         30       1.019
#    5         24.8         51.5          0          6          891 |        1.1        6.1        0         0        150       1.047
#    6        138.2        193.9          0         76        3,371 |       10.4       30.7        0         0        543       1.082
#    7        751.5        732.2          0        551       11,323 |       90.5      150.1        0        24      2,332       1.137
#    8       4013.7       2801.0          9      3,411       50,749 |      760.0      752.1        0       540      8,876       1.234
#    9      21135.0      11107.7        389     19,240      193,939 |     5798.2     3838.6        0     5,076     35,324       1.378
#   10     110061.7      46755.7      5,457    103,390      689,845 |    37876.3    19303.4       99    35,047    182,367       1.525


# --- Watts-Strogatz: totals, all accounts (ratio of sums) ---
#    k       kick steps      plain steps          extra  kick/plain
#    3       27,945,674       27,687,005        258,669       1.009
#    4       99,067,519       97,561,115      1,506,404       1.015
#    5      311,244,509      302,951,597      8,292,912       1.027
#    6      811,052,921      767,354,643     43,698,278       1.057
#    7    1,780,649,626    1,553,342,205    227,307,421       1.146
#    8    4,209,825,498    3,017,616,153  1,192,209,345       1.395
#    9   13,664,118,024    7,379,154,793  6,284,963,231       1.852
#   10   57,282,445,158   24,856,571,976 32,425,873,182       2.305
  