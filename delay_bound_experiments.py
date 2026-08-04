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
from collections import deque
from multiprocessing import Pool
import signal

from tqdm import tqdm

import experiments_base as base

CHUNK = 200

# (key, column label, proven bound) -- order fixes the column order
STATS = (
    ("wc0",  "wc0",  1),
    ("wcIn", "wcIn", 3),
    ("wcT",  "wcT",  2),
    ("amP",  "amP",  2),
)


# ------------------------------------------------------------
# BS-DFS (tight scheme, bsdfs.py) with the paper's step counter
# ------------------------------------------------------------


def measure_delays(G, s, t, k):
    """Run bsdfs once; return (p_out, wc0, wcIn, wcT, amP, n_int).

    All five ratios are in units of (k+1)(n+m); see the module docstring
    for the interval classes and the bound applying to each.  A statistic
    with no instance in the run stays 0.0.
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    unit = (k + 1) * (n + m)
    b = {x: 0 for x in G.nodes}
    S = []
    steps = 0
    prev = 0
    p_out = 0
    wc0 = wc_in = wc_T = 0.0
    amP = 0.0

    def fruitful(v, sd):
        nonlocal steps
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            steps += 1
            for p in G.predecessors(q):
                steps += 1
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        nonlocal steps, prev, p_out, wc0, wc_in, amP
        S.append(v)
        h = len(S) - 1
        steps += 1
        sd = k + 1
        for w in G.successors(v):
            steps += 1
            if b[w] + h < k:
                if w == t:
                    steps += h + 2          # output work, 1 per path node
                    p_out += 1
                    # the interval just closed is tau = p_out - 1; it is
                    # interior whenever tau >= 1, since interval T closes
                    # at termination, not at an output
                    r = (steps - prev) / unit
                    prev = steps
                    if p_out == 1:
                        # steps(o_1) == interval 0's delay, so the p = 1
                        # case of the amortized statistic is exactly wc0
                        wc0 = r             # occurs exactly once
                    else:
                        if r > wc_in:
                            wc_in = r
                        q = steps / (p_out * unit)
                        if q > amP:
                            amP = q
                    sd = 1
                elif w not in S:
                    d = search(w)
                    sd = min(sd, d + 1)
        if sd <= k:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1
        S.pop()
        return sd

    search(s)
    r = (steps - prev) / unit               # segment after the last output
    if p_out == 0:
        wc0 = r                             # T = 0: the run is interval 0
    else:
        wc_T = r
    return p_out, wc0, wc_in, wc_T, amP, p_out + 1


def replay(family, k, run):
    """Rebuild and re-measure the instance identified in the output."""
    make = {"er": base.make_erdos_renyi, "ws": base.make_watts_strogatz}[family]
    G, s, t = make(run)
    res = measure_delays(G, s, t, k)
    p_out, n_int = res[0], res[-1]
    print(f"{family} run={run} k={k}: n={G.number_of_nodes()} "
          f"m={G.number_of_edges()} s={s} t={t} outputs={p_out} "
          f"intervals={n_int}")
    for (key, _, bound), val in zip(STATS, res[1:5]):
        print(f"    {key:>4} = {val:.4f}   [bound {bound}]")
    return G, s, t


# ------------------------------------------------------------
# Workers -- instance construction verbatim, measurement swapped in
# ------------------------------------------------------------


def worker_er(args):
    k, run = args
    G, s, t = base.make_erdos_renyi(run)
    return measure_delays(G, s, t, k) + (run,)


def worker_ws(args):
    k, run = args
    G, s, t = base.make_watts_strogatz(run)
    return measure_delays(G, s, t, k) + (run,)


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------


def ignore_sigint():
    """Pool initializer: let only the parent process see Ctrl+C.

    Ctrl+C sends SIGINT to every process in the foreground group, so each
    worker would otherwise print its own 'Process ForkPoolWorker-N:'
    traceback -- one per core.  A try/except in the worker body cannot
    suppress this: KeyboardInterrupt derives from BaseException, not from
    Exception.  Workers still die promptly, since the pool's context
    manager calls terminate() (SIGTERM, which is not ignored) on exit.
    """
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


def clique_trap(leading_st_edge = True):
    import networkx as nx
    k = 5
    for c in range(k+1, 1000, 100):
        s = 0
        t = c
        G = nx.DiGraph()
        if leading_st_edge:
            G.add_edge(s, t)
        else:
            G.add_node(t)
        for u in range(c):
            for v in range(c):
                if u != v: 
                    G.add_edge(u, v)
        n, m = G.number_of_nodes(), G.number_of_edges()
        unit = (k + 1) * (n + m)                    
        res = measure_delays(G, s, t, k)
        Nck = k*c - k*(k+1)/2 + 1
        # using float calcs in assert is dangerous, needs clean-up
        if leading_st_edge:
            assert res[1] == 4 / unit
            assert res[3] == (c*Nck+c*c-1) / unit
            print(res[3])
        else:
            assert res[1] == c*Nck / unit
            print(res[1] / (k/(k+1)))


if __name__ == "__main__":
    clique_trap(False)
    clique_trap(True)
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)
