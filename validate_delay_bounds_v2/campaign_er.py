"""Interval-delay ratio campaign over multiple graph families.

Question under test: is the worst-case interval delay in fact bounded by
2(k+1)(n+m), i.e. does the proven constant 3 have slack in the *summed*
bound even though per-node event counts reach ~3k?

For every interval this campaign records
    ratio = delay / ((k+1)(n+m))
and reports, per family and size, the maximum ratio with its witness
instance -- the supremum is the informative statistic, not the absence of
violations at any particular threshold.  The proven bound (ratio <= 3, and
<= 1 for the initial interval) is asserted; everything else is measured.

Families (all directed):
    gnm        the paper's sampler: m ~ n * exp(U(0, ln(n-1))), k ~ U{1..n-1}
    gnp_dense  G(n,p) with p ~ U(0.5, 1)
    complete   K_n minus self-loops, k ~ U{2..n-1}
    layered    w x L layered DAG, all edges between adjacent layers,
               plus random back edges (cascade-rich, deep spines)
    comb       spine s -> u_1 -> ... -> u_L -> t with teeth: each u_i also
               reaches t through shared tail nodes of varied length
               (distance-diverse drop chains)
    tournament random tournament (exactly one arc per pair)
    regular    random d-regular-ish digraph, d ~ U{2..min(6,n-1)}

Usage:
    python ratio_campaign.py smoke
    python ratio_campaign.py run gnm,layered,comb 8,12,16,20 2000
    python ratio_campaign.py run gnm 10,20,30 100000 8     # 8 worker processes
(smoke is sequential; run uses all cores unless a process count is given)
"""
import math
import random
import sys
from itertools import islice

import networkx as nx

from bsdfs_traced import StepCounter, bsdfs


# ------------------------------------------------------------- families

def fam_gnm(n, rng):
    m = int(n * math.exp(rng.uniform(0, math.log(max(n - 1, 2)))))
    m = min(m, n * (n - 1))
    G = nx.gnm_random_graph(n, m, seed=rng.randrange(1 << 30), directed=True)
    k = rng.randrange(1, n)
    return G, k


def fam_gnp_dense(n, rng):
    p = rng.uniform(0.5, 1.0)
    G = nx.gnp_random_graph(n, p, seed=rng.randrange(1 << 30), directed=True)
    k = rng.randrange(1, n)
    return G, k


def fam_complete(n, rng):
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    k = rng.randrange(2, n) if n > 2 else 1
    return G, k


def fam_layered(n, rng):
    w = rng.randrange(2, max(3, n // 3))
    L = max(2, n // w)
    G = nx.DiGraph()
    layers = [[i * w + j for j in range(w)] for i in range(L)]
    nodes = [v for lay in layers for v in lay]
    G.add_nodes_from(nodes)
    for a, b in zip(layers, layers[1:]):
        for u in a:
            for v in b:
                G.add_edge(u, v)
    # random back edges to create cycles and revivals
    nback = rng.randrange(0, 2 * len(nodes))
    for _ in range(nback):
        u, v = rng.sample(nodes, 2)
        G.add_edge(u, v)
    k = rng.randrange(2, len(nodes))
    return G, k


def fam_comb(n, rng):
    """Spine with distance-diverse alternative tails: each spine node u_i has,
    besides the spine edge, a private path to t of length ~ (L - i) + delta_i,
    delta varied -- successive cascades then offer differing values to the
    shared approach nodes."""
    L = max(3, n // 3)
    G = nx.DiGraph()
    s, t = 0, 1
    spine = [2 + i for i in range(L)]
    G.add_edge(s, spine[0])
    for a, b in zip(spine, spine[1:]):
        G.add_edge(a, b)
    G.add_edge(spine[-1], t)
    nxt = 2 + L
    for i, u in enumerate(spine):
        ln = rng.randrange(1, 3 + (L - i))          # private tail length
        prev = u
        for _ in range(ln):
            G.add_node(nxt)
            G.add_edge(prev, nxt)
            prev = nxt
            nxt += 1
        G.add_edge(prev, t)
    # approach nodes reaching several spine nodes (drop-chain victims)
    for _ in range(rng.randrange(1, L)):
        x = nxt
        nxt += 1
        for u in rng.sample(spine, rng.randrange(2, min(len(spine), 5) + 1)):
            G.add_edge(x, u)
        G.add_edge(rng.choice(spine), x)            # make x reachable
    k = rng.randrange(3, G.number_of_nodes())
    return G, k


def fam_tournament(n, rng):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
    k = rng.randrange(1, n)
    return G, k


def fam_regular(n, rng):
    d = rng.randrange(2, min(6, n - 1) + 1)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in rng.sample([x for x in range(n) if x != u], d):
            G.add_edge(u, v)
    k = rng.randrange(1, n)
    return G, k


FAMILIES = {
    "gnm": fam_gnm, "gnp_dense": fam_gnp_dense, "complete": fam_complete,
    "layered": fam_layered, "comb": fam_comb, "tournament": fam_tournament,
    "regular": fam_regular,
}


# ------------------------------------------------------------- measure

def measure(G, s, t, k, limit=5000):
    """Yield (interval_index, delay) for one run; assert only the proven
    bounds."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if m == 0:
        return
    unit = (k + 1) * (n + m)
    counter = StepCounter()
    gen = bsdfs(G, s, t, k, counter=counter)
    prev, idx = 0, 0
    capped = False
    for _ in islice(gen, limit):
        delay = counter.steps - prev
        yield idx, delay, unit
        assert delay <= (unit if idx == 0 else 3 * unit), \
            f"proven bound violated: interval {idx} delay {delay}"
        prev = counter.steps
        idx += 1
    else:
        capped = idx >= limit
    if not capped:
        delay = counter.steps - prev
        yield idx, delay, unit
        assert delay <= (unit if idx == 0 else 3 * unit)


def worker(args):
    """One instance; returns per-run maxima so only small tuples cross the
    process boundary."""
    fam, n, r, seed0, limit = args
    rng = random.Random(seed0 + r)
    G, k = FAMILIES[fam](n, rng)
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return 0, (0.0, None), (0.0, None)
    s, t = rng.sample(nodes, 2)
    intervals = 0
    best = (0.0, None)
    first = (0.0, None)
    for idx, delay, unit in measure(G, s, t, k, limit):
        intervals += 1
        ratio = delay / unit
        if idx == 0:
            if ratio > first[0]:
                first = (ratio, (fam, n, r, k))
        elif ratio > best[0]:
            best = (ratio, (fam, n, r, k, idx))
    return intervals, best, first


def campaign(fams, sizes, runs, seed0=42, limit=5000, processes=None):
    """processes=0 runs sequentially (debugging); otherwise a Pool is used."""
    grand = (0.0, None)
    pool = None
    if processes != 0:
        from multiprocessing import Pool
        pool = Pool(processes)
    try:
        for fam in fams:
            for n in sizes:
                jobs = ((fam, n, r, seed0, limit) for r in range(runs))
                if pool is None:
                    results = map(worker, jobs)
                else:
                    results = pool.imap_unordered(worker, jobs, chunksize=32)
                best = (0.0, None)
                first_best = (0.0, None)
                intervals = 0
                for cnt, b, f in results:
                    intervals += cnt
                    if b[0] > best[0]:
                        best = b
                    if f[0] > first_best[0]:
                        first_best = f
                if best[0] > grand[0]:
                    grand = best
                print(f"{fam:>10} n~{n:<4} runs={runs:<6} "
                      f"intervals={intervals:>9,} "
                      f"max ratio={best[0]:.3f} at {best[1]}   "
                      f"(initial: {first_best[0]:.3f})", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"\ngrand max (non-initial): {grand[0]:.3f} at {grand[1]}")
    print("scale: proven worst-case = 3.000, proven amortized = 2.000, "
          "proven initial = 1.000")
    return grand


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        campaign(["gnm", "complete", "layered", "comb", "tournament"],
                 [8, 12, 16], 300, processes=0)
    else:
        fams = sys.argv[2].split(",")
        sizes = [int(x) for x in sys.argv[3].split(",")]
        runs = int(sys.argv[4])
        procs = int(sys.argv[5]) if len(sys.argv) > 5 else None
        campaign(fams, sizes, runs, processes=procs)
