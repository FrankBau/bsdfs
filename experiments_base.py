"""
code base used in the experiments 
"""

from collections import deque
import networkx as nx
import random


RUNS = 100_000

K_MIN = 3
K_MAX = 10
K_VALUES = range(K_MIN, K_MAX+1)

# ER params
ER_NMIN = 6
ER_NMAX = 30
ER_N_VALUES = range(ER_NMIN, ER_NMAX+1)
ER_MIN_DEG = 2
ER_MAX_DEG = 5

# WS params
WS_N = 1000
WS_D = 6
WS_P = 0.2


def limited_descendants(G, s, k = float("inf")):
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


def find_st(G, rng):
    t = None
    while t is None:
        s = rng.choice(sorted(G.nodes))
        reachable = limited_descendants(G, s)
        if reachable:
            t = rng.choice(reachable)
    return s, t


def make_erdos_renyi(run):
    """randomized n and p"""
    rng = random.Random(42 + run)
    n = rng.choice(ER_N_VALUES)
    p = rng.uniform(ER_MIN_DEG, ER_MAX_DEG) / (n - 1)
    G = nx.gnp_random_graph(n, p, seed=rng, directed=True)
    s, t = find_st(G, rng)
    return G, s, t


def make_watts_strogatz(run):
    """fixed n, d, and p"""
    rng = random.Random(73 + run)
    n = WS_N
    d = WS_D
    p = WS_P
    H = nx.watts_strogatz_graph(n, d, p, seed=rng)
    G = nx.DiGraph(H)
    s, t = find_st(G, rng)
    return G, s, t


def smoke(algo, runs_per_n=2_000, n_values=range(2, 8), complete=True):
    """Self-check an enumerator against NetworkX on small random digraphs.

    algo(G, s, t, k) must yield the simple s-t paths of at most k edges.
    Three properties are checked on every instance:

      soundness       -- every emitted path is a genuine length-bounded
                         simple s-t path (never relaxed: a spurious path
                         is a failure for every scheme)
      lexicographic   -- outputs are emitted in lexicographic order
      completeness    -- no path is missed

    Completeness is asserted only when complete=True.  BC-DFS is known to
    be incomplete, so it is checked with complete=False, and the paths it
    misses are counted and reported instead of raising.
    """
    for n in n_values:
        found = missed = 0
        for run in range(runs_per_n):
            rng = random.Random(42 + run)
            p = rng.uniform(0, 1)
            k = max(rng.getrandbits(n).bit_count(), 1)  # binomial distribution
            G = nx.gnp_random_graph(n, p, seed=rng, directed=True)
            s, t = rng.sample(range(n), 2)

            got = list(map(list, algo(G, s, t, k)))
            expected = sorted(map(list, nx.all_simple_paths(G, s, t, k)))
            where = f"{s=} {t=} {k=} {G.edges=}"

            spurious = [q for q in got if q not in expected]
            assert not spurious, f"spurious path {spurious[0]}: {where}"
            assert got == sorted(got), f"not lexicographic: {where}"

            absent = [q for q in expected if q not in got]
            assert not (complete and absent), f"missed path {absent[0]}: {where}"

            found += len(got)
            missed += len(absent)

        note = "" if complete else f", {missed:,} paths missed"
        print(f"n={n:2}  {runs_per_n:,} random digraphs ok, {found:,} paths{note}")

