"""
code base used in the experiments 
"""

from collections import deque
import networkx as nx
import random


RUNS = 100_000

K_VALUES = range(3, 10+1)

# ER params
ER_N_VALUES = range(6, 30+1)
ER_MIN_DEG =2
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

