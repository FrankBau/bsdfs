"""
python translation of pseudo-code from

  title        = {Efficient Hop-constrained s-t Simple Path Enumeration},
  volume       = {30},
  year         = {2021},
  issn         = {0949-877X},
  url          = {https://doi.org/10.1007/s00778-021-00674-5},
  doi          = {10.1007/s00778-021-00674-5},
  pages        = {799--823},
  number       = {5},
  journaltitle = {The {VLDB} Journal},
  author       = {Peng, You and Lin, Xuemin and Zhang, Ying and Zhang, Wenjie and Qin, Lu and Zhou, Jingren},
  date         = {2021-09-01}
"""
from collections import defaultdict


def bcdfs(G, s, t, k):
    """original control-flow, NO completeness"""
    S = []
    bar = {v: 0 for v in G.nodes}
    unstacking = {}


    def length(S):
        return len(S) - 1

    def UpdateBarrier(u, l):
        if bar[u] > l:
            bar[u] = l
            for v in G.predecessors(u):
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        F = k + 1
        S.append(u)
        if u == t:
            yield S.copy()
            S.pop()
            unstacking.clear()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                if v not in S:
                    if length(S) + bar[v] + 1 <= k:
                        f = yield from search(v)
                        if f != k + 1:
                            F = min(F, f + 1)

        if F == k + 1:
            bar[u] = k - length(S) + 1
            if u in unstacking:
                if length(S) > unstacking[u]:               # non-decreasing → violates S2 < S1
                    print(f"{G.edges=} {s=} {t=} {k=}")     # (or record the witness)
                unstacking[u] = length(S)                   # update to current, for the next comparison
            else:
                unstacking[u] = length(S)
        else:
            # bar[u] = k + 1
            UpdateBarrier(u, F)
        S.pop()
        return F

    yield from search(s)


import networkx as nx
import random
from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice


def worker_er(args):
    n, run = args
    random.seed(42 + run)
    p = random.uniform(0, 1)
    k = random.randint(1, n)
    
    G = nx.gnp_random_graph(n, p, directed=True)
    s, t = random.sample(range(n), 2)

    # we cutoff when a number of paths was generated
    paths1 = list(islice(bcdfs(G, s, t, k), 10_000))
    return len(paths1)


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs, processes=None):
    print(n)
    sum_paths = 0

    if processes == 0:
        for run in range(runs):
            sum_paths += worker_er((n, run))
        return sum_paths

    with Pool(processes) as pool:
        for result in pool.imap_unordered(worker_er, task_er(n, runs), chunksize=200):
            sum_paths += result
        return sum_paths


import itertools
import networkx as nx

def all_digraphs_by_edge_count(n, m_min=0, m_max=None):
    """
    Yield every directed graph on nodes 0..n-1 (no loops, no parallel edges),
    grouped by increasing edge count m. For each m in [m_min, m_max], yields
    all C(N, m) graphs where N = n*(n-1) is the number of possible edges.
    """
    nodes = list(range(n))
    possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
    N = len(possible_edges)                       # n*(n-1)
    if m_max is None:
        m_max = N
    for m in range(m_min, m_max + 1):
        for edge_subset in itertools.combinations(possible_edges, m):
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edge_subset)
            yield m, G


def search_counterexamples(n=6, k=6):
    s = 0
    for m, G in all_digraphs_by_edge_count(n,  m_min=n):
        for t in range(n):
            if s == t:
                continue
            paths = list(bcdfs(G, s, t, k))

def main():
    
    # X: counter-example to completeness 
    X = nx.parse_adjlist(
        ["A B C", "B C D E", "C B D", "D B", "E"], create_using=nx.DiGraph
    )
    s = 'A'
    t = 'E'
    k = 4
    pathsX = list(bcdfs(X, s, t, k))
    assert pathsX == [
        ["A", "B", "E"],
        ["A", "C", "B", "E"],
    ]  # missing ['A', 'C', 'D', 'B', 'E']

    # Z: counter-example to claimed monotonicity (||S2|| > ||S1||)
    Z = nx.DiGraph()
    # Z.add_edges_from([(0, 3), (1, 3), (1, 4), (1, 5), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 3), (5, 1)])
    Z.add_edges_from([('A', 'D'), ('B', 'D'), ('B', 'E'), ('B', 'F'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('E', 'A'), ('E', 'B'), ('E', 'D'), ('F', 'B')])
    s = 'E'
    t = 'C'
    k = 6
    pathsZ = list(bcdfs(Z, s, t, k))
    print(pathsZ)
    nx.nx_pydot.write_dot(Z, "Z.dot")
    # node F unstacked twice in same interval: ||S1||==2 < ||S2||==4
    
    
    # search_counterexamples()

    for n in range(5, 15):
        validate_er(n, 10_000_000)

    runs = 1_000_000
    random.seed(42)
    for n in range(6, 10):
        for k in range(n - 1, n + 2):
            for run in range(runs):
                p = random.uniform(0, 1)
                G = nx.gnp_random_graph(n, p, directed=True)
                s, t = random.sample(range(n), 2)
                bcdfs(G, s, t, k)

if __name__ == "__main__":
    main()
