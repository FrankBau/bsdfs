"""
    implementation of simple_search from the preprint
    
  title         = {Finding All Bounded-Length Simple Cycles in a Directed Graph -- Revisited},
  author        = {Frank Bauernöppel and Jörg-Rüdiger Sack},
  year          = {2026},
  eprint        = {2512.08392},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DS},
  url           = {https://arxiv.org/abs/2512.08392v3}
  
"""

from collections import deque
from math import inf
import sys
sys.setrecursionlimit(10_000)

def dist(G, F, x, y):
    r"""shortest distance of xy-path P in G, P \cap F \subset {x, y}"""
    F = set(F) - {x, y}
    d = {x: 0}
    queue = deque([x])
    while queue:
        u = queue.popleft()
        if u == y:
            return d[u]
        for v in G.successors(u):
            if v not in F and d.get(v, inf) > d[u] + 1:
                d[v] = d[u] + 1
                queue.append(v)
    return inf


def simple_search(G, s, k):
    """Enumerate all simple cycles in G of bounded length k containing node s."""

    def reach(blocked, successors, budget):
        reached = {s}
        queue = deque()
        queue.append((s, 0))
        while queue:
            (u, d) = queue.popleft()
            if d >= budget:
                break
            for v in G.predecessors(u):
                if v not in blocked and v not in reached:
                    reached.add(v)
                    queue.append((v, d + 1))

        # assert reached == {s} | {w for w in set(G.nodes)-set(blocked) if dist(G, set(blocked), w, s) <= budget}, "Observation 1"
        fruitful = [w for w in successors if w in reached]     # this keeps the internal order of the successors
        return fruitful

    def search(path, v, budget):
        # assert budget > 0
        path.append(v)
        fruitful = reach(set(path), G.successors(v), budget - 1)
        for w in fruitful:
            if w == s:
                yield path[:]   # output cycle
            else:
                yield from search(path, w, budget - 1)
        path.pop()

    path = list()
    yield from search(path, s, k)


################## testbed #################

import networkx as nx
import random
from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice


def worker_er(args, limit=1000, check_output=True):
    n, run = args
    random.seed(42 + run)
    p = random.uniform(0, 1)
    G = nx.gnp_random_graph(n, p, directed=True)
    s = random.choice(range(n))
    k = random.choice(range(1,n))

    # we cutoff when a number of cycles was generated
    cycles1 = list(islice(simple_search(G, s, k), 1000))
    if check_output:
        # actually, there is no need to sort or setify the cycles, they are produced in the same order:
        cycles2 = (list(islice(nx.algorithms.cycles._bounded_cycle_search(G, [s], length_bound=k), 1000)))
        if cycles1 != cycles2:
            ps1 = set(map(tuple, cycles1))
            ps2 = set(map(tuple, cycles2))
            missing2 = ps1 - ps2
            missing1 = ps2 - ps1
            raise AssertionError(
            # print(
                f"{s=} {k=} {G.edges=} {list(missing1)[:1]=}  {list(missing2)[:1]=}"
            )
    return len(cycles1)


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs, processes=None):
    print(n)
    sum_cycles = 0
    if processes == 0:
        for run in tqdm(range(runs)):
            sum_cycles += worker_er((n, run))
    else:
        with Pool(processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_er, task_er(n, runs), chunksize=200),
                total=runs,
            ):
                sum_cycles += result
    return sum_cycles


def main():
    seed = 42
    random.seed(seed)

    # smoke test for major error and general failure
    for n in range(2, 8):
        validate_er(n, 10_000, processes=0)

    for n in range(5, 20):
        validate_er(n, 1_000_000)

    for n in range(25, 100):
        validate_er(n, 100_000)

if __name__ == "__main__":
    main()
