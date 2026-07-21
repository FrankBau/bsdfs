from collections import deque
import math


def bsdfs(G, s, t, k=math.inf):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd < k + 1:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)


import networkx as nx
import random
from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice


def worker_er(args, limit=1000):
    n, run = args
    random.seed(42 + run)
    m = int(n * math.exp(random.uniform(0, math.log(n-1))))
    G = nx.gnm_random_graph(n, m, directed=True)
    s, t = random.sample(list(G.nodes), 2)
    k = math.inf
    
    # we cutoff when a number of paths was generated
    paths1 = islice(bsdfs(G, s, t, k), limit)
    paths2 = islice(nx.all_simple_paths(G, s, t, k), limit)
    
    
    
    if paths1 != paths2:
        ps1 = set(map(tuple, paths1))
        ps2 = set(map(tuple, paths2))
        missing2 = ps1 - ps2
        missing1 = ps2 - ps1
        raise AssertionError(
            f"{s=} {t=} {k=} {G.edges=} {list(missing1)[:1]=}  {list(missing2)[:1]=}"
        )
    return len(paths1)


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs, processes=None):
    print(n)
    sum_paths = 0

    if processes == 0:
        for run in tqdm(range(runs), leave=False):
            sum_paths += worker_er((n, run))
        return sum_paths

    with Pool(processes) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_er, task_er(n, runs), chunksize=200),
            total=runs,
            leave=False,
        ):
            sum_paths += result
        return sum_paths


def main():
    seed = 42
    random.seed(seed)

    # quick test for major error and general failure
    for n in range(2, 8):
        validate_er(n, 100_000, processes=0)

    for n in range(8, 15):
        validate_er(n, 1_000_000)

    for n in range(15, 100, 5):
        validate_er(n, 10_000)


if __name__ == "__main__":
    
    # # Y: counter-example to claimed monotonicity (||S2|| > ||S1||)
    Y = nx.DiGraph()
    Y.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (2, 5), (3, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (5, 1)])
    s = 4
    t = 5
    k = 5
    pathsY = list(bsdfs(Y, s, t, k))
    print(pathsY)
        
    main()
