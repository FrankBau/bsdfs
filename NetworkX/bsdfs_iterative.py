from collections import defaultdict, deque

# taken from https://github.com/networkx/networkx/blob/main/networkx/algorithms/cycles.py
class _NeighborhoodCache(dict):
    """Very lightweight graph wrapper which caches neighborhoods as list.

    This dict subclass uses the __missing__ functionality to query graphs for
    their neighborhoods, and store the result as a list.  This is used to avoid
    the performance penalty incurred by subgraph views.
    """

    def __init__(self, G):
        self.G = G

    def __missing__(self, v):
        Gv = self[v] = list(self.G[v])
        return Gv
    

def bsdfs(G, s, t, k):
    """Tight-scheme BS-DFS, fully iterative, parallel stacks."""
    b = defaultdict(int)          # barrier values; untouched nodes are 0
    S = [s]                       # current path
    on_stack = {s}                # set view of S for O(1) membership
    iters = [iter(G.successors(s))]
    sds = [k + 1]

    succ = _NeighborhoodCache(G)          # G[v]  -> successors
    pred = _NeighborhoodCache(G.pred)     # G.pred[v] -> predecessors

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in pred[q]:
                if p not in on_stack and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    while iters:
        h = len(S) - 1
        for w in iters[-1]:
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sds[-1] = 1
                elif w not in on_stack:
                    S.append(w)
                    on_stack.add(w)
                    iters.append(iter(succ[w]))
                    sds.append(k + 1)
                    break
        else:
            # successor list of S[-1] exhausted: finalize
            v = S.pop()
            on_stack.remove(v)
            iters.pop()
            sd = sds.pop()
            if sd <= k:
                fruitful(v, sd)
            else:
                b[v] = k - h + 1
            if sds and sd + 1 < sds[-1]:
                sds[-1] = sd + 1


import networkx as nx
import random
from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice
import math

def worker_er(args, limit=1000):
    n, run = args
    random.seed(42 + run)
    m = int(n * math.exp(random.uniform(0, math.log(n-1))))
    G = nx.gnm_random_graph(n, m, directed=True)
    s, t = random.sample(range(n), 2)
    k = random.randrange(1, n)
    
    # we cutoff when a number of paths was generated
    paths1 = list(islice(bsdfs(G, s, t, k), limit))
    paths2 = list(islice(nx.all_simple_paths(G, s, t, k), limit))
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
    