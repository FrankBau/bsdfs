"""searching counter-examples for S*U*S* event structure in the loose scheme"""


from collections import deque
from collections import defaultdict

def bsdfs(G, s, t, k):
    """loose scheme, barriers are reset to 0"""
    b = {x: 0 for x in G.nodes}
    S = []
    usu = defaultdict(str)
    violators = set()
    
    def reset(v):
        b[v] = 0
        queue = deque([v])
        while queue:
            q = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] != 0:
                    b[p] = 0
                    if "us" in usu[p]:
                        violators.add(p)
                    usu[p] += 'u'
                    queue.append(p)

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    assert not violators, f"{s=} {t=} {k=} {G.edges=} {violators=}" 
                    yield S + [t]
                    sd = 1
                    usu.clear()
                    violators.clear()
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            reset(v)
        else:
            b[v] = k - h + 1
            usu[v] += 's'

        S.pop()
        return sd

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
    k = random.getrandbits(n).bit_count()  # binomial distribution
    k = max(k, 1)

    G = nx.gnp_random_graph(n, p, directed=True)
    s, t = random.sample(range(n), 2)

    # we cutoff when a number of paths was generated
    paths1 = list(islice(bsdfs(G, s, t, k), 500))
    assert paths1 == sorted(paths1)  # lexicographic order
    paths2 = list(islice(nx.all_simple_paths(G, s, t, k), 500))
    assert paths2 == sorted(paths2)  # lexicographic order
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


import random
import time


def performance(algo):
    random.seed(42)
    runs = 10_000
    for n in range(2, 10):
        for k in range(2, n + 1):
            tasks = []
            for run in range(runs):
                m = random.randint(n, n * (n - 1))
                G = nx.gnm_random_graph(n, m, directed=True)
                s, t = random.sample(range(n), 2)
                tasks.append((G, s, t, k))
            count = 0
            tick = time.perf_counter()
            for G, s, t, k in tasks:
                paths = list(algo(G, s, t, k))
                count += len(paths)
            tock = time.perf_counter()

            print(
                f"{n=:2} {k=:2} found {count:12,} paths in {runs:8} random graphs in {tock-tick:6.2f} s, {1_000_000 * (tock-tick)/count:6.2f} us/path"
            )


def smoke():
    # quick test for major error and general failure
    for n in range(2, 8):
        validate_er(n, 10_000, processes=0)


def main():
    seed = 42
    random.seed(seed)

    for n in range(5, 15):
        validate_er(n, 1_000_000)

    for n in range(15, 100, 5):
        validate_er(n, 10_000)


if __name__ == "__main__":
    s = 1
    t = 0
    k = 2 
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 1), (3, 2)])
    list(bsdfs(G, s, t, k))
    # asserts
    
    smoke()
    main()
    # performance(bsdfs)
