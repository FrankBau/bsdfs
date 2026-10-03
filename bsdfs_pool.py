"""
bsdfs validation on random graphs

multiprocessing variant for larger campaigns
"""

import math
import random
import signal
from collections import deque
from itertools import islice
from multiprocessing import Pool

import networkx as nx
from tqdm import tqdm

import bsdfs_trivial


def bsdfs(G, s, t, k):
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
            if b[w] < k - h:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)


# cap the number of outputs generated, mitigating combinatorial explosion
CAP = 10_000


def worker(args):
    """per instance graph generator and algo validator"""
    n, run, seed = args
    rng = random.Random(f"{seed}:{n}:{run}")
    m = int(n * math.exp(rng.uniform(0, math.log(n - 1))))
    k = rng.randint(1, n)
    G = nx.gnm_random_graph(n, m, directed=True, seed=rng)
    s, t = rng.sample(range(n), 2)

    outputs1 = list(islice(bsdfs(G, s, t, k), CAP))
    outputs2 = list(islice(bsdfs_trivial.bsdfs(G, s, t, k), CAP))
    # output orders are the same, determined by adj.list orders
    if outputs1 != outputs2:
        ps1 = set(map(tuple, outputs1))
        ps2 = set(map(tuple, outputs2))
        missing2 = ps1 - ps2
        missing1 = ps2 - ps1
        raise AssertionError(
            f"{s=} {t=} {k=} {G.edges=} {list(missing1)[:1]=}  {list(missing2)[:1]=}"
        )
    return len(outputs1)


def task(n, runs, seed):
    """generator for (many) worker's args"""
    for run in range(runs):
        yield (n, run, seed)


def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def validate_with_pool(pool, n, runs, seed):
    sum_outputs = 0
    try:
        for result in tqdm(
            pool.imap_unordered(worker, task(n, runs, seed)),
            total=runs,
            leave=False,
        ):
            sum_outputs += result
    except KeyboardInterrupt:
        pool.terminate()
        raise
    return sum_outputs


def validate_sequential(n, runs, seed=42):
    """validator running 'runs' random graphs with 'n' nodes"""
    print(f"{n=:4} {runs=:12,} {seed=:2}")
    sum_outputs = 0
    for run in tqdm(range(runs), leave=False):
        sum_outputs += worker((n, run, seed))
    return sum_outputs


def main():
    # debuggable smoke test
    for n in range(2, 7):
        validate_sequential(n, 10_000)

    # use multithreading in larger test campaign
    with Pool(initializer=ignore_sigint) as pool:
        for n in range(8, 20):
            runs = 10_000
            seed = 42
            print(f"{n=:4} {runs=:12,} {seed=:2}")
            validate_with_pool(pool, n, runs, seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
