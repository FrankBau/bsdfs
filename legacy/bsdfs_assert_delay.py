from collections import deque


class StepCounter:
    """Counts only real-algorithm elementary steps, matching thm:delay-bound's
    cost model: 1+|suc(v)| per search call, 1+|pre(q)| per cascade dequeue."""
    __slots__ = ('steps',)
    def __init__(self):
        self.steps = 0


def bsdfs(G, s, t, k, counter=None):
    """tight scheme (original BSDFS), instrumented for delay-bound checking"""
    if counter is None:
        counter = StepCounter()
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            counter.steps += 1                     # new: dequeue overhead
            for p in G.predecessors(q):
                counter.steps += 1                 # new: predecessor-scan step
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1
        counter.steps += 1                          # new: entry/exit + post-loop bookkeeping
        sd = k + 1
        for w in G.successors(v):
            counter.steps += 1                      # new: successor-scan step
            if b[w] + h < k:
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


from itertools import islice

def check_delay_bound(G, s, t, k):
    n, m = len(G.nodes), len(G.edges)
    general_bound = 3 * (k + 1) * (n + m)
    first_bound = (k + 1) * (n + m)

    counter = StepCounter()
    gen = bsdfs(G, s, t, k, counter)
    prev = counter.steps           # 0: generator hasn't executed anything yet
    delays = []

    for _ in islice(gen, 10_000):       # cap potential exponetial explosion
        delay = counter.steps - prev
        delays.append(delay)

        # conjecture Amortized Delay (sharpend)
        assert sum(delays) / len(delays) <= 2*k*m, f"conjecture Amortized Delay {s=} {t=} {k=} {G.edges=}"
        # assert sum(delays) / len(delays) < 2*k*m # breaks
        
        idx = len(delays)
        bound = first_bound if idx == 1 else general_bound
        assert delay <= bound, (
            f"interval {idx} delay {delay} exceeds "
            f"{'first-interval' if idx == 1 else 'general'} bound {bound}"
        )
        prev = counter.steps

    # terminal interval: last output (or start, if none) -> termination
    delay = counter.steps - prev
    delays.append(delay)
    idx = len(delays)
    bound = first_bound if idx == 1 else general_bound
    assert delay <= bound, f"terminal interval delay {delay} exceeds bound {bound}"

    return delays


import networkx as nx
import random
from multiprocessing import Pool
from tqdm import tqdm
import math


def worker_er(args):
    n, run = args
    random.seed(42 + run)
    p = random.uniform(0, 1)
    k = random.getrandbits(n).bit_count()  # binomial distribution
    k = max(k, 1)
    G = nx.gnp_random_graph(n, p, directed=True)
    s, t = random.sample(range(n), 2)

    delays = check_delay_bound(G, s, t, k)

    return len(delays)


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


def smoke():
    # quick sequential test for major error and general failure
    for n in range(2, 10):
        validate_er(n, 1_000, processes=0)


def main():
    for n in range(5, 15):
        validate_er(n, 1_000_000)

    for n in range(15, 100, 5):
        validate_er(n, 10_000)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)

    smoke()
    main()
