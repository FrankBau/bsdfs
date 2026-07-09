from collections import deque

def dist(G, S, x, y):
    q = deque()
    q.append((x, 0))
    seen = ({x} | set(S)) - {y}

    while q:
        u, d = q.popleft()
        for v in G.successors(u):
            if v in seen:
                continue
            if v == y:
                return d + 1
            seen.add(v)
            q.append((v, d + 1))
    return float('inf')


def check_edge_consistency(G, S, b):
    for x, y in G.edges:
        if x not in S and y not in S:
            assert (
                b[x] <= b[y] + 1
            ), f"edge {x}→{y}: {b[x]=} > {b[y]+1=}"


def bsdfs(G, s, t, k):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []
    seen_search_paths = set()
    seen_outputs = set()

    def fruitful(v, sd):
        b[v] = sd
        enqueued = set()
        queue = deque([(v, sd)])
        enqueued.add(v)
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    old = b[p]
                    b[p] = d + 1
                    assert b[p] < old, "Cascade Strictly Decreases (Obs. 9)"
                    assert b[p] >= dist(G, S[:-1], p, t), "Fruitful Distance Lower Bound" 
                    queue.append((p, d + 1))
                    enqueued.add(p)
        for u in G.nodes:
            if u not in S:
                if u in enqueued:
                    assert b[u] == sd + dist(G, S, u, v), "Cascade Distance (enqueued)"
                else:
                    assert b[u] <= sd + dist(G, S, u, v), "Cascade Distance (other)"

    def search(v):
        assert v not in S, "Search Path Stays Simple (Obs. 1)"
        key = tuple(S + [v])
        assert key not in seen_search_paths, "Search Path Occurs At Most Once (Obs. 3)"
        seen_search_paths.add(key)
        bar_entry = b.copy()
        check_edge_consistency(G, S, b) # S does *not* contain v here
        assert b[s] == 0, "Source Barrier Stays 0 (Obs. 7)"
        assert b[t] == 0, "Target Barrier Stays 0 (Obs. 8)"

        S.append(v)
        h = len(S) - 1
        assert b[v] <= k - h, "Parent Pruning Guard"
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    out = tuple(S + [t])
                    assert out not in seen_outputs, "No Output Produced Twice (Obs. 4)"
                    seen_outputs.add(out)
                    assert h + 1 <= k, "Output Length Bound"
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)
            else:
                if w == t:
                    assert h == k, "Target Only Pruned at Max Depth"
                elif w not in S:
                    assert dist(G, S, w, t) + h >= k, "Pruning-is-Permissive"

        assert b[v] == bar_entry[v], "Barrier of v Untouched Until Final Write (Obs. 6)"
        assert all(b[x] >= bar_entry[x] for x in G.nodes), "all(bar[x] >= bar_before[x]) (before update)"
        check_edge_consistency(G, S, b) # S *does* contain v here
        assert b[v] <= sd, "bar[v] <= sd (Barrier Invariant)"
        assert sd >= 0, "Return Value Non-Negative (Obs. 5)"

        if sd <= k:
            assert sd == dist(G, S, v, t), "Strict Barrier Invariant"
            fruitful(v, sd)
            assert b[v] >= bar_entry[v], "Fruitful Non-Decreasing"
        else:
            b[v] = k - h + 1
            assert b[v] > bar_entry[v], "Fruitless Increasing"
            
        assert all(b[x] >= bar_entry[x] for x in G.nodes), "Per-Call Global Monotonicity"

        S.pop()
        check_edge_consistency(G, S, b) # S does *not* contain v here
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
    for n in range(2, 10):
        validate_er(n, 1_000, processes=0)


def main():
    seed = 42
    random.seed(seed)

    for n in range(5, 15):
        validate_er(n, 100_000)

    for n in range(15, 100, 5):
        validate_er(n, 10_000)


if __name__ == "__main__":
    smoke()
    main()
    # performance(bsdfs)
