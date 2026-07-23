"""BS-DFS drop-in replacement for networkx.algorithms.simple_paths._all_simple_edge_paths.

Same contract and same output ORDER as the stock generator:
  - `targets` arrives as a set (wrapper has already normalized node-vs-iterable),
  - `cutoff` may be None  =>  len(G) - 1,
  - DFS over EDGE paths: parallel edges are branched in the search itself,
    so multigraph combinations interleave exactly as in stock,
  - yields (u, v) tuples, or (u, v, key) for multigraphs,
  - trivial path (source in targets) is the empty edge path [].

Algorithm: BS-DFS in the target-set formulation (unmaterialized t*),
unit-shifted so all constants coincide with the single-target paper version:
  entry hit    v in targets  =>  yield, sd = 0
  guard        b[w] + h < k
  sentinel     k + 1
  fold         sd_parent = min(sd_parent, sd_child + 1)
  fruitless    b[v] <- k - h + 1
  fruitful     sd <= k  =>  Fruitful(v, sd), v still on the stack,
               unconditional root assignment b[v] <- sd (cascade kick-start)

Patch:
    import networkx.algorithms.simple_paths as simple_paths
    simple_paths._all_simple_edge_paths = bsdfs_all_simple_edge_paths
"""

from collections import defaultdict, deque
import networkx as nx


def bsdfs_all_simple_edge_paths(G, source, targets, cutoff):
    if cutoff is None:
        cutoff = len(G) - 1
        # this is correct, but the resulting O(k(n+m)) delay for BS-DFS is not optimal for k = n-1.
        # A more efficient algorithm is an adapted Johnson's circuit finding algo with O(n+m) delay,
        # using Boolean barriers and B-lists.
    k = cutoff
    if k < 0 or not targets:
        return

    get_edges = (
        (lambda v: G.edges(v, keys=True))
        if G.is_multigraph()
        else (lambda v: G.edges(v))
    )
    pred = G.pred if G.is_directed() else G.adj   # cascade direction

    b = defaultdict(int)      # barriers, persistent over the whole run
    nodes = []                # node stack S
    edges = []                # entering edge per node; edges[0] is a dummy
    on_path = set()
    sd_stack = []             # per-frame sd
    iters = []                # per-frame outgoing-edge iterator

    def fruitful(v, sd):
        b[v] = sd                                 # unconditional: kick-start
        queue = deque([(v, sd)])
        while queue:
            u, d = queue.popleft()
            for p in pred[u]:
                if p not in on_path and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def push(v, e):
        nodes.append(v)
        edges.append(e)
        on_path.add(v)
        iters.append(iter(get_edges(v)))
        if v in targets:                          # virtual edge v -> t*
            sd_stack.append(0)
            return True
        sd_stack.append(k + 1)
        return False

    if push(source, None):
        yield []

    while nodes:
        h = len(nodes) - 1                        # depth of the top node v
        e = next((e for e in iters[-1]
                  if e[1] not in on_path and b[e[1]] + h < k), None)
        if e is not None:
            if push(e[1], e):
                yield edges[1:]                   # slice copies; dummy removed
            continue

        # return from Search(v)
        v = nodes[-1]
        iters.pop()
        sd = sd_stack.pop()
        if sd <= k:
            fruitful(v, sd)                       # v still on the stack
        else:
            b[v] = k - h + 1
        nodes.pop()
        edges.pop()
        on_path.discard(v)
        if sd_stack:
            sd_stack[-1] = min(sd_stack[-1], sd + 1)


import random
from itertools import zip_longest, islice
from multiprocessing import Pool
from tqdm import tqdm


def compare_generators(gen1, gen2, limit, msg=""):
    sentinel = object()

    n_paths = 0
    for i, (a, b) in enumerate(zip_longest(
            islice(gen1, limit),
            islice(gen2, limit),
            fillvalue=sentinel)):

        if a is sentinel and b is sentinel:
            break  # both ended

        if a is sentinel:
            raise AssertionError(
                f"{msg} generator1 ended early at index {i}, "
                f"generator2 produced: {b!r}"
            )

        if b is sentinel:
            raise AssertionError(
                f"{msg} generator2 ended early at index {i}, "
                f"generator1 produced: {a!r}"
            )

        if a != b:
            raise AssertionError(
                f"first difference at index {i}: {a!r} vs {b!r}"
            )

        n_paths = i + 1

    return n_paths


def sample_instance(n, run):
    rng = random.Random(42 + run)
    m = int(n * 10 ** rng.uniform(-0.3, 1.3))          # log-uniform density
    G = nx.MultiDiGraph() if rng.random() < 0.5 else nx.MultiGraph()
    G.add_nodes_from(range(n))
    pairs = []
    for _ in range(m):
        if pairs and rng.random() < 0.4:               # duplicate an existing pair
            u, v = rng.choice(pairs)
        else:
            u, v = rng.randrange(n), rng.randrange(n)  # self-loops included
            pairs.append((u, v))
        G.add_edge(u, v)
    k = rng.randint(0, n)
    s = rng.randrange(n)
    t_set = set(rng.sample(range(n), rng.randint(1, min(n, 3))))
    return G, s, t_set, k


def worker_er(args, limit=10_000):
    n, run = args

    G, s, t_set, k = sample_instance(n, run)

    length = compare_generators(
        nx.algorithms.simple_paths._all_simple_edge_paths(G, s, t_set, cutoff=k),
        bsdfs_all_simple_edge_paths(G, s, t_set, k),
        limit,
        msg = f"{args=}"
    )

    return length


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs, processes=None):
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
        print(f"{n=:4} {sum_paths=:,}")
        return sum_paths


def monkey_patching_pytest():
    import networkx.algorithms.simple_paths as simple_paths

    orig = simple_paths._all_simple_edge_paths
    simple_paths._all_simple_edge_paths = bsdfs_all_simple_edge_paths

    import pytest
    pytest.main(["--doctest-modules", "--pyargs", "networkx"])
    # expected:  7720 passed, 87 skipped, 1 xfailed, 11 warnings in 76.91s (0:01:16)
    assert simple_paths._all_simple_edge_paths is bsdfs_all_simple_edge_paths # not restored mid-way
    # (the warnings are unrelated)

    simple_paths._all_simple_edge_paths = orig
    assert simple_paths._all_simple_edge_paths is not bsdfs_all_simple_edge_paths
    # now restored


def random_test():
    # quick test for major error and general failure
    for n in range(2, 5):
        validate_er(n, 1_000, processes=0)

    for n in range(2, 10):
        validate_er(n, 100_000)

    for n in range(10, 15):
        validate_er(n, 10_000)


def performance_test():
    import networkx as nx
    import random
    import time
    import math
    import matplotlib.pyplot as plt
    from itertools import islice

    def circulant(n, r):
        """directed r-neighbour circulant digraph"""
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u+1, u+r+1):
                G.add_edge(u, v % n)
        return G

    seed = 42
    random.seed(seed)

    runs = 1000
    limit = 1000
    r = 2
    xs = []
    ys = []
    cs = []
    for run in range(runs):
        n = random.randint(2, 35)   # danger, explosive!
        G = circulant(n, r)
        s, t = random.sample(list(G.nodes), 2)
        k = random.randrange(0, len(G))

        repeats = 3
        while True:
            dt1 = dt2 = 0
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                paths1 = list(islice(bsdfs_all_simple_edge_paths(G, s, [t], k), limit))
                dt1 += time.perf_counter_ns() - t0
                t0 = time.perf_counter_ns()
                paths2 = list(islice(nx.algorithms.simple_paths._all_simple_edge_paths(G, s, [t], k), limit))
                dt2 += time.perf_counter_ns() - t0
            if dt1 + dt2 > 1_000_000:
                break
            else:
                repeats = 2*repeats + 1 # stay odd
        dt1 //= repeats
        dt2 //= repeats
        assert paths1 == paths2 # likely
        print(f"{run=:6}; {n=:4}; {s=:4}; {t=:4}; {k=:4}; {dt1=:12}; {dt2=:12}; {dt1/dt2=:10.4f}; {len(paths1)=:10}")
        xs.append(dt1/1e9)  # seconds
        ys.append(dt2/1e9)  # seconds
        cs.append(math.log10(1 + k))

    fig, ax = plt.subplots()
    plt.grid(True)
    ax.set_title("path enum performance on circulant graphs (max. 1000 paths)")
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=18)
    lo, hi = min(xs+ys), max(xs+ys)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)      # identity line
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel(f"bsdfs time [s]"); ax.set_ylabel(f"all_simple_edge_paths time [s]")
    fig.colorbar(sc, label=f"log10(1+k)")
    fig.savefig("circulant_family.png")

if __name__ == "__main__":
    monkey_patching_pytest()
    random_test()
    performance_test()