"""Johnson-style fallback for the unbounded case of bsdfs_all_simple_edge_paths.
 
Contract (identical to the BS-DFS replacement / stock generator):
  - `targets` arrives as a set (wrapper already normalized),
  - yields edge paths: (u, v) tuples, (u, v, key) for multigraphs,
    once per parallel-edge combination, native key iteration order,
  - trivial path (source in targets) is the empty edge path [],
  - works for directed / undirected, simple / multi.
 
Algorithm: Johnson's blocked DFS on the implicit G' = G + t* with virtual
arcs T -> t*, virtual arc first in Adj:
  entry hit   v in targets  =>  yield, frame fruitful
  fruitful    on exit: Unblock(v), mark parent fruitful
  fruitless   on exit: register v on B(w) for all real successors w
 
Delay O(n+m); completeness by Johnson's blocking lemma (u blocked after
its call  =>  every u->t* path meets the current stack).
 
Output order equals bsdfs_all_simple_edge_paths(G, source, targets, len(G)-1)
and stock: all are same-edge-order DFS with entry yield, pruning only
fruitless subtrees, hence share the unpruned DFS's output sequence.
 
Iterative (no recursion limit, no yield-from chain overhead).
"""
 
from collections import defaultdict
 
 
def johnson_all_simple_edge_paths(G, source, targets):
    if not targets:
        return
 
    adj = G._adj  # successors (directed) / neighbors (undirected)
 
    if G.is_multigraph():
        def out_edges(v):
            for w, keydict in adj[v].items():
                for key in keydict:
                    yield (v, w, key)
    else:
        def out_edges(v):
            for w in adj[v]:
                yield (v, w)
 
    blocked = set()
    B = defaultdict(set)
 
    def unblock(v):
        pending = [v]
        while pending:
            u = pending.pop()
            if u in blocked:
                blocked.discard(u)
                pending.extend(B[u])
                B[u].clear()
 
    edge_path = []
    # frame: [node, out-edge iterator, fruitful]
    blocked.add(source)
    stack = [[source, out_edges(source), source in targets]]
    if source in targets:
        yield []
 
    while stack:
        frame = stack[-1]
        for e in frame[1]:
            w = e[1]
            if w not in blocked:
                edge_path.append(e)
                blocked.add(w)
                stack.append([w, out_edges(w), w in targets])
                if w in targets:
                    yield list(edge_path)
                break
        else:
            # frame[0] exhausted: close the call
            stack.pop()
            v = frame[0]
            if frame[2]:
                unblock(v)
                if stack:
                    stack[-1][2] = True
            else:
                for w in adj[v]:
                    B[w].add(v)
            if stack:
                edge_path.pop()


import networkx as nx
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
    s = rng.randrange(n)
    t_set = set(rng.sample(range(n), rng.randint(1, min(n, 3))))
    return G, s, t_set


def worker_er(args, limit=10_000):
    n, run = args

    G, s, t_set = sample_instance(n, run)

    length = compare_generators(
        nx.all_simple_edge_paths(G, s, t_set),
        johnson_all_simple_edge_paths(G, s, t_set),
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


def random_test():
    # quick test for major error and general failure
    for n in range(2, 5):
        validate_er(n, 1_000, processes=0)

    for n in range(2, 10):
        validate_er(n, 100_000)

    for n in range(10, 15):
        validate_er(n, 10_000)


if __name__ == "__main__":
    random_test()
