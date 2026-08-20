"""BS-DFS with in-code assertions of the single-pass claims of the paper.

Every assertion below refers to a numbered statement of bsdfs.tex (current
release).  Only claims that can be decided from the state of one execution
*at the moment the assertion is reached* are checked; the delay bounds and
everything else that needs the phase/period/fork decomposition of a whole run
is deliberately out of scope here.

Two conventions of the paper are followed literally:
  * h = ||S|| is the length of the search path in *edges*, i.e. len(S) - 1
    after v has been appended;
  * P denotes the prefix (search path before v is pushed), Pv the search path
    during the call.

The checked version assumes s != t.  The cycle case s = t is covered by
Lemma (Cycle Reduction); `split_source` builds the graph G' of that lemma, so an
s-cycle search can be run and checked as check_paths(*split_source(G, s), k).
The random harness does not do so: cycle mode is a path search on G' and adds no
BS-DFS invariant of its own.

The ground truth is the plain DFS `simple_paths` below, not networkx: the reference
must not share code, and hence bugs, with the implementation under test, and it must
not change underneath the harness.  networkx is used for graph storage and sampling only.

Run with asserts enabled (no -O / -OO).
"""

import hashlib
import math
import random
import time
from collections import Counter, deque
from itertools import islice
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx

INF = float("inf")


# ---------------------------------------------------------------------------
# helpers used only by the assertions
# ---------------------------------------------------------------------------

def dist_to(G, target, forbidden):
    """dist_forbidden(x, target) for every node x, cf. Definition (Relative Distance).

    Returns the length of a shortest x-target-path whose nodes meet `forbidden`
    at most in the endpoints {x, target}; INF if no such path exists.
    One backward BFS suffices: a node is *expanded* (i.e. used as an interior node
    of the path) only if it is the target itself or lies outside `forbidden`, while
    a distance is *assigned* to every node, including nodes of `forbidden`, which are
    legal as the start x (endpoint exemption).
    """
    d = {x: INF for x in G.nodes}
    d[target] = 0
    Q = deque([target])
    while Q:
        y = Q.popleft()
        for x in G.predecessors(y):
            if d[x] == INF:
                d[x] = d[y] + 1
                if x not in forbidden:
                    Q.append(x)
    return d


def inconsistent_edge(G, b, S, exempt_source=None):
    """First edge violating Definition (Edge-Consistent Labeling) w.r.t. S, else None.

    Constrained are the edges (x, y) with {x, y} cap S = empty; the constraint is
    b[x] <= b[y] + 1.  `exempt_source` additionally skips all edges leaving that
    node, which is the exception left open by Lemma (Update Repairs).
    """
    for x, y in G.edges:
        if x in S or y in S:
            continue
        if x == exempt_source:
            continue
        if b[x] > b[y] + 1:
            return (x, y, b[x], b[y])
    return None


# ---------------------------------------------------------------------------
# BS-DFS, instrumented
# ---------------------------------------------------------------------------

def bsdfs_checked(G, s, t, k):
    """Tight scheme (Algorithm BS-DFS / search / fruitful) with assertions."""
    assert s in G.nodes and t in G.nodes
    assert s != t, "checked variant assumes s != t; use split_source() for s-cycles (Lemma Cycle Reduction)"
    assert 0 < k <= G.number_of_nodes(), "standing assumption 0 < k <= n of Section 'Algorithm BS-DFS'"
    assert not any(x == y for x, y in G.edges), "graphs are simple: no self-loops"

    b = {x: 0 for x in G.nodes}
    S = []                  # search path, exactly the stack of the pseudocode
    on_stack = set()        # 'Note for implementations': O(1) membership test
    cascades = 0            # number of fruitful() executions so far
    seen_search_paths = set()
    seen_outputs = set()

    def fruitful(v, sd):
        nonlocal cascades
        cascades += 1

        Pv = frozenset(on_stack)            # v is on the stack during the whole cascade
        P = Pv - {v}
        assert inconsistent_edge(G, b, Pv) is None, "Lemma (Cascade Distance): edge-consistency w.r.t. Pv at cascade entry"

        b_entry = dict(b)
        dist_P_t = dist_to(G, t, P)         # dist_P(., t)
        dist_Pv_v = dist_to(G, v, Pv)       # dist_Pv(., v)
        enqueued = {v}

        b[v] = sd
        Q = deque([(v, sd)])
        while Q:
            q, d = Q.popleft()
            assert b[q] == d, "each node is relaxed at most once, so the queued value is still current"
            for p in G.predecessors(q):
                if p not in on_stack and b[p] > d + 1:
                    assert p not in enqueued, "FIFO order relaxes every node at most once (proof of Lemma (Cascade Distance))"
                    assert b[p] > d + 1, "Observation (9): every cascade assignment strictly decreases the barrier"
                    b[p] = d + 1
                    assert b[p] >= 2 > 0, "Lemma (Barrier non-negative), case 4: cascade assignments are >= 2"
                    assert b[p] >= dist_P_t[p], "Lemma (Fruitful Distance Lower Bound): b[u] >= dist_P(u,t)"
                    Q.append((p, d + 1))
                    enqueued.add(p)

        for x in G.nodes:
            assert x == v or b[x] <= b_entry[x], "Observation (9): apart from the call's own write b[v] <- sd, a cascade never raises a barrier"
            if x in P:
                continue
            assert b[x] <= sd + dist_Pv_v[x], "Lemma (Cascade Distance), upper bound for x not in P"
            if x in enqueued:
                assert b[x] == sd + dist_Pv_v[x], "Lemma (Cascade Distance), equality for enqueued nodes"

        assert inconsistent_edge(G, b, P, exempt_source=v) is None, "Lemma (Update Repairs): edge-consistency w.r.t. P at cascade exit, except for edges leaving v"

    def search(v):
        # ---- entry -------------------------------------------------------
        assert v not in on_stack, "Observation (1): every search path is simple"
        P = frozenset(on_stack)                     # prefix, before v is pushed
        assert inconsistent_edge(G, b, P) is None, "entry hypothesis of the joint induction: edge-consistency w.r.t. P"

        S.append(v)
        on_stack.add(v)
        h = len(S) - 1
        Pv = frozenset(on_stack)

        assert 0 <= h <= k, "Lemma (Search Path Length)"
        assert b[v] <= k - h, "Lemma (Parent Pruning Guard)"
        assert tuple(S) not in seen_search_paths, "Observation (3): every search path occurs at most once"
        seen_search_paths.add(tuple(S))
        assert all(0 <= b[x] <= k for x in G.nodes), "Corollary (Barrier Upper Bound): 0 <= b[x] <= k at every read"
        assert b[t] == 0, "Observation (8): b[t] = 0 throughout"
        assert b[s] == 0, "Observation (7): b[s] = 0 until the final assignment of the initial call"

        b_entry = dict(b)
        cascades_at_entry = cascades
        dist_Pv_t = dist_to(G, t, Pv)               # dist_Pv(., t); constant during the call

        sd = k + 1
        for w in G.successors(v):
            assert inconsistent_edge(G, b, Pv) is None, "loop invariant: edge-consistency w.r.t. Pv whenever a successor is examined"
            if w not in Pv:
                assert b[w] <= dist_Pv_t[w], "Lemma (Barrier Distance Bound) with b[t] = 0: admissibility at the moment of the test"

            if b[w] + h < k:
                if w == t:
                    output = S + [t]
                    assert output[0] == s and output[-1] == t, "Theorem (Soundness): output is an s-t-path"
                    assert len(set(output)) == len(output), "Theorem (Soundness): output is simple"
                    assert all(G.has_edge(x, y) for x, y in zip(output, output[1:])), "Theorem (Soundness): output is a path in G"
                    assert len(output) - 1 <= k, "Theorem (Soundness): output length <= k"
                    assert tuple(output) not in seen_outputs, "Observation (4): no output is produced twice"
                    seen_outputs.add(tuple(output))
                    yield output
                    sd = 1
                elif w not in on_stack:
                    d = yield from search(w)
                    sd = min(sd, d + 1)
            else:
                assert w in Pv or h + 1 + dist_Pv_t[w] > k, "Lemma (Pruning is Permissive): a pruned successor starts no feasible completion"

        # ---- post-loop ---------------------------------------------------
        assert b[v] == b_entry[v], "Observation (6): b[v] is written only by the call's final assignment"
        assert sd >= 1, "Observation (5): return values are non-negative (in fact >= 1)"
        is_fruitful = sd <= k
        assert is_fruitful == (h + dist_Pv_t[v] <= k), "Lemma (Call Completeness) + Theorem (Soundness): the call is fruitful iff a feasible completion of Pv exists"

        if is_fruitful:
            assert sd <= k - h, "Corollary (Fruitful Return)"
            assert sd >= dist_Pv_t[v], "Lemma (Fruitful Lower Bound)"
            assert sd == dist_Pv_t[v], "Lemma (Strict Barrier Invariant): sd = dist_Pv(v,t)"
            assert sd >= 1, "Lemma (Barrier non-negative), case 2"
            fruitful(v, sd)
        else:
            assert sd == k + 1, "the fruitless case leaves sd at its initial value"
            assert cascades == cascades_at_entry, "Lemma (Fruitless Monotonicity): no fruitful nested call, hence no cascade, within a fruitless call"
            assert all(b[x] >= b_entry[x] for x in G.nodes), "Lemma (Fruitless Monotonicity): all barriers non-decreasing during a fruitless call"
            assert k - h + 1 > b_entry[v], "Lemma (Fruitless Increasing): b_exit[v] > b_entry[v]"
            assert k - h + 1 > 0, "Lemma (Barrier non-negative), case 3"
            assert k - h + 1 <= k or (h == 0 and v == s), "Corollary (Barrier Upper Bound): the value k+1 occurs only as the terminal sentinel b[s]"
            b[v] = k - h + 1

        # ---- exit --------------------------------------------------------
        S.pop()
        on_stack.discard(v)

        assert all(b[x] >= b_entry[x] for x in G.nodes), "Lemma (Per-Call Global Monotonicity)"
        assert inconsistent_edge(G, b, P) is None, "Lemma (Search Preserves): edge-consistency w.r.t. P at exit"
        sentinel = (h == 0 and not is_fruitful)     # terminal b[s] = k+1 of a completely fruitless run
        assert all(0 <= b[x] <= (k + 1 if sentinel and x == s else k) for x in G.nodes), "Corollary (Barrier Upper Bound)"
        assert b[t] == 0, "Observation (8): b[t] = 0 throughout"

        return sd

    yield from search(s)

    # ---- after the run -----------------------------------------------------
    assert all(b[x] >= 0 for x in G.nodes), "Lemma (Barrier non-negative)"


# ---------------------------------------------------------------------------
# Lemma (Cycle Reduction): s-cycles as s_o-s_i-paths
# ---------------------------------------------------------------------------

def split_source(G, s):
    """The graph G' of Lemma (Cycle Reduction), together with s_o and s_i."""
    s_o, s_i = (s, "o"), (s, "i")
    H = nx.DiGraph()
    H.add_nodes_from(x for x in G.nodes if x != s)
    H.add_nodes_from([s_o, s_i])
    for x, y in G.edges:
        H.add_edge(s_o if x == s else x, s_i if y == s else y)
    return H, s_o, s_i


# ---------------------------------------------------------------------------
# ground truth and per-instance check
# ---------------------------------------------------------------------------

from simple_search import simple_search


def check_paths(G, s, t, k, limit=None):
    """Cross-check the asserted run against the DFS ground truth.

    Both enumerations are depth-first in successor-list order, so their output
    *sequences* agree and a prefix comparison is exact: with `limit` set, only the
    first `limit` outputs of each are compared and everything beyond the cutoff stays
    unexamined -- the generator is then abandoned, so the assertions of the calls still
    open at the cutoff never run.
    """
    paths = [tuple(p) for p in islice(bsdfs_checked(G, s, t, k), limit)]
    reference = [tuple(p) for p in islice(simple_search(G, s, t, k), limit)]
    assert len(paths) == len(set(paths)), "Observation (4): no output is produced twice"
    assert paths == reference, f"output differs at index {next((i for i, (a, r) in enumerate(zip(paths, reference)) if a != r), min(len(paths), len(reference)))}"
    try:
        adjacency_sorted = all(list(G.successors(v)) == sorted(G.successors(v)) for v in G.nodes)
    except TypeError:                   # node labels not totally ordered, as in the split source of Lemma (Cycle Reduction)
        adjacency_sorted = False
    if adjacency_sorted:
        assert paths == sorted(paths), "for sorted adjacency lists that order is the lexicographic one"
    return len(paths)


# ---------------------------------------------------------------------------
# instances: an instance is a pure function of (seed, run, nmax), so a failing run is
# replayable from its index alone and nothing but small tuples is pickled
# ---------------------------------------------------------------------------

def instance(seed, run, nmax=9):
    """The (G, s, t, k) of a given run.  Seeding from a string is deterministic across processes."""
    rng = random.Random(f"{seed}:{run}")
    n = rng.randint(3, nmax)
    m = int(n * math.exp(rng.uniform(0, math.log(n - 1))))
    G = nx.gnm_random_graph(n, m, directed=True, seed=rng.randint(0, 10**9))
    k = rng.randint(1, n)
    s, t = rng.sample(range(n), 2)
    return G, s, t, k


def run_key(G, s, t, k):
    """Everything that determines the execution: node order, successor order (search loop),
    predecessor order (fruitful cascade), and the parameters."""
    return (tuple(G.nodes),
            tuple(tuple(G.successors(v)) for v in G.nodes),
            tuple(tuple(G.predecessors(v)) for v in G.nodes),
            s, t, k)


def digest(key):
    """Stable 64-bit fingerprint; unlike hash() it does not depend on the interpreter or the key's types."""
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")


def replay(seed, run, nmax=9, limit=None):
    """Re-run a single instance in-process, with the AssertionError propagating for post-mortem debugging."""
    G, s, t, k = instance(seed, run, nmax)
    print(f"seed={seed} run={run}: n={G.number_of_nodes()} m={G.number_of_edges()} {s=} {t=} {k=} {limit=} edges={list(G.edges)}")
    return check_paths(G, s, t, k, limit)


# ---------------------------------------------------------------------------
# worker / pool harness
# ---------------------------------------------------------------------------

def worker(args):
    """One run.  Returns (run, n, paths, instance_digest, graph_digest, failure)."""
    seed, run, nmax, limit, catch = args
    G, s, t, k = instance(seed, run, nmax)
    key = run_key(G, s, t, k)
    fingerprints = (digest(key), digest(key[:3]))
    try:
        paths = check_paths(G, s, t, k, limit)
    except AssertionError as failure:
        if not catch:
            raise
        return (run, G.number_of_nodes(), 0) + fingerprints + (f"{type(failure).__name__}: {failure}",)
    return (run, G.number_of_nodes(), paths) + fingerprints + (None,)


def tasks(seed, runs, first, nmax, limit, catch):
    for run in range(first, first + runs):
        yield (seed, run, nmax, limit, catch)


def validate(runs, seed=42, processes=None, first=0, nmax=9, limit=None, chunksize=64, track_keys=True, stop_after=10):
    """Run `runs` instances.  processes=0 runs sequentially in-process and lets assertions
    propagate (use for debugging); otherwise failures are collected and reported by run index."""
    sequential = (processes == 0)
    seen_instances, seen_graphs = set(), set()
    per_n = Counter()
    total_paths = done = 0
    failures = []

    work = tasks(seed, runs, first, nmax, limit, catch=not sequential)
    tick = time.perf_counter()
    if sequential:
        results = map(worker, work)
        pool = None
    else:
        pool = Pool(processes)
        results = pool.imap_unordered(worker, work, chunksize=chunksize)

    try:
        for run, n, paths, key_digest, graph_digest, failure in tqdm(results, total=runs, leave=False):
            done += 1
            per_n[n] += 1
            total_paths += paths
            if track_keys:
                seen_instances.add(key_digest)
                seen_graphs.add(graph_digest)
            if failure is not None:
                failures.append((run, failure))
                print(f"\nFAILED seed={seed} run={run}: {failure}\n  replay with: replay({seed}, {run}, nmax={nmax}, limit={limit})")
                if len(failures) >= stop_after:
                    break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    elapsed = time.perf_counter() - tick
    print(f"nmax={nmax} limit={limit}: {done:,} runs in {elapsed:.1f} s ({done / max(elapsed, 1e-9):,.0f} runs/s), {total_paths:,} s-t-paths")
    if track_keys:
        print(f"distinct instances: {len(seen_instances):,}, distinct graphs: {len(seen_graphs):,}")
    print(f"runs per n: {dict(sorted(per_n.items()))}")
    print(f"failures: {len(failures)}" if failures else "all assertions passed")
    return failures


def smoke():
    """Small and sequential: assertions propagate with a live traceback."""
    return validate(runs=2_000, seed=42, processes=0)


def main():
    """Let the fans roar: exhaustive small instances, then truncated larger ones."""
    validate(runs=5_000_000, seed=42, processes=None, nmax= 9, limit=None, chunksize=256)
    validate(runs=4_000_000, seed=43, processes=None, nmax=20, limit= 100, chunksize= 64)
    validate(runs=1_000_000, seed=44, processes=None, nmax=50, limit=  10, chunksize=  8)


if __name__ == "__main__":
    smoke()
    main()

# reference output
# nmax=9 limit=None: 2,000 runs in 5.1 s (395 runs/s), 89,926 s-t-paths                                                                                                                                                                           
# distinct instances: 1,986, distinct graphs: 1,861
# runs per n: {3: 271, 4: 255, 5: 300, 6: 307, 7: 289, 8: 315, 9: 263}
# all assertions passed
# nmax=9 limit=None: 5,000,000 runs in 1197.6 s (4,175 runs/s), 284,255,136 s-t-paths                                                                                                                                                             
# distinct instances: 4,191,209, distinct graphs: 3,844,602
# runs per n: {3: 715243, 4: 713510, 5: 713935, 6: 714219, 7: 715569, 8: 712813, 9: 714711}
# all assertions passed
# nmax=20 limit=100: 4,000,000 runs in 1436.3 s (2,785 runs/s), 126,315,697 s-t-paths                                                                                                                                                             
# distinct instances: 3,769,620, distinct graphs: 3,679,272
# runs per n: {3: 221651, 4: 222444, 5: 222516, 6: 222162, 7: 222383, 8: 221571, 9: 222243, 10: 221348, 11: 221655, 12: 222634, 13: 222450, 14: 222080, 15: 222273, 16: 222826, 17: 222653, 18: 222388, 19: 222361, 20: 222362}
# all assertions passed
# nmax=50 limit=10: 1,000,000 runs in 879.1 s (1,138 runs/s), 6,402,155 s-t-paths                                                                                                                                                                 
# distinct instances: 982,549, distinct graphs: 975,415
# runs per n: {3: 21120, 4: 21303, 5: 20760, 6: 20677, 7: 20900, 8: 21061, 9: 20881, 10: 20942, 11: 20962, 12: 20792, 13: 20648, 14: 20639, 15: 20818, 16: 20876, 17: 20729, 18: 20930, 19: 20907, 20: 20881, 21: 20882, 22: 21062, 23: 20787, 24: 20887, 25: 20720, 26: 20786, 27: 20913, 28: 20613, 29: 20703, 30: 21062, 31: 20828, 32: 20682, 33: 20804, 34: 20801, 35: 20778, 36: 20681, 37: 20855, 38: 20722, 39: 20866, 40: 20841, 41: 20684, 42: 20746, 43: 20920, 44: 20788, 45: 20751, 46: 20804, 47: 20870, 48: 20650, 49: 20788, 50: 20900}
# all assertions passed
