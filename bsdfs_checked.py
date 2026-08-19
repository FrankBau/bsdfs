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
Lemma (Cycle Reduction); `split_source` builds the graph G' of that lemma so
that s-cycle search can be checked as an s_o-s_i-path search.

Run with asserts enabled (no -O / -OO).
"""

import hashlib
import math
import random
import time
from collections import Counter, deque
from multiprocessing import Pool

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
# validation harness (small graphs only: the assertions are quadratic per call)
# ---------------------------------------------------------------------------

def check_paths(G, s, t, k):
    """Cross-check the asserted run against networkx.all_simple_paths."""
    paths = [tuple(p) for p in bsdfs_checked(G, s, t, k)]
    reference = [tuple(p) for p in nx.all_simple_paths(G, s, t, k)]
    assert len(paths) == len(set(paths)), "Observation (4): no output is produced twice"
    assert set(paths) == set(reference), f"output differs: missing={set(reference) - set(paths)} spurious={set(paths) - set(reference)}"
    assert paths == reference, "both are depth-first in successor-list order, so even the output order agrees"
    if all(list(G.successors(v)) == sorted(G.successors(v)) for v in G.nodes):
        assert paths == sorted(paths), "for sorted adjacency lists that order is the lexicographic one"
    return len(paths)


def check_cycles(G, s, k):
    """Cross-check s-cycle search via Lemma (Cycle Reduction) against networkx.simple_cycles."""
    H, s_o, s_i = split_source(G, s)
    cycles = set()
    for p in bsdfs_checked(H, s_o, s_i, k):
        assert p[0] == s_o and p[-1] == s_i
        cycles.add(tuple([s] + list(p[1:-1])))
    reference = set()
    for c in nx.simple_cycles(G, length_bound=k):
        if s in c:
            i = c.index(s)
            reference.add(tuple(c[i:] + c[:i]))
    assert cycles == reference, f"cycles differ: missing={reference - cycles} spurious={cycles - reference}"
    return len(cycles)


# ---------------------------------------------------------------------------
# instances: an instance is a pure function of (seed, run), so a failing run is
# replayable from its index alone and nothing but small tuples is pickled
# ---------------------------------------------------------------------------

def instance(seed, run):
    """The (G, s, t, k) of a given run.  Seeding from a string is deterministic across processes."""
    rng = random.Random(f"{seed}:{run}")
    n = rng.randint(3, 9)
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


def replay(seed, run):
    """Re-run a single instance in-process, with the AssertionError propagating for post-mortem debugging."""
    G, s, t, k = instance(seed, run)
    print(f"seed={seed} run={run}: n={G.number_of_nodes()} m={G.number_of_edges()} {s=} {t=} {k=} edges={list(G.edges)}")
    return check_paths(G, s, t, k), check_cycles(G, s, k)


# ---------------------------------------------------------------------------
# worker / pool harness
# ---------------------------------------------------------------------------

def worker(args):
    """One run.  Returns (run, n, paths, cycles, instance_digest, graph_digest, failure)."""
    seed, run, catch = args
    G, s, t, k = instance(seed, run)
    key = run_key(G, s, t, k)
    fingerprints = (digest(key), digest(key[:3]))
    try:
        paths = check_paths(G, s, t, k)
        cycles = check_cycles(G, s, k)
    except AssertionError as failure:
        if not catch:
            raise
        return (run, G.number_of_nodes(), 0, 0) + fingerprints + (f"{type(failure).__name__}: {failure}",)
    return (run, G.number_of_nodes(), paths, cycles) + fingerprints + (None,)


def tasks(seed, runs, first, catch):
    for run in range(first, first + runs):
        yield (seed, run, catch)


def validate(runs, seed=42, processes=None, first=0, chunksize=64, track_keys=True, stop_after=10):
    """Run `runs` instances.  processes=0 runs sequentially in-process and lets assertions
    propagate (use for debugging); otherwise failures are collected and reported by run index."""
    sequential = (processes == 0)
    seen_instances, seen_graphs = set(), set()
    per_n = Counter()
    total_paths = total_cycles = done = 0
    failures = []

    work = tasks(seed, runs, first, catch=not sequential)
    tick = time.perf_counter()
    if sequential:
        results = map(worker, work)
        pool = None
    else:
        pool = Pool(processes)
        results = pool.imap_unordered(worker, work, chunksize=chunksize)

    try:
        for run, n, paths, cycles, key_digest, graph_digest, failure in results:
            done += 1
            per_n[n] += 1
            total_paths += paths
            total_cycles += cycles
            if track_keys:
                seen_instances.add(key_digest)
                seen_graphs.add(graph_digest)
            if failure is not None:
                failures.append((run, failure))
                print(f"\nFAILED seed={seed} run={run}: {failure}\n  replay with: replay({seed}, {run})")
                if len(failures) >= stop_after:
                    break
            if done % 10 == 0:
                print(f"{done:9,} / {runs:,}", end="\r", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    elapsed = time.perf_counter() - tick
    print(f"{done:,} runs in {elapsed:.1f} s ({done / max(elapsed, 1e-9):,.0f} runs/s), {total_paths:,} s-t-paths, {total_cycles:,} s-cycles")
    if track_keys:
        print(f"distinct instances: {len(seen_instances):,}, distinct graphs: {len(seen_graphs):,}")
    print(f"runs per n: {dict(sorted(per_n.items()))}")
    print(f"failures: {len(failures)}" if failures else "all assertions passed")
    return failures


def smoke():
    """Small and sequential: assertions propagate with a live traceback."""
    return validate(runs=2_000, seed=42, processes=0)


def main():
    """Let the fans roar."""
    return validate(runs=10_000_000, seed=42, processes=None, chunksize=256)


if __name__ == "__main__":
    smoke()
    main()