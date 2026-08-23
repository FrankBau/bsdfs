"""Bounded-Scope Depth-First Search (BS-DFS) for length-bounded simple path and cycle enumeration."""


# :func:`bsdfs` enumerates all length-bounded simple paths or cycles
# in a graph extending a given prefix path.
#
# :func:`bsdfs` is used in NetworkX functions `all_simple_edge_paths` and
# `simple_cycles` when a length bound is specified.
#
# Cycle mode and path mode
# ------------------------
# The two modes differ in one bit, selected by ``targets``:
#
# * ``targets is None`` -- *cycle mode*.  The implied target is ``prefix[0]``,
#   so the walk reported returns to the node it started from.
# * ``targets`` a set -- *path mode*.  The walk reported ends at a target.
#   Targets are otherwise ordinary nodes, so a walk may run through one on its
#   way to another.
#
# In path mode ``prefix`` and ``targets`` need not be disjoint, but a target on
# the prefix cannot be reached again.  The only visible effect is that
# ``prefix[-1] in targets`` yields the empty extension, which is NetworkX's
# trivial path.
#
# Implementation overview
# -----------------------
# An iterative depth-limited depth-first search starting at ``prefix[-1]``.
# A set of forbidden nodes -- the prefix, plus everything currently on the search
# stack -- keeps the reported walks simple.
#
# The modes differ only in how the last step is taken.  In cycle mode the root
# stays forbidden and is never entered, so the edge ``v -> prefix[0]`` is reported
# where it is found and consumes one edge of the bound.  In path mode a target
# is entered like any other node and the path ending there is reported on
# arrival, consuming nothing further.  Both are the same rule seen through a
# virtual target ``t*``, joined to every target by an edge of length 1 resp. 0.
#
# To prevent excessive fruitless searches, the search is pruned by node barriers.
#
# Barriers
# --------
# The search maintains one integer per node, ``b[v]``, a *certified lower bound*
# on the number of edges from ``v`` to ``t*`` in the graph minus the nodes
# currently on the search stack ``stack``.  All barriers start at 0, which is
# trivially valid.  With ``v`` on top of ``stack`` at depth ``h`` (so ``h`` edges
# from the root to ``v``), a successor ``w`` is *admissible* iff
#
#     b[w] + h < k                                                        (A)
#
# Rationale: entering ``w`` costs one edge, and ``b[w]`` further edges are needed
# before ``t*`` can possibly be reached, so any output through ``w`` has length
# at least ``h + 1 + b[w]``; requiring that to be at most ``k`` is exactly (A).
# Because ``b[w]`` is a *lower* bound, (A) never discards an output: it prunes
# only branches that provably cannot finish within the budget.
#
# Barriers are written in exactly two places, both on return from ``Search(v)``
# with ``v`` at depth ``h``:
#
# * **unfruitful return** -- ``v`` produced no output.  The subsearch had a
#   budget of ``k - h`` edges and exhausted it, so ``t*`` is farther than that:
#
#       b[v] = k - h + 1                                    (a *raise*)
#
# * **fruitful return** -- ``v`` produced an output, and ``sd`` is the exact
#   number of edges from ``v`` to ``t*`` along the shortest output found below
#   ``v``.  Setting ``b[v] = sd`` may *invalidate* the barriers of predecessors,
#   which were justified relative to a larger distance from ``v``.  A backward
#   BFS restores the invariant
#
#       b[u] <= b[w] + 1     for every edge u -> w with u not on stack       (EC)
#
#   ("edge-consistency"), lowering barriers where needed; see ``cascade`` below.
#   Nodes on ``stack`` are skipped: their barriers are written when they are popped,
#   not while they are forbidden.
#
# Edge-consistency is what makes the pop obligation local, and it is why the
# cascade terminates quickly: a node is re-entered by the BFS only when its
# barrier strictly drops, and each barrier only ever moves within ``[0, k+1]``.
#
# Delay
# -----
# Every cascade is triggered by a fruitful return, i.e. it is charged to an
# output that has already been emitted.  Together with the raise bound this
# gives worst-case delay ``3(k+1)(n+m)`` and amortized delay ``2(k+1)(n+m)``,
# with ``n`` nodes and ``m`` edges -- in particular ``O(k(n+m))`` with a small
# constant.  See [1]_.
#
# Prior work
# ----------
# Two earlier algorithms for the same enumeration problems are listed here as
# prior art, not as alternatives: [1]_ and [2]_ demonstrate inputs on which
# [3]_ and [4]_ omit valid outputs.
#
# References
# ----------
# .. [1] Frank Bauernoeppel, Joerg-Ruediger Sack,
#     "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs
#     with $O(k(n+m))$ Delay Using Edge-Consistent Node Barriers", 2026,
#     https://arxiv.org/abs/2607.14745
# .. [2] Frank Bauernoeppel, Joerg-Ruediger Sack,
#     "Finding All Bounded-Length Simple Cycles in a Directed Graph --
#     Revisited", 2025, https://arxiv.org/abs/2512.08392
# .. [3] Y. Peng et al., "Efficient Hop-constrained s-t Simple Path
#     Enumeration", The VLDB Journal 30(5):799-823, 2021,
#     https://doi.org/10.1007/s00778-021-00674-5
# .. [4] A. Gupta and T. Suzumura, "Finding All Bounded-Length Simple Cycles
#     in a Directed Graph", 2021, https://arxiv.org/abs/2105.10094

from collections import defaultdict, deque

import networkx as nx

__all__ = ["bsdfs"]


_NO_ROOT = object()  # sentinel: never equal to any node


class _OutEdgeCache(dict):
    """Caches, per node, the list of its outgoing edges as tuples.

    Same purpose as ``_NeighborhoodCache`` in ``cycles.py`` -- avoid the
    per-access cost of subgraph views -- but stores edge tuples ``(v, w)``,
    resp. ``(v, w, key)`` for multigraphs, since the path caller needs the
    multigraph key and cannot recover it from the node sequence.
    """

    def __init__(self, G):
        self.G = G
        self.keys_ = G.is_multigraph()

    def __missing__(self, v):
        G = self.G
        out = self[v] = list(G.edges(v, keys=True) if self.keys_ else G.edges(v))
        return out


class _InNodeCache(dict):
    """Caches, per node, the list of its predecessors.  The cascade direction."""

    def __init__(self, adj):
        self.adj = adj

    def __missing__(self, v):
        out = self[v] = list(self.adj[v])
        return out


@nx._dispatchable
def bsdfs(G, prefix, targets, k):
    """Yield all length-bounded simple paths or cycles extending ``prefix`` to ``targets``.

    Parameters
    ----------
    G : NetworkX graph
        Directed or undirected, graph or multigraph.
    prefix : list
        Non-empty prefix, a simple path in ``G`` given as a node list.
        Not mutated.
    targets : set or None
        Selects the mode.  Not mutated.

        * a set -- *path mode*.  The walk reported ends at a target node.
          Targets are otherwise ordinary nodes, so a walk may run through
          one on its way to another.
        * ``None`` -- *cycle mode*.  The implied target is ``prefix[0]``,
          so the walk reported returns to the node it started from.
    k : int
        Length bound in edges, counted from ``prefix[0]``, i.e. the prefix
        already consumes ``len(prefix) - 1`` of the budget.

    Yields
    ------
    list of edges
        The edges extending ``prefix[-1]`` to some node in ``targets``.
        If ``targets`` is ``None``, cycles to ``prefix[0]`` are yielded.
        Empty only for the trivial path, i.e. when ``prefix[-1]`` is itself a target.

    Raises
    ------
    ValueError
        If ``k`` is negative, ``prefix`` is empty, or ``targets`` is an empty set.
    NodeNotFound
        If a node of ``prefix`` is not in ``G``.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)])

    Cycle mode.  The cycle 0, 1, 2, 3 has 4 edges and exceeds the bound, so
    only two are reported.  The caller prepends the prefix to obtain nodes.

    >>> list(bsdfs(G, [0], None, 3))
    [[(0, 1), (1, 3), (3, 0)], [(0, 2), (2, 3), (3, 0)]]
    >>> [[e[0] for e in E] for E in bsdfs(G, [0], None, 3)]
    [[0, 1, 3], [0, 2, 3]]

    A longer prefix restricts the search, and only the extension is reported.

    >>> list(bsdfs(G, [0, 1], None, 3))
    [[(1, 3), (3, 0)]]
    >>> [[0] + [e[0] for e in E] for E in bsdfs(G, [0, 1], None, 3)]
    [[0, 1, 3]]

    Path mode, to one target and to a set of targets.  Targets are entered
    like any other node, so 0, 1, 2, 3 is reported although it runs through
    the target 2.

    >>> list(bsdfs(G, [0], {3}, 3))
    [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2), (2, 3)]]
    >>> list(bsdfs(G, [0], {2, 3}, 3))
    [[(0, 1), (1, 2)], [(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2)], [(0, 2), (2, 3)]]

    A target on the prefix is forbidden and cannot be entered again, so
    ``prefix[-1]`` in ``targets`` adds the empty extension and nothing else.

    >>> list(bsdfs(G, [0], {0, 3}, 3))
    [[], [(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2), (2, 3)]]

    See Also
    --------
    :func:`~networkx.algorithms.simple_paths.all_simple_edge_paths`
    :func:`~networkx.algorithms.simple_paths.all_simple_paths`
    :func:`~networkx.algorithms.cycles.simple_cycles`

    Notes
    -----
    Between consecutive outputs the algorithm performs at most
    ``3(k+1)(n+m)`` elementary steps on a graph with ``n`` nodes and ``m``
    edges, and at most ``2(k+1)(n+m)`` amortized over all outputs [1]_.
    One elementary step is a single adjacency-list entry scanned, plus
    constant bookkeeping per call and per barrier update, so the delay is
    ``O(k(n+m))``.

    References
    ----------
    .. [1] Frank Bauernoeppel, Joerg-Ruediger Sack,
        "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs
        with $O(k(n+m))$ Delay Using Edge-Consistent Node Barriers", 2026,
        https://arxiv.org/abs/2607.14745
    """

    if k < 0:
        raise ValueError(f"length bound {k=} must be non-negative")
    if not prefix:
        raise ValueError(f"{prefix=} must be a non-empty list of nodes")
    for v in prefix:
        if v not in G:
            raise nx.NodeNotFound(f"prefix node {v} not in graph")
    if targets is not None and not targets:
        raise ValueError(f"{targets=} must be None or a non-empty set of nodes")

    succ = _OutEdgeCache(G)
    pred = _InNodeCache(G.pred if G.is_directed() else G.adj)

    if targets is None:  # cycle mode: only the root is a target
        cycle_root, targets = prefix[0], frozenset()
    else:  # path mode: no cycle is closed
        cycle_root = _NO_ROOT

    barrier = defaultdict(int)  # barriers, persistent over the whole run
    stack = list(prefix)  # node stack; do not mutate the caller's list
    plen = len(stack)  # where the reported edge list starts
    forbidden = set(stack)  # node set on stack, forbidden for re-visiting
    edges = [None] * plen  # edges[i] enters stack[i]; prefix part unused
    iters = [iter(succ[stack[-1]])]  # only the last prefix node gets a frame
    shortest_distances = [k + 1]  # per frame: shortest distance to t* found below

    if stack[-1] in targets and plen - 1 <= k:  # the prefix already ends at a target
        shortest_distances[-1] = 0
        yield []

    def cascade(v, sd):
        """Fruitful write ``b[v] = sd``, then restore (EC) by backward BFS."""
        barrier[v] = sd
        queue = deque([(v, sd)])
        while queue:
            w, d = queue.popleft()
            for u in pred[w]:
                if u not in forbidden and barrier[u] > d + 1:
                    barrier[u] = d + 1  # (EC) was violated at u -> w
                    queue.append((u, d + 1))

    while iters:
        h = len(stack) - 1  # depth of the top node v
        for e in iters[-1]:
            w = e[1]
            if barrier[w] + h < k:  # admissible
                if w == cycle_root:  # the root is forbidden: close, do not push
                    yield edges[plen:] + [e]
                    if shortest_distances[-1] > 1:
                        shortest_distances[-1] = 1
                elif w not in forbidden:  # descend: call Search(w)
                    stack.append(w)
                    edges.append(e)
                    forbidden.add(w)
                    iters.append(iter(succ[w]))
                    if w in targets:  # a target: report the path on arrival
                        shortest_distances.append(0)
                        yield edges[plen:]
                    else:
                        shortest_distances.append(k + 1)
                    break
        else:  # return from Search(v)
            v = stack[-1]
            iters.pop()
            sd = shortest_distances.pop()
            if sd <= k:
                cascade(v, sd)  # fruitful; v still forbidden
            else:
                barrier[v] = k - h + 1  # unfruitful raise
            stack.pop()
            edges.pop()
            forbidden.discard(v)
            if shortest_distances and sd + 1 < shortest_distances[-1]:
                shortest_distances[-1] = sd + 1  # propagate distance to caller


# ------------------------------------------------------------- the wrappers


def wrap_paths(G, source, targets, cutoff):
    """What simple_paths.py would call."""
    return bsdfs(G, [source], targets, cutoff)
 
 
def wrap_cycles(G, path, length_bound):
    """What cycles.py would call, for both the node and the edge prefix."""
    head = path[:-1]
    for E in bsdfs(G, path, None, length_bound):
        yield head + [e[0] for e in E]


if __name__ == "__main__":
    """Smoke test for the unified bsdfs(G, source, targets, k).

    Covers the three NetworkX call patterns:
        bsdfs(G,    [s],   None,  k)   cycles through node s        (directed)
        bsdfs(G, [u, v],   None,  k)   cycles through edge [u,v]    (undirected)
        bsdfs(G,    [s], targets, k)   simple edge paths s -> some t in targets

    Compared against: the two original patches, brute force, and (for paths) the
    stock NetworkX implementation.  Multiset comparison, so drops and duplicates
    both surface.
    """

    import itertools
    import random
    from collections import defaultdict, deque

    import networkx as nx
    from networkx.algorithms.cycles import _NeighborhoodCache
    from networkx.algorithms.simple_paths import _all_simple_edge_paths


    # ---------------------------------------------------------------- originals

    def orig_paths(G, source, targets, cutoff):
        assert cutoff >= 0 and targets
        k = cutoff
        get_edges = (lambda v: G.edges(v, keys=True)) if G.is_multigraph() else (lambda v: G.edges(v))
        pred = G.pred if G.is_directed() else G.adj
        b = defaultdict(int)
        nodes, edges, on_path, sd_stack, iters = [], [], set(), [], []

        def fruitful(v, sd):
            b[v] = sd
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
            if v in targets:
                sd_stack.append(0)
                return True
            sd_stack.append(k + 1)
            return False

        if push(source, None):
            yield []
        while nodes:
            h = len(nodes) - 1
            e = next((e for e in iters[-1] if e[1] not in on_path and b[e[1]] + h < k), None)
            if e is not None:
                if push(e[1], e):
                    yield edges[1:]
                continue
            v = nodes[-1]
            iters.pop()
            sd = sd_stack.pop()
            if sd <= k:
                fruitful(v, sd)
            else:
                b[v] = k - h + 1
            nodes.pop()
            edges.pop()
            on_path.discard(v)
            if sd_stack:
                sd_stack[-1] = min(sd_stack[-1], sd + 1)


    def orig_cycles(G, path, length_bound):
        k = length_bound
        t = path[0]
        if G.is_directed():
            succ = _NeighborhoodCache(G)
            pred = _NeighborhoodCache(G.pred)
        else:
            succ = pred = _NeighborhoodCache(G)
        b = defaultdict(int)
        S = list(path)
        on_stack = set(S)
        iters = [iter(succ[S[-1]])]
        sds = [k + 1]

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
                        yield S[:]
                        sds[-1] = 1
                    elif w not in on_stack:
                        S.append(w)
                        on_stack.add(w)
                        iters.append(iter(succ[w]))
                        sds.append(k + 1)
                        break
            else:
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


    # ------------------------------------------------------------- brute force

    def brute_edge_paths(G, source, targets, k):
        out = []
        get_edges = (lambda v: G.edges(v, keys=True)) if G.is_multigraph() else (lambda v: G.edges(v))

        def rec(v, visited, acc):
            if v in targets:
                out.append(list(acc))
            if len(acc) == k:
                return
            for e in get_edges(v):
                w = e[1]
                if w in visited:
                    continue
                visited.add(w)
                acc.append(e)
                rec(w, visited, acc)
                acc.pop()
                visited.discard(w)

        rec(source, {source}, [])
        return out


    def brute_cycles(G, prefix, k):
        t = prefix[0]
        out = []

        def rec(S, visited):
            v = S[-1]
            if len(S) <= k and t in G[v]:
                out.append(list(S))
            if len(S) == k:
                return
            for w in G[v]:
                if w in visited:
                    continue
                visited.add(w)
                S.append(w)
                rec(S, visited)
                S.pop()
                visited.discard(w)

        rec(list(prefix), set(prefix))
        return out


    # ------------------------------------------------------------------ driver

    def norm(seq):
        return sorted(tuple(x) for x in seq)


    def random_graph(cls, n, p, rng):
        G = cls()
        G.add_nodes_from(range(n))
        for u, v in itertools.permutations(range(n), 2):
            if rng.random() < p:
                G.add_edge(u, v)
                if cls in (nx.MultiDiGraph, nx.MultiGraph) and rng.random() < 0.3:
                    G.add_edge(u, v)
        return G


    def main(trials=10_000, n=9, seed=20260820):
        rng = random.Random(seed)
        src_in_tgt = 0
        for trial in range(trials):
            print(trial, end='\r', flush=True)
            cls = [nx.DiGraph, nx.Graph, nx.MultiDiGraph, nx.MultiGraph][trial % 4]
            p = rng.choice([0.2, 0.35, 0.5])
            G = random_graph(cls, n, p, rng)
            if G.is_directed() and rng.random() < 0.3:
                G.add_edge(rng.randrange(n), rng.randrange(n))   # self-loop

            k = rng.randrange(0, 6)
            s = rng.randrange(n)
            targets = set(rng.sample(range(n), rng.randrange(1, 3)))
            if rng.random() < 0.3:
                targets.add(s)                                   # exercise source in targets
            if s in targets:
                src_in_tgt += 1

            got = norm(wrap_paths(G, s, targets, k))
            assert got == norm(orig_paths(G, s, targets, k)), (trial, "paths vs original", list(G.edges()), s, targets, k)
            assert got == norm(brute_edge_paths(G, s, targets, k)), (trial, "paths vs brute", list(G.edges()), s, targets, k)
            assert got == norm(_all_simple_edge_paths(G, s, targets, k)), (trial, "paths vs networkx", list(G.edges()), s, targets, k)

            if cls is nx.DiGraph:                                # pattern 1: cycles through a node
                t = rng.randrange(n)
                got = norm(wrap_cycles(G, [t], k))
                assert got == norm(orig_cycles(G, [t], k)), (trial, "node cycles vs original", list(G.edges()), t, k)
                assert got == norm(brute_cycles(G, [t], k)), (trial, "node cycles vs brute", list(G.edges()), t, k)

            if cls is nx.Graph:                                  # pattern 2: cycles through an edge
                es = list(G.edges())
                if es:
                    u, v = es[rng.randrange(len(es))]
                    G.remove_edge(u, v)                          # as _undirected_cycle_search does
                    got = norm(wrap_cycles(G, [u, v], k))
                    assert got == norm(orig_cycles(G, [u, v], k)), (trial, "edge cycles vs original", list(G.edges()), (u, v), k)
                    assert got == norm(brute_cycles(G, [u, v], k)), (trial, "edge cycles vs brute", list(G.edges()), (u, v), k)

        print(f"OK: {trials:10,} trials, n={n:2}; {src_in_tgt:10,} of them with source in targets.")

    main(trials=10_000, n=10)
    main(trials=100_000, n=9)
    main(trials=1_000_000, n=8)
    main(trials=10_000_000, n=7)

# reference output
# OK:     10,000 trials, n=10;      4,035 of them with source in targets.
# OK:    100,000 trials, n= 9;     41,656 of them with source in targets.
# OK:  1,000,000 trials, n= 8;    431,601 of them with source in targets.
# OK: 10,000,000 trials, n= 7;  4,499,245 of them with source in targets.
