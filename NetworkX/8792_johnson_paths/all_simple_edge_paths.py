from bsdfs_all_simple_edge_paths import bsdfs_all_simple_edge_paths
from johnson_paths import johnson_all_simple_edge_paths

import networkx as nx


def all_simple_edge_paths(G, source, target, cutoff=None):
    """Generate lists of edges for all simple paths in G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths with length <= `cutoff` are returned.
        Note that the length of an edge path is the number of edges.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output.
       For multigraphs, the list of edges have elements of the form `(u,v,k)`.
       Where `k` corresponds to the edge key.

    Examples
    --------

    Print the simple path edges of a Graph::

        >>> g = nx.Graph([(1, 2), (2, 4), (1, 3), (3, 4)])
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 4)):
        ...     print(path)
        [(1, 2), (2, 4)]
        [(1, 3), (3, 4)]

    Print the simple path edges of a MultiGraph. Returned edges come with
    their associated keys::

        >>> mg = nx.MultiGraph()
        >>> mg.add_edge(1, 2, key="k0")
        'k0'
        >>> mg.add_edge(1, 2, key="k1")
        'k1'
        >>> mg.add_edge(2, 3, key="k0")
        'k0'
        >>> for path in sorted(nx.all_simple_edge_paths(mg, 1, 3)):
        ...     print(path)
        [(1, 2, 'k0'), (2, 3, 'k0')]
        [(1, 2, 'k1'), (2, 3, 'k0')]

    When ``source`` is one of the targets, the empty path starting and ending at
    ``source`` without traversing any edge is considered a valid simple edge path
    and is included in the results:

        >>> G = nx.Graph()
        >>> G.add_node(0)
        >>> paths = list(nx.all_simple_edge_paths(G, 0, 0))
        >>> for path in paths:
        ...     print(path)
        []
        >>> len(paths)
        1

    You can use the `cutoff` parameter to only generate paths that are
    shorter than a certain length:

        >>> g = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (1, 5)])
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 5)):
        ...     print(path)
        [(1, 2), (2, 3), (3, 4), (4, 5)]
        [(1, 4), (4, 5)]
        [(1, 5)]
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 5, cutoff=1)):
        ...     print(path)
        [(1, 5)]
        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 5, cutoff=2)):
        ...     print(path)
        [(1, 4), (4, 5)]
        [(1, 5)]

    Notes
    -----
    This algorithm enumerates paths with polynomial delay, i.e., the time
    between two consecutive output paths (and before the first and after
    the last) is bounded:  $O(V+E)$ per path if ``cutoff`` is ``None``,
    using a Johnson-style blocked depth-first search [1]_, and
    $O(k \\cdot (V+E))$ per path for ``cutoff=k``, using barrier-based
    search [2]_.  The *number* of simple paths can nevertheless be very
    large, e.g. $\\Theta((n-2)!)$ between two fixed nodes of the complete
    graph of order $n$, so consume the generator lazily rather than
    materializing the full list.

    References
    ----------
    .. [1] D. B. Johnson. "Finding all the elementary circuits of a directed graph." 
           SIAM J. Comput.,1213 4(1):77-84, 1975. doi:10.1137/0204007.
    .. [2] Frank Bauernöppel, Jörg-Rüdiger Sack.
           "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs
            with O(k(n+m)) Delay Using Edge-Consistent Node Barriers"
            https://arxiv.org/abs/2607.14745

    See Also
    --------
    all_shortest_paths, shortest_path, all_simple_paths

    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError as err:
            raise nx.NodeNotFound(f"target node {target} not in graph") from err

    if cutoff is None and targets:
        yield from johnson_all_simple_edge_paths(G, source, targets)
    elif cutoff >= 0 and targets:
        yield from bsdfs_all_simple_edge_paths(G, source, targets, cutoff)


def monkey_patching_pytest():
    orig = nx.all_simple_edge_paths
    nx.all_simple_edge_paths = all_simple_edge_paths

    import pytest
    pytest.main(["--doctest-modules", "--pyargs", "networkx", "-v"])
    # expected:  7720 passed, 87 skipped, 1 xfailed, 11 warnings in 71.54s (0:01:11)
    assert nx.all_simple_edge_paths is all_simple_edge_paths # not restored mid-way
    # (the warnings are unrelated)

    nx.all_simple_edge_paths = orig
    assert nx.all_simple_edge_paths is not all_simple_edge_paths
    # now restored


def circulant_test():
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
        k = None

        repeats = 3
        while True:
            dt1 = dt2 = 0
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                paths1 = list(islice(all_simple_edge_paths(G, s, [t]), limit))
                dt1 += time.perf_counter_ns() - t0
                t0 = time.perf_counter_ns()
                paths2 = list(islice(nx.all_simple_edge_paths(G, s, [t]), limit))
                dt2 += time.perf_counter_ns() - t0
            if dt1 + dt2 > 1_000_000:
                break
            else:
                repeats = 2*repeats + 1 # stay odd
        dt1 //= repeats
        dt2 //= repeats
        assert paths1 == paths2 # likely
        print(f"{run=:6}; {n=:4}; {s=:4}; {t=:4}; {k=}; {dt1=:12}; {dt2=:12}; {dt1/dt2=:10.4f}; {len(paths1)=:10}")
        xs.append(dt1/1e9)  # seconds
        ys.append(dt2/1e9)  # seconds
        cs.append(n)

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
    fig.colorbar(sc, label=f"n")
    fig.savefig("circulant_family_paths.png")
    

import time


def enhanced_clique_test():
    for n in range(100, 1000, 100):
#    for n in range(1, 10_000, 1_000):
#    for n in range(1, 20):
        G = nx.complete_graph(n)
        s = 0
        t = n
        G.add_edge(s, t)
        tick = time.perf_counter()
        paths = list(all_simple_edge_paths(G, s, t))
        # paths = list(nx.all_simple_edge_paths(G, s, t))
        # paths = list(all_simple_edge_paths(G, s, t, cutoff=10))
        tock = time.perf_counter()
        # assert paths == [[s, t]]
        print(f"{n} took {tock-tick:10.4f} seconds")


if __name__ == "__main__":
    # monkey_patching_pytest()
    # circulant_test()
    enhanced_clique_test()
    