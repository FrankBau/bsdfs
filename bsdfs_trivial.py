"""BS-DFS trivial scheme (depth-limited DFS)"""

import networkx as nx
    
def bsdfs(G, s, t, k):
    """hard coded: all barriers are implicit and remain at 0.
       This means: no prevention of repeated fruitless searches;
       expect unbounded output delays, use for small n,m,k only.
    """
    S = []

    def search(v):
        S.append(v)
        h = len(S) - 1
        for w in G.successors(v):
            if h < k:
                if w == t:
                    yield S + [t]
                elif w not in S:
                    yield from search(w)
        S.pop()

    yield from search(s)


def cycle_reference(G, s, k):
    """s-cycles via node splitting and NetworkX all_simple_paths"""
    sin = ("in", s)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    H.add_node(sin)
    for u, v in G.edges:
        H.add_edge(u, sin if v == s else v)
    return [p[:-1] + [s] for p in nx.all_simple_paths(H, s, sin, k)]


if __name__ == "__main__":
    # The trivial scheme code is obvious and accessible to human code review.
    # In addition, it is cross-checked against NetworkX code, mainly to ensure compatibility:

    import random
    # internal routine for finding all simple s-cycles (fixed s)
    from networkx.algorithms.cycles import _bounded_cycle_search

    RUNS = 100_000

    for n in range(2, 9):
        paths = 0
        cycles = 0
        for run in range(RUNS):
            rng = random.Random(42 + run)
            m = rng.randint(0, n * (n - 1))
            k = rng.randint(0, n)
            # gnm_random_graph inserts edges in random order
            G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            # s==t deliberately allowed to support s-cycle search
            s = rng.choice(list(G.nodes))
            t = rng.choice(list(G.nodes))

            # no sorting needed, both produce the same output order
            # order is not guaranteed by NetworkX API, but observed.
            got = list(bsdfs(G, s, t, k))

            if s == t:
                # NetworkX drops the closing edge, bsdfs does not.
                expected = [c + [c[0]] for c in _bounded_cycle_search(G, [s], k)]
                cycles += len(expected)
            else:
                expected = list(nx.all_simple_paths(G, s, t, k))
                paths += len(expected)
            assert got == expected, (
                f"{s=} {t=} {k=} edges={sorted(G.edges)}"
                f"\n  got      {got}\n  expected {expected}"
            )

        print(f"n={n}  {RUNS:,} random digraphs ok, checked {paths:10,} paths and {cycles:10,} cycles.")
