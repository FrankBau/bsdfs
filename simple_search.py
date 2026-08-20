"""
The `simple_search` of

    Finding All Bounded-Length Simple Cycles in a Directed Graph -- Revisited
    Frank Bauernöppel, Jörg-Rüdiger Sack
    https://arxiv.org/abs/2512.08392

generalized from s-cycles to s-cycles and st-paths depending on s==t.

It is included here as a faster known-good reference implementation than a plain depth-limited DFS (bsdfs_trivial), especially for larger graphs and k.
"""


from collections import deque


def simple_search(G, s, t, k):
    """Enumerate all simple st-paths in G of bounded length k; for s == t, all simple cycles through s."""

    def reach(blocked, successors, budget):
        assert budget >= 0
        reached = {t}
        queue = deque()
        queue.append((t, 0))
        while queue:
            (u, d) = queue.popleft()
            if d >= budget:
                break
            for v in G.predecessors(u):
                if v not in blocked and v not in reached:
                    reached.add(v)
                    queue.append((v, d + 1))

        fruitful = [w for w in successors if w in reached]     # keeps the internal order of the successors
        return fruitful

    def search(path, v, budget):
        if budget > 0:
            path.append(v)
            fruitful = reach(path, G.successors(v), budget - 1)
            for w in fruitful:
                if w == t:
                    yield path + [t] # output path
                else:
                    yield from search(path, w, budget - 1)
            path.pop()

    path = list()
    yield from search(path, s, k)


if __name__ == "__main__":
    # The simple_search code is accessible to human code review.
    # In addition, it is cross-checked against bsdfs_trivial (a plain depth-limited DFS):

    import random
    import networkx as nx
    import bsdfs_trivial

    RUNS = 10_000

    for n in range(2, 8):
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
            got = list(simple_search(G, s, t, k))

            expected = list(bsdfs_trivial.bsdfs(G, s, t, k))
            assert got == expected, (
                f"{s=} {t=} {k=} edges={sorted(G.edges)}"
                f"\n  got      {got}\n  expected {expected}"
            )
            if s==t:
                cycles += len(got)
            else:
                paths += len(got)

        print(f"n={n}  {RUNS:,} random digraphs ok, checked {paths:10,} paths and {cycles:10,} cycles.")