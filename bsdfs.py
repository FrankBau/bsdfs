"""BS-DFS -- the tight scheme, the reference implementation of the paper.

    bsdfs(G, s, t, k)

enumerates every simple path from s to t in the directed graph G that uses
at most k edges, in lexicographic order, with O(k(n+m)) delay per output.
Called as bsdfs(G, s, s, k) it enumerates the simple cycles through s.

Run this file directly for a self-check against NetworkX:

    python bsdfs.py

The systematic, claim-by-claim validation of the delay bounds lives in
validate_delay_bounds_v2/; the measurements reported in the paper are
produced by delay_bounds.py, missed_paths.py, steps.py and runtime.py.
"""

from collections import deque


def bsdfs(G, s, t, k):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
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


if __name__ == "__main__":
    # the algorithm above needs only the standard library; the self-check
    # additionally uses NetworkX to draw the random instances and bsdfs_trival.py
    # as known-good implementation

    import random
    import networkx as nx
    import bsdfs_trivial

    RUNS = 10_000

    for n in range(2, 8):
        found = 0
        for run in range(RUNS):
            rng = random.Random(42 + run)
            m = rng.randint(0, n * (n - 1))
            k = max(rng.getrandbits(n).bit_count(), 1)  # binomial distribution
            G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            s, t = rng.sample(range(n), 2)

            # both enumerators scan successors in adjacency order, so the
            # outputs must agree element by element, not just as sets
            got = list(map(list, bsdfs(G, s, t, k)))
            expected = list(map(list, bsdfs_trivial.bsdfs(G, s, t, k)))
            assert got == expected, (
                f"{s=} {t=} {k=} edges={sorted(G.edges)}"
                f"\n  got      {got}\n  expected {expected}"
            )
            found += len(got)

        print(f"n={n}  {RUNS:,} random digraphs ok, {found:,} paths")
