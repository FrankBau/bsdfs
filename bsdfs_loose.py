"""BS-DFS loose scheme"""

from collections import deque


def bsdfs(G, s, t, k):
    """loose scheme, barriers are reset to 0"""
    b = {x: 0 for x in G.nodes}
    S = []

    def reset(v):
        b[v] = 0
        queue = deque([v])
        while queue:
            q = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] != 0:
                    b[p] = 0
                    queue.append(p)

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
            reset(v)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)

if __name__ == "__main__":
    import random

    import networkx as nx

    import dfs

    RUNS = 10_000

    for n in range(2, 8):
        found = 0
        for run in range(RUNS):
            rng = random.Random(42 + run)
            m = rng.randint(0, n * (n - 1))
            k = max(rng.getrandbits(n).bit_count(), 1)  # binomial distribution
            G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            s, t = rng.sample(range(n), 2)

            got = list(map(list, bsdfs(G, s, t, k)))
            expected = list(map(list, dfs.all_simple_paths(G, s, t, k)))
            assert got == expected, (
                f"{s=} {t=} {k=} edges={sorted(G.edges)}"
                f"\n  got      {got}\n  expected {expected}"
            )
            found += len(got)

        print(f"n={n}  {RUNS:,} random digraphs ok, {found:,} paths")
