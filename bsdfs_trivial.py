"""BS-DFS trivial scheme"""

def bsdfs(G, s, t, k):
    """hard coded: all barriers remain at 0; resulting in a length limited dfs"""
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
