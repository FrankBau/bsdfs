from math import inf


def all_simple_paths(G, s, t, k=inf):
    """minimalistic depth-limited DFS for simple path enumeration"""
    path = []

    def dfs(v):
        if v == t:
            yield path + [t]
        elif len(path) < k:
            path.append(v)
            for w in G.successors(v):
                if w not in path:
                    yield from dfs(w)
            path.pop()

    yield from dfs(s)


if __name__ == "__main__":
    # this file is the ground truth for the other self-checks, so it is the
    # only one checked against NetworkX rather than against itself
    import random

    import networkx as nx

    RUNS = 10_000

    for n in range(2, 8):
        found = 0
        for run in range(RUNS):
            rng = random.Random(42 + run)
            m = rng.randint(0, n * (n - 1))
            k = max(rng.getrandbits(n).bit_count(), 1)  # binomial distribution
            G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            s, t = rng.sample(range(n), 2)

            got = sorted(map(list, all_simple_paths(G, s, t, k)))
            expected = sorted(map(list, nx.all_simple_paths(G, s, t, k)))
            assert got == expected, (
                f"{s=} {t=} {k=} edges={sorted(G.edges)}"
                f"\n  got      {got}\n  expected {expected}"
            )
            found += len(got)

        print(f"n={n}  {RUNS:,} random digraphs ok, {found:,} paths")
