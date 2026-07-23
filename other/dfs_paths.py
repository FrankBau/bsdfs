import networkx as nx
import random


def dfs_all_simple_paths(G, s, t, k=float("inf")):
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


import math
import random

def main(runs=10_000):
    for run in range(runs):
        print(run, end='\r')
        rng = random.Random(42 + run)
        n = rng.randint(2, 10)
        m = int(n * math.exp(rng.uniform(0, math.log(n-1))))
        G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
        s = rng.choice(list(G.nodes))
        t = rng.choice(list(G.nodes))
        k = rng.randrange(n)
        paths1 = list(dfs_all_simple_paths(G, s, t, k))
        paths2 = list(nx.all_simple_paths(G, s, t, cutoff=k))
        assert paths1 == paths2, "dfs_all_simple_paths: test failed"
    print("dfs_all_simple_paths: test passed.")


if __name__ == "__main__":
    main()
