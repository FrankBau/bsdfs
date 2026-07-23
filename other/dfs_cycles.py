import networkx as nx

def dfs_cycle_search(G, s, k=float("inf")):
    stack = []

    def dfs(v):
        if len(stack) >= k:
            return
        stack.append(v)
        for w in G.successors(v):
            if w == s:
                yield stack.copy()
            elif w not in stack:
                yield from dfs(w)
        stack.pop()

    yield from dfs(s)

import math
import random

def main(runs=10_000):
    for run in range(runs):
        print(run, end='\r')
        rng = random.Random(42 + run)
        n = random.randint(2, 10)
        m = int(n * math.exp(rng.uniform(0, math.log(n-1))))
        G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
        s = rng.choice(list(G.nodes))
        k = rng.randrange(n)
        cycles1 = list(dfs_cycle_search(G, s, k))
        cycles2 = list(nx.algorithms.cycles._bounded_cycle_search(G, [s], length_bound=k))
        assert cycles1 == cycles2, "dfs_cycle_search: test failed"
    print("dfs_cycle_search: test passed.")


if __name__ == "__main__":
    main()