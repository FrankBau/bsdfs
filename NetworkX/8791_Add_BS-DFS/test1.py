import random
import timeit

import networkx as nx
# from networkx.algorithms.traversal import bsdfs   # not yet
from bounded_dfs import bsdfs

G = nx.complete_graph(4)
paths = list(bsdfs(G, 0, 1, 3))
print(paths)

rng = random.Random(42)
runs = 1000
test_cases = []
for run in range(runs):
    H = nx.barabasi_albert_graph(60, 2, seed=rng)
    G = nx.DiGraph(H)
    s, t = rng.sample(list(G.nodes), 2)
    k = rng.randint(3, 10)
    test_cases.append((G, s, t, k))


def count_paths():
    return sum(len(list(bsdfs(G, s, t, k))) for G, s, t, k in test_cases)


results = []
elapsed = timeit.repeat(lambda: results.append(count_paths()), repeat=9, number=1)
path_count = results[0]
print(f"{min(elapsed)=:10.6f} s: {path_count=:4} found in {runs=:4} graphs.")
if path_count != 1844378: raise AssertionError("wrong path_count")

# baseline:
# commit c598b0c326b24a8485bd0359732885eeb9138eca (HEAD -> issue-8737, origin/issue-8737)
# Author: Frank Bauernoeppel <frank@bauernoeppel.de>
# Date:   Mon Oct 5 07:43:59 2026 +0200
#     applied minor local optimizations for speed
#
# > set PYTHONPATH=networkx
# python -OO test1.py
# [[0, 1], [0, 2, 1], [0, 2, 3, 1], [0, 3, 1], [0, 3, 2, 1]]
# min(elapsed)=  5.334755 s: path_count=1844378 found in runs=1000 graphs.

# also use: pytest networkx\networkx -k bsdfs -v