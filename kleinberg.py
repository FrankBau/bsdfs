import networkx as nx
import random
import time
from bsdfs import bsdfs
from bcdfs import bcdfs


# https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.navigable_small_world_graph.html
n = 50
p = 1   # node has a directed edge to every other node within lattice distance — these are its local contacts
q = 1   # construct q directed edges from to other nodes (the long-range contacts) using independent random trials;
r = 2   # u has endpoint v with probability proportional to grid_dist(u, v)^{-r}
dim = 2
seed = 42
random.seed(seed)

G = nx.navigable_small_world_graph(n, p, q, r, dim, seed)
assert isinstance(G, nx.DiGraph)
# assert nx.number_of_selfloops(G) == 0 # breaks

for run in range(1000):
    s, t = random.sample(list(G.nodes), 2)
    for k in range(6, 16, 2):
        tick = time.perf_counter_ns()
        paths = list(bsdfs(G, s, t, k))
        tock = time.perf_counter_ns()
        us = (tock - tick) / 1000 
        iv = len(paths) + 1
        
        tick = time.perf_counter_ns()
        paths2 = list(bcdfs(G, s, t, k))
        tock = time.perf_counter_ns()
        us2 = (tock - tick) / 1000
        iv2 = len(paths2) + 1
        
        print(f"{iv=:8} {iv2=:8} {us/iv=:10.2f} {us2/iv2=:10.2f}", end=" | ")
    print("")        