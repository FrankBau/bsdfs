# Bounded Scope Depth-First Search

Python translation of the pseudocode in our paper.
This algorithm extends [dldfs](dldfs.md) and [bbdfs](bbdfs.md), avoiding their pitfalls.

```python
from collections import deque


def bsdfs(G, s, t, k):
    """Enumerate all simple s-t paths of length at most k in G."""

    b = {x: 0 for x in G.nodes}                 # init barriers
    S = []                                      # current search path


    def cascade(v):
        """Distance propagation (reverse BFS)."""

        Q = deque([v])                          # initialize worklist with node v
        while Q:
            q = Q.popleft()                     # dequeue next node q (initially v)
            d = b[q]                            # get its barrier (distance) value
            for p in G.predecessors(q):         # scan each predecessor of q
                if b[p] > d + 1 and p not in S: # edge-consistency broken at p?
                    b[p] = d + 1                # repair it, drop barrier at p
                    Q.append(p)                 # propagate cascade further


    def search(v):
        """Recursive bounded-scope DFS."""

        S.append(v)                             # search call entry
        h = len(S) - 1                          # number of edges in S

        # sd = shortest v→t distance avoiding S, if ≤ k−h; else k+1
        sd = k + 1                              # sentinel (∞): no path to t found yet
        for w in G.successors(v):               # scan each successor of v
            if b[w] + h < k:                    # is w admissible? (the *scope*)
                if w == t:                      # target reached?
                    yield S + [t]               # output, report path
                    sd = 1                      # edge (v, t)
                elif w not in S:                # keep simplicity of S
                    d = yield from search(w)    # descend into w
                    sd = min(sd, d + 1)         # shortest distance wins

        if sd <= k:                             # any path to t found?
            b[v] = sd                           # fruitful, update barrier
            cascade(v)                          # repair edge-consistency
        else:
            b[v] = k + 1 - h                    # fruitless, raise barrier

        S.pop()                                 # search call exit
        return sd                               # return shortest distance found

    yield from search(s)
```

Note: Several optimizations are useful but not shown for clarity.


# Directed Triangular Snake Graph Example

Path 0→1→…→2d plus shortcuts 2i→2i+2; the two-edge leg comes first in adjacency order.

```python
import networkx as nx
import time

d = 32 # number of triangles

G = nx.path_graph(2 * d + 1, create_using=nx.DiGraph)   # the long path
G.add_edges_from((2 * i, 2 * i + 2) for i in range(d))  # shortcuts skipping odd node
assert G.number_of_nodes() == 2 * d + 1
assert G.number_of_edges() == 3 * d
s = min(G.nodes)
t = max(G.nodes)
k = d

tick = time.perf_counter()
P0 = next(bsdfs(G, s, t, k), None)
tock = time.perf_counter()
print(f"bsdfs {tock-tick:10.8f}s:", P0)

tick = time.perf_counter()
P0 = next(dldfs(G, s, t, k), None)
tock = time.perf_counter()
print(f"dldfs {tock-tick:10.8f}s:", P0)
```

Both outputs are correct.
Bounded Scope Depth-First Search took 0.00047s, plain depth-limited DFS needs more than 5s (i7-14700K, 64GB, Win11).
