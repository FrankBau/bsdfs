# Bounded Scope Depth-First Search

Python translation of the pseudocode in our paper.
This algorithm extends [dldfs](dldfs.md) and [bbdfs](bbdfs.md), avoiding their pitfalls.

```python
from collections import deque

def bsdfs(G, s, t, k):
    """Enumerate all simple s-t paths of length at most k in G."""

    b = {x: 0 for x in G.nodes}                 # init barriers
    S = []                                      # current search path


    def cascade(v, sd):
        """Distance propagation (reverse BFS)."""

        Q = deque([(v, sd)])                    # initialize worklist
        while Q:
            q, d = Q.popleft()                  # dequeue next item
            for p in G.predecessors(q):
                                                # predecessor scan
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1                # drop barrier
                    Q.append((p, d + 1))        # propagate further


    def search(v):
        """Recursive bounded-scope DFS from node v."""

        S.append(v)                             # entry
        h = len(S) - 1                          # number of edges in S

        # shortest v-t distance w.r.t. current path S
        sd = k + 1                              # "infinity" (no path found yet)
        for w in G.successors(v):
                                                # successor scan
            if b[w] + h < k:                    # is w admissible?
                if w == t:
                    yield S + [t]               # output, report path
                    sd = 1                      # edge (v,t)
                elif w not in S:                # keep simplicity of S
                    d = yield from search(w)    # descend into w
                    sd = min(sd, d + 1)         # shortest distance wins

        if sd <= k:                             # any path to t found?
            b[v] = sd                           # fruitful, update barrier
            cascade(v, sd)                      # repair edge-consistency
        else:
            b[v] = k + 1 - h                    # fruitless, raise barrier

        S.pop()
        return sd

    yield from search(s)
```

Note: Several optimizations are useful but not shown for clarity.



# Directed triangular snake graph example

Path 0→1→…→2d plus shortcuts 2i→2i+2; the two-edge leg comes first in adjacency order.

```python
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

Both outputs are correct. Bounded-Scope DFS took 0.00047s, plain depth-limited DFS needs more than 5s (i7-14700K Win11).
