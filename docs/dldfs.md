# Depth-Limited Depth-First Search

The most simple way of enumerating s-t-paths which can have superpolynomial delay.
The delay heavily depends on the adjacency list orders of the graph representation.

See [bsdfs](bsdfs.md) for an efficient enumeration algorithm and maybe [bbdfs](bbdfs.md) if only one path is asked for.

```python
def dldfs(G, s, t, k):
    """Generate all simple s-t paths with at most k edges by plain depth-limited DFS (no pruning)."""
    S = [] # current search path (stack)

    def search(v):
        S.append(v)
        h = len(S) - 1  # number of edges in S
        for w in G.successors(v):
            if h < k: # admissible length?
                if w == t:
                    yield S + [t]  # output
                elif w not in S:
                    yield from search(w)
        S.pop()

    yield from search(s)
```

## Clique-Trap Example

```python
import networkx as nx
from bsdfs import bsdfs

c = 100
G = nx.complete_graph(c, create_using=nx.DiGraph)
s = 0
t = c
k = c - 1
G.add_edge(s, t) #  # appended last to adj. list of s

P0 = next(bsdfs(G, s, t, k), None) # path [0, 100] is found within about 10⁶ steps
print("bsdfs:", P0)

P0 = next(dldfs(G, s, t, k), None) # explores ⌊e·99!⌋ ≈ 2.5·10¹⁵⁶ fruitless paths first
print("dldfs:", P0) # will we be still alive here?
```


## Triangular-Snake Example

The directed triangular-snake is a sparse, acyclic, and planar digraph.

The `dldfs` algorithm still takes exponential time for finding the first path.

```python
import networkx as nx
from bsdfs import bsdfs

d = 40
"""Directed triangular snake: path 0→1→…→2d plus chords 2i→2i+2; the two-edge leg comes first in adjacency order."""
G = nx.path_graph(2 * d + 1, create_using=nx.DiGraph)
G.add_edges_from((2 * i, 2 * i + 2) for i in range(d))
s = min(G.nodes)
t = max(G.nodes)
k = d # the smallest fruitful value, yielding path [0, 2, 4, ..., 2d]

P0 = next(bsdfs(G, s, t, k), None) # found within ⌊5d²/4⌋ + 5d + 1 < (k+1)(n+m) = (d+1)(5d+1) steps.
print("bsdfs:", P0) # found within 2201 steps (bound (k+1)(n+m) = 8241)

P0 = next(dldfs(G, s, t, k), None) # found within F(d+6) + d − 7 ≈ φᵈ (φ ≈ 1.618) steps.
print("dldfs:", P0) # found after F(46) + 33 ≈ 1.8e9 steps
```

F(·) is the [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_sequence)
