A short intro to the [BS-DFS](bsdfs.md) algorithm from our paper: 

> Frank Bauernöppel, Jörg-Rüdiger Sack; "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with O(k(n+m)) Delay Using Edge-Consistent Node Barriers"; [https://doi.org/10.48550/arXiv.2607.14745](https://doi.org/10.48550/arXiv.2607.14745); submitted to [The Journal of Graph Algorithms and Applications (JGAA)](https://jgaa.info/)


# Basics

A call `bsdfs(G, s, t, k)` enumerates all simple paths in graph `G` starting at vertex `s` and terminating at vertex `t` having at most `k` edges. 
For the special case `s==t`, simple cycles through vertex `s` will be enumerated. This includes cycles of length 2 which are excluded in some literature.

For graph terminology, we follow the book [Algorithms, 4th Edition by Robert Sedgewick and Kevin Wayne](https://algs4.cs.princeton.edu/).
For coding, we use [Python](https://www.python.org/) and the [NetworkX](https://networkx.org/) graph library.
For simplicity, we consider directed simple graphs (no self-loops, no parallel edges) here,
but the algorithm can be adapted to multigraphs and undirected graphs

# Example

With the [BS-DFS implementation](bsdfs.md), the code
```python
import networkx as nx
G = nx.DiGraph()
G.add_edges_from([(0,1), (0,2), (1, 2), (2,3), (2,4), (3, 4)])
s = 0
t = 4
k = 3
for path in bsdfs(G, s, t, k): 
    print(path)
```

prints 3 paths
```python
[0, 1, 2, 4]
[0, 2, 3, 4]
[0, 2, 4]
```
The output order depends on the adjacency-list order.

Path `[0, 1, 2, 3, 4]` is too long to be reported for `k = 3`.
If we set `k = 4`, it will be reported too.


# Efficiency - The Delay Bounds

The number of paths or cycles can grow exponentially in the graph size.
Luckily, `bsdfs` is implemented as a generator function and can be stopped whenever "enough" paths are generated.
But: how long to wait?

Let's call the start, each path generation (output), and termination the *events* of the algorithm.
Then, the algorithm guarantees that the delay (waiting time) between two consecutive events is bounded by O(k(n+m)).
Here, n=|V| the number of vertices in G, and m=|E| the number of edges in G.
So, for any fixed k, the delay is linear in the graph size.

As shown in our paper, the Big-O formulation does not hide huge constants.
With a suitable definition of elementary steps (node visits, edge scans, barrier writes, output),
the following is proven:

- the worst-case delay between two consecutive events is at most 3(k+1)(n+m) steps, and
- for every p≥1, the first p events are produced within 2p(k+1)(n+m) steps, i.e. the amortized delay is at most 2(k+1)(n+m) steps per event.


# Motivation

If only the first s-t path P0 of length ≤ k is wanted (or only its existence), a depth-limited depth-first search suffices. Its problem is that it searches the same vertices over and over again. For example, let `s` lie in a large clique from which `t` is reachable, but only via a path longer than `k`. Then all paths of length ≤ k inside the clique are explored fruitlessly.

 Hence each vertex `v` gets a barrier `b[v]`, a lower bound on the remaining distance from `v` to `t`. Let `h` denote the length of the current search path from `s` to `v`. If `search(v)` finds no path to `t`, the barrier is raised to `b[v] = k - h + 1`. The search descends from `v` into a successor `w` only if `b[w] + h < k`. This suppresses repeated searches of `v` at the same or a greater depth. In this phase barriers only increase. Each vertex is entered at most k+1 times, so the total work is bounded by (k+1)(n+m).

When more paths than P0 are wanted, we want to reuse the barriers already established. But a formerly fruitless search may become fruitful once vertices are popped from the search path, because they no longer block. So barriers raised with respect to an earlier search path may need correction. This is done by the `fruitful` procedure, and it must be done carefully. Resetting all barriers to 0 keeps the algorithm correct but throws away the work already spent, and the delay bounds no longer hold. Instead, `b[v]` is set to `sd`, the length of the shortest path to `t` found from `v`.
The barriers of its direct and indirect predecessors are then repaired, walking backwards over in-edges, only as far as needed to restore *edge-consistency*, a property defined and discussed in the paper (informally: a vertex's barrier may exceed that of its successors by at most one).
