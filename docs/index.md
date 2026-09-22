**BS-DFS** (Bounded Scope Depth-First Search) - a graph algorithm for enumerating all length bounded simple paths or cycles.

This is a short intro to the BS-DFS algorithm from our paper: 

> Frank Bauernöppel, Jörg-Rüdiger Sack; "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with O(k(n+m)) Delay Using Edge-Consistent Node Barriers"; https://doi.org/10.48550/arXiv.2607.14745; submitted to [The Journal of Graph Algorithms and Applications (JGAA)](https://jgaa.info/)


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
But: how long to wait for that?

Let's call the algo start, each path generation (output), and algo termination an event.
Then, the algorithm guarantees that the delay (waiting time) between two consecutive events is bounded by O(k(n+m)).
Here, n=|V| the number of vertices in G, and m=|E| the number of edges in G.
So, for any fixed k, the delay is linear in the graph size.

As shown in our paper, the Big-O formulation does not hide huge constants.
With a suitable definition of elementary steps (node visits, edge scans, barrier writes, output),
the following is proven:

- the worst-case delay to the next event (output, termination) is at most 3(k+1)(n+m) steps, and
- for every p≥1, the first p events are produced within 2p(k+1)(n+m) steps, i.e. the amortized delay is at most 2(k+1)(n+m) steps per event.
