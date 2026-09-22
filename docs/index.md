A short intro to the [BS-DFS](bsdfs.md) algorithm from our paper: 

> Frank Bauernöppel, Jörg-Rüdiger Sack; "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with O(k(n+m)) Delay Using Edge-Consistent Node Barriers"; [https://doi.org/10.48550/arXiv.2607.14745](https://doi.org/10.48550/arXiv.2607.14745); submitted to [The Journal of Graph Algorithms and Applications (JGAA)](https://jgaa.info/)


# Basics

A call `bsdfs(G, s, t, k)` enumerates all simple paths in graph `G` starting at vertex `s` and terminating at vertex `t` having at most `k` edges. 
For the special case `s==t`, simple cycles through vertex `s` will be enumerated. This includes cycles of length 2 which are excluded in some literature.

For graph terminology, we follow the book [Algorithms, 4th Edition by Robert Sedgewick and Kevin Wayne](https://algs4.cs.princeton.edu/).
For coding, we use [Python](https://www.python.org/) and the [NetworkX](https://networkx.org/) graph library.
For simplicity, we consider directed simple graphs (no self-loops, no parallel edges) here,
but the algorithm can be adapted to multigraphs and undirected graphs.

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

Path `[0, 1, 2, 3, 4]` is too long to be reported for `k = 3`.
If we set `k = 4`, it will be reported as well.
The output order depends on the adjacency-list order of the internal graph representation.


# Efficiency - The Delay Bounds

The number of length-bounded paths or cycles can grow exponentially in the graph size.
Luckily, `bsdfs` is implemented as a generator function and can be stopped whenever "enough" paths are generated.
But: how long to wait for a (next) output or termination?

Let's call the start, each path generation (output), and the termination the *events* of a call `bsdfs(G, s, t, k)`.
Then, the delay (waiting time) between two consecutive events is bounded by O(k(n+m)).
Here, n=|V| is the number of vertices and m=|E| is the number of edges in G.
So, for any fixed k, the delay is linear in the graph size.

As shown in our paper, the Big-O formulation does not hide huge constants.
With a suitable definition of elementary steps (node visits, edge scans, barrier writes, output),
the following is proven:

- the worst-case delay between two consecutive events is at most 3(k+1)(n+m) steps, and
- for every p≥1, the first p events are produced within 2p(k+1)(n+m) steps,
  i.e. the amortized delay is at most 2(k+1)(n+m) steps per event.


# Motivation

If only the first s-t path P0 of length ≤ k is wanted (or only its existence),
a depth-limited depth-first search ([dldfs](dldfs.md)) will eventually find it.
The problem is that this search recurses over and over again when a vertex is re-visited.
For example, let `s` lie in a large clique from which `t` is reachable via a single `(s, t)` edge which comes last in the adjacency list of `s`.
Then all paths of length ≤ k inside the clique are fruitlessly explored before the edge and hence the path `(s, t)` is finally found.

To speed-up the enumeration, each vertex `x` gets a barrier `b[x]`, a lower bound on the remaining distance from `x` to `t`. 
Let `h` denote the length of the current search path from `s` to `x`.
If `search(x)` finds no path to `t`, the barrier is raised to `b[x] = k - h + 1`.
When some later `search(v)` with search path length `h'` scans the same node `x` in its successor loop,
the entry condition `if b[x] + h' < k` ensures that `search(x)` is called 
only if the new search path length `h' + 1` for `x` is smaller than `h`;
i.e. if that later search arrives at `x` with more budget left to reach `t`. 
In this phase barriers only increase.
Each vertex is entered at most k+1 times, so the total work for finding P0 is bounded by (k+1)(n+m). 
For code, see [bbdfs](bbdfs.md).

Now, when more paths than P0 are requested, the barriers already established shall be reused.
But a formerly fruitless search may become fruitful once vertices are popped from the search path, because they no longer block.
So barriers raised with respect to an earlier search path may need correction.
This is done by the `fruitful` procedure, and it must be done carefully.
Resetting all barriers to 0 when a node is popped keeps the algorithm correct
but throws away the work already spent, and the delay bounds no longer hold.
Instead, `b[v]` is set to `sd`, the length of the shortest path to `t` found from `v` with respect to the current search path.
The barriers of its direct and indirect predecessors are then decreased, walking backwards over in-edges. 
But only as far as needed to restore *edge-consistency*, a property defined and discussed in the paper.
Informally: a vertex's barrier may exceed that of its successors by at most one. This is [bsdfs](bsdfs.md).
