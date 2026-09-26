A short intro to the [BS-DFS](bsdfs.md) algorithm from our [preprint](https://doi.org/10.48550/arXiv.2607.14745):

> Frank Bauernöppel, Jörg-Rüdiger Sack;
> "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with O(k(n+m)) Delay Using Edge-Consistent Node Barriers"

submitted to [The Journal of Graph Algorithms and Applications (JGAA)](https://jgaa.info/).


# Basics

A call `bsdfs(G, s, t, k)` enumerates all simple paths in graph `G`
starting at vertex `s` and terminating at vertex `t` having at most `k` edges.
For the special case `s==t`, simple cycles through vertex `s` are enumerated,
including cycles of length 2, which are excluded in some literature.

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

prints 3 paths:
```python
[0, 1, 2, 4]
[0, 2, 3, 4]
[0, 2, 4]
```

Path `[0, 1, 2, 3, 4]` is too long to be reported for `k = 3`.
If we set `k = 4`, it will be reported as well.
The output order depends on the adjacency-list order of the internal graph representation.


# Efficiency - The Delay Bounds

The number of length-bounded paths or cycles can grow exponentially in the graph size, even for fixed k.
Luckily, `bsdfs` is implemented as a generator function and can be stopped whenever "enough" paths have been generated.
But: how long do we have to wait for the next output or termination?

Let's call the start, each path generation (output), and the termination the *events* of a call `bsdfs(G, s, t, k)`.
Then, the *delay* (waiting time between two consecutive events) is bounded by O(k(n+m)).
Here, n=|V| is the number of vertices and m=|E| is the number of edges in G.
So, for any fixed k, the delay is linear in the graph size.

With a suitable definition of elementary steps (vertex visits, edge scans, barrier writes, output),
the following is proven in our paper:

- the worst-case delay between two consecutive events is at most 3(k+1)(n+m) steps, and
- for every p≥1, the first p events are produced within 2p(k+1)(n+m) steps,
  i.e. the amortized delay is at most 2(k+1)(n+m) steps per event.

In benchmark tests, the observed delays were typically well below these theoretical bounds.


# Motivation

A depth-first search will eventually enumerate all s-t paths or s-cycles.
If the search depth is limited to k (see [dldfs](dldfs.md)), length-bounded
s-t paths or s-cycles will be enumerated.
The problem is that this kind of search recurses over and over again when a vertex is revisited.
For example, let `s` lie in a large clique from which `t` is reachable via a single edge `(s, t)` which comes last in the adjacency list of `s`.
Then all search paths of length ≤ k inside the clique are fruitlessly explored
*before* edge (s, t) is scanned and the path (s, t) is reported.
The number of such search paths, and hence the delay until the first output, can be exponential in the clique size.

To avoid repeated fruitless searches, each vertex `x` gets a barrier `b[x]`,
initially 0, see [bbdfs](bbdfs.md).
Let `h` denote the length of the current search path from `s` to `x`.
If `search(x)` finds no path to `t`, the barrier is raised to `b[x] = k - h + 1`.
When some later `search(y)` with search path length `h'` scans the same vertex `x` in its successor loop,
the entry condition `if b[x] + h' < k` ensures that `search(x)` is called
only if the new search path length `h' + 1` (including `x`) is smaller than the old length `h`;
i.e. if that later search arrives at `x` with more budget left to reach `t`.
In this phase barriers only increase.
Each vertex is entered at most k+1 times, so the total work for finding a first path/cycle
is bounded by (k+1)(n+m).

The above method works if only the *first* s-t path P0 of length ≤ k is wanted (or only its existence).
Barriers were raised with respect to the then-current search path.
When the search backtracks, popped vertices are no longer blocked for a re-visit
and a formerly fruitless search may become a fruitful one.

How does [bsdfs](bsdfs.md) repair barriers for re-use when more paths than P0 are requested?
Resetting all barriers to 0 when a vertex `v` is popped keeps the algorithm correct
but throws away the work already spent, and the delay bounds no longer hold.
Instead, `b[v]` is set to `sd`,
the length of the shortest path to `t` found from `v` with respect to the now-current search path.
The barriers of its direct and indirect predecessors are then decreased by walking backwards over in-edges in the `fruitful` procedure.
But only as far as needed to restore *edge-consistency*, a property defined and discussed in the paper.


# BS-DFS in Action

```python
  Z = nx.parse_adjlist(['s a d', 'a c t', 'b a', 'c b z', 'd t z', 'z b c'], create_using=nx.DiGraph)
  list(bsdfs(Z, "s", "t", 5))
```

<video class="video-full" controls playsinline preload="metadata">
  <source src="{{ '/assets/videos/bsdfs_steps_example.mp4' | relative_url }}" type="video/mp4">
  <a href="{{ '/assets/videos/bsdfs_steps_example.mp4' | relative_url }}">Download the animation (MP4, 3 MB)</a>
</video>


# BS-DFS in Competition

<video class="video-full" controls playsinline preload="metadata">
  <source src="{{ '/assets/videos/bsdfs_calls_N14_seed88_k5.mp4' | relative_url }}" type="video/mp4">
  <a href="{{ '/assets/videos/bsdfs_calls_N14_seed88_k5.mp4' | relative_url }}">Download the animation (MP4, 3 MB)</a>
</video>
