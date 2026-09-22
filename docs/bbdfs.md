# Barrier Bounded Depth-First Search

Half-way improvement of [dldfs](dldfs.md).
Finds correctly the first path (or returns None) in O(k(n+m)) time by maintaining barriers.
May miss output after the first path due to stale barrier vaules.
See [bsdfs](bsdfs.md) for correction.

```python
def bbdfs(G, s, t, k):
    """barrier bounded DFS"""
    b = {x: 0 for x in G.nodes} # init barriers
    S = [] # current search path (stack)

    def search(v):
        S.append(v)
        h = len(S) - 1 # h = number of edges in S
        sd = k + 1 # shortest distance to t found so far (k + 1 acts as inf)
        for w in G.successors(v):
            if b[w] + h < k: # admissible length?
                if w == t:
                    yield S + [t] # output
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd > k:
            # fruitless, raise barrier
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)

if __name__ == "__main__":
    from bsdfs import bsdfs

    c = 100
    G = nx.complete_graph(c, create_using=nx.DiGraph)
    s = 0
    t = c
    k = c - 1
    G.add_edge(s, t) # appended last to adj. list of s

    P0 = next(bbdfs(G, s, t, k), None) # get first path fast, like bsdfs
    print("bbdfs:", P0)

    # an example where bbdfs misses a path after the first
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 5), (1, 2), (1, 4), (5, 2), (2, 3), (2, 4), (3, 1)])
    s = 0
    t = 4
    k = 5
    paths = list(bbdfs(G, s, t, k))
    print(f"bbdfs: {paths}") # output [[0, 1, 2, 4], [0, 1, 4], [0, 5, 2, 4]]
    paths = list(bsdfs(G, s, t, k))
    print(f"bsdfs: {paths}") # output [[0, 1, 2, 4], [0, 1, 4], [0, 5, 2, 3, 1, 4], [0, 5, 2, 4]]
```
