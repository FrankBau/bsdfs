# Depth-Limited Depth-First Search

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

if __name__ == "__main__":
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
