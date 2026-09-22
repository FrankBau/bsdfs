# Barrier Bounded Depth-First Search

```python
def bbdfs(G, s, t, k):
    """barrier bounded DFS"""
    b = {x: 0 for x in G.nodes # init barriers
    S = [] # current search path (stack)

    def search(v):
        S.append(v)
        h = len(S) - 1 # h = number of edges in S
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:  # admissible length?
                if w == t:
                    yield S + [t] # output
                elif w not in S:
                    yield from search(w)

        if sd > k:
            # fruitless, raise barrier
            b[v] = k - h + 1

        S.pop()

    yield from search(s)

if __name__ == "__main__":
    c = 100
    G = nx.complete_graph(c, create_using=nx.DiGraph)
    s = 0
    t = c
    k = c - 1
    G.add_edge(s, t) # appended last to adj. list of s

    P0 = next(bbdfs(G, s, t, k), None) # get first path fast, like bsdfs
    print("bbdfs:", P0)
```
