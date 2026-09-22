# Bounded Scope Depth-First Search

Python translation of the pseudocode in our paper.
This algorithm extends [dldfs](dldfs.md) and [bbdfs](bbdfs.md), avoiding their pitfalls.

```python
from collections import deque

def bsdfs(G, s, t, k):
    """enumerate all length k bounded st-paths in G"""
    b = {x: 0 for x in G.nodes} # init barriers
    S = [] # current search path (stack)

    def fruitful(v, sd):
        """reverse BFS from v, updating barriers"""
        b[v] = sd # update own barrier to sd found
        # predecessor cascade: drop excess barriers
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1 # drop barrier
                    queue.append((p, d + 1))

    def search(v):
        """recursive DFS at node v"""
        S.append(v)
        h = len(S) - 1 # h = number of edges in S
        sd = k + 1 # shortest distance to t found so far, (k+1 acts as inf)
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t] # output
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            # fruitless, raise barrier
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)
```

Several optimizations are possible but not shown for clarity.
