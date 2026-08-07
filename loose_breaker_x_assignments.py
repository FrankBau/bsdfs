"""
Demonstration that for the loose_breaker(k) graph family,
- the tight scheme holds, but
- the loose and lazy schemes break 
the O(k(n+m)) delay per output

By printing b[x] assignments (sawtooth) on small instances
"""

from collections import deque
import sys
sys.setrecursionlimit(100_000)


def bcdfs(G, s, t, k):
    """original control-flow, completeness NOT guaranteed"""
    S = []
    bar = {v: 0 for v in G.nodes}

    def length(S):
        return len(S) - 1

    def UpdateBarrier(u, l):
        if bar[u] > l:
            bar[u] = l
            for v in G.predecessors(u):
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        F = k + 1
        S.append(u)
        if u == t:
            yield S.copy()
            S.pop()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                if v not in S:
                    if length(S) + 1 + bar[v] <= k:
                        f = yield from search(v)
                        if f != k + 1:
                            F = min(F, f + 1)
        if F == k + 1:
            if u=='x': print(f"{bar['x']:2}", end=' ')
            bar[u] = k - length(S) + 1
        else:
            UpdateBarrier(u, F)
            print(f"{u}")
        S.pop()
        return F

    yield from search(s)
    print("")


def bsdfs_tight(G, s, t, k):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = list()
    on_path = set()
    
    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in on_path and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        on_path.add(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in on_path:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
            print(f"{v}")
        else:
            b[v] = k - h + 1
            if v=='x': print(f"{b['x']:2}", end=' ')
        S.pop()
        on_path.remove(v)
        return sd

    yield from search(s)
    print("")


def bsdfs_loose(G, s, t, k):
    """loose scheme, barriers are reset to 0"""
    b = {x: 0 for x in G.nodes}
    S = []

    def reset(v):
        b[v] = 0
        queue = deque([v])
        while queue:
            q = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] != 0:
                    b[p] = 0
                    queue.append(p)

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            reset(v)
            print(f"{v}")
        else:
            b[v] = k - h + 1
            if v=='x': print(f"{b['x']:2}", end=' ')
        S.pop()
        return sd

    yield from search(s)
    print("")


def bsdfs_lazy(G, s, t, k):
    """lazy version, using B sets"""
    b = {x: 0 for x in G.nodes}
    B = {x: set() for x in G.nodes}
    S = []

    def update(v):
        for u in B[v]:
            if u not in S and b[u] > 0:
                b[u] = 0
                update(u)
        B[v].clear()

    def search(v):
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            b[v] = 0
            print(f"{v}")
            update(v)
        else:
            b[v] = k - h + 1
            if v=='x': print(f"{b['x']:2}", end=' ')
            for w in G.successors(v):
                B[w].add(v)

        S.pop()
        return sd

    yield from search(s)
    print("")


import networkx as nx


def loose_breaker(k):
    """
        - spine s->v1..vp->t (+ s->t),      p = ⌊(k-2)/2⌋
        - d-chain d1..dD each dj->x,        D = k-2
        - hub x with 
        - fan of size F,                    F = k
        - conduit x->d1, 
        - branches d_{j(i)}->v_i            j(i) = p+i. 
        Requires k>=8. Let k%4 == 0 for simplicity.
    """
    F = k                                   # fan size
    p = (k - 2) // 2
    D = k - 2
    assert p >= 2 and k - p >= 1
    G = nx.DiGraph()
    s, t, x = 's', 't', 'x'
    v = [f'v{i}' for i in range(1, p + 1)]
    d = [f'd{j}' for j in range(1, D + 1)]
    G.add_edge(s, v[0]); G.add_edge(s, t)
    for i in range(p):
        G.add_edge(v[i], v[i + 1] if i < p - 1 else t)
        G.add_edge(v[i], d[0])
    for j in range(D):
        if j < D - 1:
            G.add_edge(d[j], d[j + 1])
        G.add_edge(d[j], x)
    for i in range(2, p + 1):               # branches
        j = p + i                           # 1-based chain position
        assert 1 <= j <= D
        G.add_edge(d[j - 1], v[i - 1])
    for q in range(F):
        G.add_edge(x, f'f{q}')
    G.add_edge(x, d[0])                     # cascade conduit
    return G, s, t, k


def run(algo, k_max=40):
    print(algo.__name__)
    for k in range(8, k_max, 4):
        G, s, t, k = loose_breaker(k)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"{k=:4} {n=:4} {m=:4}   ", end='')
        paths = list(algo(G, s, t, k))
        print(paths)


if __name__ == "__main__":
    run(bsdfs_tight)
    run(bsdfs_loose)
    run(bsdfs_lazy)
    run(bcdfs)
