"""
Demonstration that for the loose_breaker(k) graph family,
- the tight scheme holds, but
- the loose and lazy schemes break
the O(k(n+m)) delay per output

By counting steps and comparing to lower bound.
"""


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


steps = {"count": 0}

from collections import deque
import networkx as nx


def bsdfs_loose(G, s, t, k):
    """loose scheme, barriers are reset to 0"""
    b = {x: 0 for x in G.nodes}
    S = []

    def reset(v):
        b[v] = 0
        queue = deque([v])
        while queue:
            steps["count"] += 1
            q = queue.popleft()
            for p in G.predecessors(q):
                steps["count"] += 1
                if p not in S and b[p] != 0:
                    b[p] = 0
                    queue.append(p)

    def search(v):
        steps["count"] += 1
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            steps["count"] += 1
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)
        if sd <= k:
            reset(v)
        else:
            b[v] = k - h + 1
        S.pop()
        return sd

    yield from search(s)


def bsdfs_lazy(G, s, t, k):
    """lazy version, using B sets"""
    b = {x: 0 for x in G.nodes}
    B = {x: set() for x in G.nodes}
    S = []

    def update(v):
        steps["count"] += 1
        for u in B[v]:
            steps["count"] += 1
            if u not in S and b[u] > 0:
                b[u] = 0
                update(u)
        B[v].clear()

    def search(v):
        steps["count"] += 1
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            steps["count"] += 1
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            b[v] = 0
            update(v)
        else:
            b[v] = k - h + 1
            for w in G.successors(v):
                steps["count"] += 1
                B[w].add(v)

        S.pop()
        return sd

    yield from search(s)


# k = 12
# G, s, t, k = loose_breaker(k)
# paths = list(bsdfs_loose(G, s, t, k))
# nx.drawing.nx_pydot.write_dot(G, f"L{k}.dot")


def run(algo, k_max=300):
    print(algo.__name__)
    for k in range(8, k_max, 4):
        G, s, t, k = loose_breaker(k)
        steps["count"] = 0
        paths = list(algo(G, s, t, k))
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # hub entry count: E(r) = min(D, k − r − 1) = k − r − 1  for 1 ≤ r ≤ k/4 − 1
        lower_bound = sum((k-r-1)*(k+1) for r in range(1, k//4)) # lower bound claim, F = k, k % 4 == 0
        R = k // 4 - 1; assert lower_bound == (k+1)*((k-1)*R - R*(R+1)//2)     # same in closed form

        print(f"{k=:5} {steps["count"]=:10} {lower_bound=:10} {steps["count"]/(k*k*(n+m))=:10.4f} {lower_bound/(k*k*(n+m))=:10.4f}")
        assert steps["count"] >= lower_bound

        # loose scheme:
        # the ratio converges to 1/32 = 0.03125
        # Armed-fed explorations r = 1..R, R ≈ (k−2)/4;
        # expl(r) makes ≈ k−r hub entries, each costing F+1 = k+1 successor scans.
        # Dominant total: Σ_{r≤R} (k−r)(k+1) ≈ k(Rk − R²/2) = (1/4 − 1/32)k³ = (7/32)k³.
        # With n+m = 7k − 6 (count: n = 5k/2, m = 9k/2 − 6),
        # the ratio tends to (7/32)k³ / (7k³) = 1/32 = 0.03125

run(bsdfs_loose)
run(bsdfs_lazy) # ratio converges to 2/32 = 0.0625. The B[w].add(v) loop is a second pass over the successor list.
