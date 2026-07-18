"""adversarial graph family"""

import networkx as nx


def diamond_chain(t):
    G = nx.DiGraph()
    for i in range(t):
        G.add_edges_from([(f"d{i}", f"a{i}"), (f"d{i}", f"b{i}"),
                          (f"a{i}", f"d{i+1}"), (f"b{i}", f"d{i+1}")])
    G.add_edge(f"d{t}", "d0")     # every cycle has length 2t+1
    s = "d0"
    k = 2*t
    return G, s, k


def wave_gadget(t):
    """Family B: conjectured Theta(t^3) total work for the -1-fixed
    _bounded_cycle_search; k = t+1, n = 3t+2, m = 6t+1, exactly one cycle.
    Adjacency order is load-bearing (nx preserves insertion order):
    spine edge first, long arm before short arm, subscription last."""
    assert t >= 3
    G = nx.DiGraph()
    S = "S"
    Q = [f"Q{r}" for r in range(1, t + 1)]        # Q[r-1] is q_r
    D = [f"D{i}" for i in range(t + 1)]
    X = [f"X{i}" for i in range(t)]

    G.add_edge(S, Q[0])                            # spine, first at every vertex
    for r in range(t - 1):
        G.add_edge(Q[r], Q[r + 1])
    G.add_edge(Q[t - 1], S)                        # the unique <=k cycle, length t+1

    for r in range(t):                             # launches, after the spine edge
        G.add_edge(Q[r], D[0])

    for i in range(t):                             # unequal diamonds, long arm first
        G.add_edge(D[i], X[i])
        G.add_edge(X[i], D[i + 1])
        G.add_edge(D[i], D[i + 1])

    for i in range(1, t + 1):                      # subscriptions d_i -> q_i, last
        G.add_edge(D[i], Q[i - 1])
    
    s = "S"
    k = t + 1
    return G, s, k


def dag_plus_backedges(n, p, beta, rng):
    """Random DAG G(n, p) on topological order 0..n-1 (edges i->j, i<j only)
    plus beta feedback edges j->i (i<j) sampled uniformly without repetition.
    Every cycle uses >= 1 back edge; for small beta, cycles through a given
    node are scarce while the forward reachable cone stays large."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):                       # forward: upper-triangular G(n,p)
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(i, j)

    back = set()
    max_back = n * (n - 1) // 2
    assert beta <= max_back
    while len(back) < beta:                  # backward: beta distinct pairs
        i = rng.randrange(n - 1)
        j = rng.randrange(i + 1, n)
        e = (j, i)
        if e not in back and not G.has_edge(j, i):
            back.add(e)
            G.add_edge(j, i)
    return G


def ring(n, r):
    """ r‑neighbour circulant digraph"""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u+1, u+r+1):
            G.add_edge(u, v % n)
    return G 

def loose_breaker(k, F):
    """Spine s->v1..vp->t (+ s->a->t), d-chain d1..d_{k-2} each dj->x,
    hub x with fan of size F, conduit x->d1, branches d_{j(t)}->v_t
    with j(t)=k-p+t-2. Requires k>=8."""
    p = (k - 2) // 2
    D = k - 2
    assert p >= 2 and k - p >= 1
    G = nx.DiGraph()
    s, t, a, x = 's', 't', 'a', 'x'
    v = [f'v{i}' for i in range(1, p + 1)]
    d = [f'd{j}' for j in range(1, D + 1)]
    G.add_edge(s, v[0]); G.add_edge(s, a); G.add_edge(a, t)
    for i in range(p):
        G.add_edge(v[i], v[i + 1] if i < p - 1 else t)
        G.add_edge(v[i], d[0])
    for j in range(D):
        if j < D - 1:
            G.add_edge(d[j], d[j + 1])
        G.add_edge(d[j], x)
    for tt in range(2, p + 1):            # branches
        jt = k - p + tt - 2                # 1-based chain position
        assert 1 <= jt <= D
        G.add_edge(d[jt - 1], v[tt - 1])
    for q in range(F):
        G.add_edge(x, f'f{q}')
    G.add_edge(x, d[0])                    # cascade conduit
    return G, s, t


if False:
    G, s, k = diamond_chain(10)
    nx.drawing.nx_pydot.write_dot(G, "diamond_chain.dot")

    G, s, k = wave_gadget(5)
    nx.drawing.nx_pydot.write_dot(G, "wave_gadget.dot")

    k = 8
    F = 20
    G, s, k = loose_breaker(k, F)
    nx.drawing.nx_pydot.write_dot(G, "loose_breaker.dot")
