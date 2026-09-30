# The gate-trap graph (Γc) family
#
# inner interval reaches (2k−5−o(1))(n+m) steps
#
# Max.Delay(Γc, k)/((k+1)(n+m)) → 2 for c, k → ∞, k = o(c) 
#
# This asymptotically reaches the recently found worst-case delay bound of max. 2(k+1)(n+m) steps per interval.
#
# 2026-09-30 Frank Bauernöppel


from collections import deque


def bsdfs_delays(G, s, t, k):
    """returns interval delays"""
    b = {x: 0 for x in G.nodes}
    S = []
    steps = [0]

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            steps[-1] += 1                              # dequeue
            for p in G.predecessors(q):
                steps[-1] += 1                          # pred scan
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        steps[-1] += 1                                  # entry
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            steps[-1] += 1                              # succ scan
            if b[w] + h < k:
                if w == t:
                    # output S + [t]
                    steps[-1] += len(S) + 1             # output
                    steps.append(0)
                    sd = 1
                elif w not in S:
                    d = search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    search(s)
    return steps


import networkx as nx


def gate_k(c):
    """the k maximising rho_1 = delay(1) / ((k+1)(n+m)) on Gamma_c, by the closed form of gate_family.tex (valid for k >= 6); about sqrt(7c)"""
    def N(L):
        L = min(L, c - 1)
        return 1 + L * (c - 1) - L * (L - 1) // 2
    return max(range(6, c + 5), key=lambda k: ((c + 1) * (N(k - 3) + N(k - 5)) + 3 * c * c + 4 * c + 21) / ((k + 1) * (c * c + 2 * c + 11)))


def gate(c, k=None):
    """Gamma_c of gate_family.tex as (G, s, t, k):  s:[q,g];  q:[t,g,B0..B{c-1}];  clique B0..B{c-1} (index order), then every Bi -> g;  g:[B0,q,s]
    Edges are inserted in scan order, so G.successors(v) yields exactly these lists. Default k: gate_k(c), the k of the longest normalized interval."""
    B = [f"B{i}" for i in range(c)]
    E = [("s", "q"), ("s", "g"), ("q", "t"), ("q", "g")] + [("q", v) for v in B]
    for u in B:
        E += [(u, v) for v in B if v != u] + [(u, "g")]
    E += [("g", "B0"), ("g", "q"), ("g", "s")]
    G = nx.DiGraph()
    G.add_edges_from(E)
    return G, "s", "t", gate_k(c) if k is None else k


for c in range(10, 1100, 100):
    G, s, t, k = gate(c)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    delays = bsdfs_delays(G, s, t, k)
    steps = delays[1]
    print(f"{c=:6}: {steps=:16} {steps/((2*k-5)*(n+m))=:10.6f}")