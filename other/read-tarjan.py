"""
Read-Tarjan cycle listing (procedure CYCLE), modern Python transcription of

  author      = {Ronald C. Read and Robert E. Tarjan},
  title       = {Bounds on Backtrack Algorithms for Listing Cycles, Paths, and Spanning Trees},
  institution = {University of California, Berkeley},  (TR version, pp. 11-14)
  year        = {1973/1975},

Transcription notes (deviations from the printed pseudocode):
  * C1 prints "mark u unconnectable"; this is a typo, C1 is the same backward search as C3 ("connectable").
  * "let C = {v | v in A(v) and v is connectable}" reuses v; read as C = {y in A(v) | y connectable}.
  * "while some x >= s is connectable and unscanned" is realised by a worklist: a vertex is pushed when
    it becomes connectable and scanned when popped. Same marks, each vertex scanned at most once.
  * C5: SCCs via NetworkX; inside each SCC one DFS gives the numbering and the cycle arcs
    (arcs into a vertex on the DFS stack, self-loops included). Vertices of an SCC are relabelled 0..n-1 by DFS number.
Cycles are yielded as vertex lists starting at their minimum-DFS-number vertex s, without repeating s.
"""

import networkx as nx


def read_tarjan(G):
    for comp in nx.strongly_connected_components(G):
        yield from _cycle(G, comp)


def _cycle(G, comp):
    # C5: depth-first search of the component; number vertices, locate vertices with entering cycle arcs
    root = next(iter(comp))
    num = {root: 0}
    order = [root]
    active = {root}
    starts = set()
    dfs = [(root, iter(G.successors(root)))]
    while dfs:
        u, it = dfs[-1]
        for w in it:
            if w not in comp:
                continue
            if w not in num:
                num[w] = len(order)
                order.append(w)
                active.add(w)
                dfs.append((w, iter(G.successors(w))))
                break
            if w in active:
                starts.add(num[w])            # cycle arc u -> w
        else:
            dfs.pop()
            active.discard(u)

    n = len(order)
    A = [[num[w] for w in G.successors(v) if w in comp] for v in order]
    B = [[num[u] for u in G.predecessors(v) if u in comp] for v in order]
    onpath = [False] * n
    connectable = [False] * n
    P = []

    def mark(s):
        """C1 / C3: backward search from s through vertices >= s not on the current path"""
        for i in range(n):
            connectable[i] = False
        connectable[s] = True
        work = [s]
        while work:
            x = work.pop()                    # x scanned
            for u in B[x]:
                if u >= s and not onpath[u] and not connectable[u]:
                    connectable[u] = True
                    work.append(u)

    def backtrack(s, v):
        mark(s)                                                  # C1
        C = [y for y in A[v] if connectable[y]]
        k = len(P)
        for w in C:                                              # C2
            mark(s)                                              # C3
            P.append(w)
            onpath[w] = True
            while w != s:                                        # C4
                xs = [x for x in A[w] if connectable[x]]
                if len(xs) != 1:
                    break
                w = xs[0]
                P.append(w)
                onpath[w] = True
            if w == s:
                yield [order[i] for i in P[:-1]]
            else:
                yield from backtrack(s, w)
            for u in P[k:]:                                      # delete vertices after v
                if u != s:
                    onpath[u] = False
            del P[k:]

    for s in sorted(starts):                                     # C6
        P.append(s)
        onpath[s] = True
        yield from backtrack(s, s)
        P.pop()
        onpath[s] = False


def main():
    # Tiernan Fig. 1, vertices 1..5; circuits {1,2,3,5}, {1,2,4,3,5}, {2}
    G = nx.DiGraph([(1, 2), (2, 2), (2, 3), (2, 4), (3, 5), (4, 3), (5, 1)])
    cycles = list(read_tarjan(G))
    print(cycles)
    assert sorted(map(sorted, cycles)) == [[1, 2, 3, 4, 5], [1, 2, 3, 5], [2]]


if __name__ == "__main__":
    main()
    