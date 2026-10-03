"""
Read-Tarjan procedure CYCLE, Networks version, modern Python transcription of

  author  = {Ronald C. Read and Robert E. Tarjan},
  title   = {Bounds on Backtrack Algorithms for Listing Cycles, Paths, and Spanning Trees},
  journal = {Networks}, volume = {5}, number = {3}, pages = {237--252}, year = {1975},  (procedure CYCLE, pp. 246-247)

Faithful to the printed steps C1-C5, with two parameters that the paper leaves open or gets wrong:
  c2  = "dfs" | "bfs"   how C2 constructs the path w = w_1, ..., w_n = s ("implemented as a search", unspecified)
  fix = False | True    printed C3 tests "u != w_{i+1} and d(u) = false" only; since d(s) = p(s) = true, a direct edge w_i -> s
                        is never recognised as an alternate route. fix=True adds "if u = s then flag := true".
Vertex numbers are DFS numbers (C4); "w > z" in C1 = successors in increasing number; C5 = only vertices >= s.
Self-loops are reported as cycles (the paper's convention excludes them; nx.simple_cycles includes them).
cnt (optional dict): calls = BACKTRACK calls, work = successor/adjacency entries scanned + vertices visited in all searches.
"""

from collections import deque
import networkx as nx


def read_tarjan_75(G, c2="dfs", fix=False, cnt=None):
    if cnt is None:
        cnt = {}
    cnt.setdefault("calls", 0)
    cnt.setdefault("work", 0)
    cnt.setdefault("starts", 0)
    for comp in nx.strongly_connected_components(G):
        yield from _cycle(G, comp, c2, fix, cnt)


def _cycle(G, comp, c2, fix, cnt):
    # C4: depth-first search; number vertices; cycle arcs = arcs into a vertex on the DFS stack
    root = next(v for v in G.nodes if v in comp)
    num = {root: 0}
    order = [root]
    active = {root}
    starts = set()
    stack = [(root, iter(G.successors(root)))]
    while stack:
        u, it = stack[-1]
        for w in it:
            if w not in comp:
                continue
            if w not in num:
                num[w] = len(order)
                order.append(w)
                active.add(w)
                stack.append((w, iter(G.successors(w))))
                break
            if w in active:
                starts.add(num[w])
        else:
            stack.pop()
            active.discard(u)

    n = len(order)
    A_full = [[num[w] for w in G.successors(v) if w in comp] for v in order]   # adjacency order kept
    p = [False] * n
    path = []

    for s in sorted(starts):
        cnt["starts"] += 1
        A = [[w for w in A_full[v] if w >= s] for v in range(n)]              # C5

        def route(w):
            """C1/C2: a path w = w_1, ..., w_n = s avoiding the current path (except s), or None"""
            if w == s:
                return [s]
            if p[w]:
                return None
            parent = {w: None}
            found = False
            if c2 == "bfs":
                queue = deque([w])
                while queue and not found:
                    x = queue.popleft()
                    cnt["work"] += 1 + len(A[x])
                    for y in A[x]:
                        if y == s:
                            parent[s] = x
                            found = True
                            break
                        if y not in parent and not p[y]:
                            parent[y] = x
                            queue.append(y)
            else:
                dfs = [(w, iter(A[w]))]
                cnt["work"] += 1 + len(A[w])
                while dfs and not found:
                    x, it = dfs[-1]
                    for y in it:
                        if y == s:
                            parent[s] = x
                            found = True
                            break
                        if y not in parent and not p[y]:
                            parent[y] = x
                            dfs.append((y, iter(A[y])))
                            cnt["work"] += 1 + len(A[y])
                            break
                    else:
                        dfs.pop()
            if not found:
                return None
            E = [s]
            while E[-1] != w:
                E.append(parent[E[-1]])
            E.reverse()
            return E

        def backtrack(v):
            cnt["calls"] += 1
            base = len(path)
            for w in sorted(A[v]):                                            # C1: w > z
                cnt["work"] += 1
                E = route(w)
                if E is None:
                    continue
                # C2 done; C3
                d = p[:]
                flag = False

                def DFS(u):
                    nonlocal flag
                    d[u] = True
                    cnt["work"] += 1
                    for x in A[u]:
                        if flag:
                            break
                        cnt["work"] += 1
                        if x == s:
                            flag = True
                        elif not d[x]:
                            DFS(x)

                i = 0
                while E[i] != s and not flag:
                    wi = E[i]
                    path.append(wi)
                    p[wi] = d[wi] = True
                    for u in A[wi]:
                        cnt["work"] += 1
                        if u != E[i + 1]:
                            if fix and u == s:
                                flag = True
                            elif not d[u]:
                                DFS(u)
                    i += 1
                if not flag:
                    yield [order[x] for x in path]                            # add w_i = s; output cycle
                else:
                    yield from backtrack(E[i - 1])
                for x in path[base:]:                                         # delete vertices after v
                    p[x] = False
                del path[base:]

        p[s] = True
        path.append(s)
        yield from backtrack(s)
        path.pop()
        p[s] = False


def main():
    # direct edge a -> s next to the C2 route a -> b -> s; adjacency order of a: b before s
    G = nx.DiGraph([("s", "a"), ("a", "b"), ("a", "s"), ("b", "s")])
    for c2 in ("dfs", "bfs"):
        for fix in (False, True):
            print(f"{c2=} {fix=}: {list(read_tarjan_75(G, c2, fix))}")


if __name__ == "__main__":
    main()
