def bcdfs(G, s, t, k, stats=None):
    """original control-flow, NO completeness

    stats, if given, is a dict mutated in place (pass the same dict
    across multiple calls to accumulate corpus-wide totals):
      search_calls            -- search() invocations (DFS tree nodes visited)
      successors_considered   -- iterations of the successor loop in search()
      predecessors_considered -- iterations of the predecessor loop in UpdateBarrier()
      bar_updates             -- bar[] entries actually assigned (gated by
                                  `bar[u] > l`, so the top-level assignment
                                  can be skipped where bsdfs's cannot -- see note)
    """
    if stats is None:
        stats = {}
    for key in ("search_calls", "successors_considered",
                "predecessors_considered", "bar_updates"):
        stats.setdefault(key, 0)

    S = []
    bar = {v: 0 for v in G.nodes}

    def length(S):
        return len(S) - 1

    def UpdateBarrier(u, l):
        if bar[u] > l:
            bar[u] = l
            stats["bar_updates"] += 1
            for v in G.predecessors(u):
                stats["predecessors_considered"] += 1
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        stats["search_calls"] += 1
        F = k + 1
        S.append(u)
        if u == t:
            yield S.copy()
            S.pop()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                stats["successors_considered"] += 1
                if v not in S:
                    if length(S) + 1 + bar[v] <= k:
                        f = yield from search(v)
                        if f != k + 1:
                            F = min(F, f + 1)
        if F == k + 1:
            bar[u] = k - length(S) + 1
        else:
            UpdateBarrier(u, F)
        S.pop()
        return F

    yield from search(s)
