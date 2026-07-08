from collections import deque


def bsdfs(G, s, t, k, stats=None):
    """tight scheme (original BSDFS)

    stats, if given, is a dict mutated in place (pass the same dict
    across multiple calls to accumulate corpus-wide totals):
      search_calls            -- search() invocations (DFS tree nodes visited)
      successors_considered   -- iterations of the successor loop in search()
      predecessors_considered -- iterations of the predecessor loop in fruitful()
      bar_updates             -- b[] entries actually assigned (v's own value,
                                  unconditional, plus cascaded predecessor updates)
    """
    if stats is None:
        stats = {}
    for key in ("search_calls", "successors_considered",
                "predecessors_considered", "bar_updates"):
        stats.setdefault(key, 0)

    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        stats["bar_updates"] += 1
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                stats["predecessors_considered"] += 1
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    stats["bar_updates"] += 1
                    queue.append((p, d + 1))

    def search(v):
        stats["search_calls"] += 1
        S.append(v)
        h = len(S) - 1
        sd = k + 1
        for w in G.successors(v):
            stats["successors_considered"] += 1
            if b[w] + h < k:
                if w == t:
                    yield S + [t]
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)
