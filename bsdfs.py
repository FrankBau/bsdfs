"""BS-DFS -- the tight scheme, the reference implementation of the paper.

    bsdfs(G, s, t, k)

enumerates every simple path from s to t in the directed graph G that uses
at most k edges, in lexicographic order, with O(k(n+m)) delay per output.
Called as bsdfs(G, s, s, k) it enumerates the simple cycles through s.

Run this file directly for a self-check against NetworkX:

    python bsdfs.py

The systematic, claim-by-claim validation of the delay bounds lives in
validate_delay_bounds_v2/; the measurements reported in the paper are
produced by delay_bound_experiments.py, missed_paths_experiments.py and
runtime_experiments.py.
"""

from collections import deque


def bsdfs(G, s, t, k):
    """tight scheme (original BSDFS)"""
    b = {x: 0 for x in G.nodes}
    S = []

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

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
            fruitful(v, sd)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)


if __name__ == "__main__":
    # the algorithm above needs only the standard library; the self-check
    # additionally uses NetworkX as ground truth
    import experiments_base as base

    base.smoke(bsdfs)
