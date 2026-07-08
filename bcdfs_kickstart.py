"""

"""


def bcdfs(G, s, t, k):
    """original with added kick-start, fixing completeness bug"""
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
            bar[u] = k - length(S) + 1
        else:
            bar[u] = k + 1      # kick-start
            UpdateBarrier(u, F)
        S.pop()
        return F

    yield from search(s)


import networkx as nx
from bsdfs import bsdfs


def main():
    G = nx.parse_adjlist(
        ["a b c", "b c d e", "c b d", "d b", "e"], create_using=nx.DiGraph
    )

    paths1 = list(bcdfs(G, s="a", t="e", k=4))
    assert paths1 == [
        ["a", "b", "e"],
        ["a", "c", "b", "e"],
        ['a', 'c', 'd', 'b', 'e']
    ]  # missing None


if __name__ == "__main__":
    main()
