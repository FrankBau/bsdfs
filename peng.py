"""
python translation of pseudo-code from

  title        = {Efficient Hop-constrained s-t Simple Path Enumeration},
  volume       = {30},
  year         = {2021},
  issn         = {0949-877X},
  url          = {https://doi.org/10.1007/s00778-021-00674-5},
  doi          = {10.1007/s00778-021-00674-5},
  pages        = {799--823},
  number       = {5},
  journaltitle = {The {VLDB} Journal},
  author       = {Peng, You and Lin, Xuemin and Zhang, Ying and Zhang, Wenjie and Qin, Lu and Zhou, Jingren},
  date         = {2021-09-01}
"""


def bcdfs(G, s, t, k):
    """original control-flow, NO completeness"""
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
    ]  # missing ['a', 'c', 'd', 'b', 'e']


if __name__ == "__main__":
    main()
