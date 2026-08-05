from collections import deque
from math import inf


def bsdfs(G, s, t, k):
    """hard coded: all barriers remain at 0; resulting in a length limited dfs"""
    S = []

    def search(v):
        S.append(v)
        h = len(S) - 1
        for w in G.successors(v):
            if h < k:
                if w == t:
                    yield S + [t]
                elif w not in S:
                    yield from search(w)
        S.pop()

    yield from search(s)

if __name__ == "__main__":
    import experiments_base as base

    base.smoke(bsdfs)
