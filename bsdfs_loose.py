from collections import deque


def bsdfs(G, s, t, k):
    """loose scheme, barriers are reset to 0"""
    b = {x: 0 for x in G.nodes}
    S = []

    def reset(v):
        b[v] = 0
        queue = deque([v])
        while queue:
            q = queue.popleft()
            for p in G.predecessors(q):
                if p not in S and b[p] != 0:
                    b[p] = 0
                    queue.append(p)

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
            reset(v)
        else:
            b[v] = k - h + 1

        S.pop()
        return sd

    yield from search(s)

if __name__ == "__main__":
    import experiments_base as base

    base.smoke(bsdfs)
