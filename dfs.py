from math import inf


def all_simple_paths(G, s, t, k=inf):
    """minimalistic depth-limited DFS for simple path enumeration"""
    path = []

    def dfs(v):
        if v == t:
            yield path + [t]
        elif len(path) < k:
            path.append(v)
            for w in G.successors(v):
                if w not in path:
                    yield from dfs(w)
            path.pop()

    yield from dfs(s)


if __name__ == "__main__":
    import experiments_base as base

    base.smoke(all_simple_paths)
