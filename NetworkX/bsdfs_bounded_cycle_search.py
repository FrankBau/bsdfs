"""BS-DFS drop-in replacement for networkx.algorithms.simple_paths._all_simple_edge_paths.
"""


from collections import defaultdict, deque
from networkx.algorithms.cycles import _NeighborhoodCache


def bsdfs_bounded_cycle_search(G, path, length_bound):
    """Drop-in for _bounded_cycle_search: simple cycles beginning with
    the given prefix, length (= #nodes = #edges) <= length_bound."""
    k = length_bound
    t = path[0]

    if G.is_directed():
        succ = _NeighborhoodCache(G)          # G[v]  -> successors
        pred = _NeighborhoodCache(G.pred)     # G.pred[v] -> predecessors
    else:
        succ = pred = _NeighborhoodCache(G)   # one shared cache: neighbors

    b = defaultdict(int)
    S = list(path)                     # do not mutate caller's list (nx does; we needn't)
    on_stack = set(S)                  # prefix nodes stay blocked forever
    iters = [iter(succ[S[-1]])]        # only the last prefix node gets a frame
    sds = [k + 1]

    def fruitful(v, sd):
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            for p in pred[q]:
                if p not in on_stack and b[p] > d + 1:
                    b[p] = d + 1
                    queue.append((p, d + 1))

    while iters:
        h = len(S) - 1
        for w in iters[-1]:
            if b[w] + h < k:
                if w == t:
                    yield S[:]                     # nx format: no closing repetition
                    sds[-1] = 1
                elif w not in on_stack:
                    S.append(w)
                    on_stack.add(w)
                    iters.append(iter(succ[w]))
                    sds.append(k + 1)
                    break
        else:
            v = S.pop()
            on_stack.remove(v)
            iters.pop()
            sd = sds.pop()
            if sd <= k:
                fruitful(v, sd)
            else:
                b[v] = k - h + 1
            if sds and sd + 1 < sds[-1]:
                sds[-1] = sd + 1


def monkey_patching_pytest():
    import networkx.algorithms.cycles as cycles

    orig = cycles._bounded_cycle_search
    cycles._bounded_cycle_search = bsdfs_bounded_cycle_search

    import pytest
    pytest.main(["--doctest-modules", "--pyargs", "networkx"])
    # expected:  7720 passed, 87 skipped, 1 xfailed, 11 warnings in 76.91s (0:01:10)
    assert cycles._bounded_cycle_search is bsdfs_bounded_cycle_search # not restored mid-way
    # expected output: 65 passed, 5 skipped, 6979 deselected, 3 warnings in 4.46s
    # (the warnings are unrelated)

    cycles._bounded_cycle_search = orig
    assert  cycles._bounded_cycle_search is not bsdfs_bounded_cycle_search 
    # now restored


if __name__ == "__main__":
    monkey_patching_pytest()
