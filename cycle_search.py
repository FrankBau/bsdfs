"""
python translation of pseudo-code from

  author     = {Anshul Gupta and
                Toyotaro Suzumura},
  title      = {Finding All Bounded-Length Simple Cycles in a Directed Graph},
  journal    = {CoRR},
  volume     = {abs/2105.10094},
  year       = {2021},
  url        = {https://arxiv.org/abs/2105.10094},
  eprinttype = {arXiv},
  eprint     = {2105.10094},
  timestamp  = {Mon, 31 May 2021 16:16:57 +0200},
"""


from math import inf


def CYCLE_SEARCH(G, s, k):
    """original control flow, NO completeness"""
    stack = []
    lock = {v: inf for v in G.nodes}
    Blist = {v: set() for v in G.nodes}

    def relax_locks(v, k, blen):
        if lock[v] < k - blen + 1:
            lock[v] = k - blen + 1
            for w in Blist[v]:
                if w not in stack:
                    relax_locks(w, k, blen + 1)

    def cycle_search(G, v, k, flen):
        blen = inf
        lock[v] = flen
        stack.append(v)

        for w in G.successors(v):
            if w == stack[0]:
                blen = 1
                yield stack.copy()
            elif (flen + 1 < lock[w]) and (flen + 1 < k):
                d = yield from cycle_search(G, w, k, flen + 1)
                blen = min(blen, 1 + d)
        if blen < inf:
            relax_locks(v, k, blen)
        else:
            for w in G.successors(v):
                if v not in Blist[w]:
                    Blist[w].add(v)
        stack.pop()
        return blen

    yield from cycle_search(G, s, k, 0)


def main():
    import networkx as nx

    # CYCLE_SEARCH counter-example
    G = nx.parse_adjlist(
        ["a d e", "b d e", "c a b", "d a b", "e c"], create_using=nx.DiGraph
    )
    s = "a"
    k = 5

    cycles_good = list(nx.algorithms.cycles._bounded_cycle_search(G, [s], length_bound=k))
    assert cycles_good == [
        ["a", "d"],
        ["a", "d", "b", "e", "c"],
        ["a", "e", "c"],
        ["a", "e", "c", "b", "d"],
    ]  # missing: None

    cycles_bad = list(CYCLE_SEARCH(G, s, k))
    assert cycles_bad == [
        ["a", "d"],
        ["a", "d", "b", "e", "c"],
        ["a", "e", "c"],
    ]  # missing: ['a', 'e', 'c', 'b', 'd']

    # CYCLE_SEARCH is incomplete: it misses ['a', 'e', 'c', 'b', 'd'].
    # See https://arxiv.org/abs/2512.08392
    missed = [c for c in cycles_good if c not in cycles_bad]
    assert missed == [["a", "e", "c", "b", "d"]], "unexpected result"
    print(f"CYCLE_SEARCH missed {len(missed)} of {len(cycles_good)} cycles: {missed} as expected.")


if __name__ == "__main__":
    main()
