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
    """original control-flow, completeness NOT guaranteed"""
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


def main():
    import random

    import networkx as nx

    import bsdfs_trivial

    # smallest counter-example to BC-DFS's completeness, the graph traced
    # in legacy/bcdfs_trace.py
    G = nx.parse_adjlist(
        ["a b c", "b c d e", "c b d", "d b", "e"], create_using=nx.DiGraph
    )
    got = list(map(list, bcdfs(G, s="a", t="e", k=4)))
    expected = list(map(list, bsdfs_trivial.bsdfs(G, "a", "e", 4)))
    assert got == [
        ["a", "b", "e"],
        ["a", "c", "b", "e"],
    ]
    absent = [q for q in expected if q not in got]
    assert absent == [["a", "c", "d", "b", "e"]]
    print(f"counter-example: BC-DFS misses {len(absent)} of {len(expected)} paths: {absent}")

    # the same incompleteness at scale: BC-DFS stays sound, but misses
    # paths on a noticeable fraction of random instances, so the missed
    # paths are counted and reported instead of asserted away

    RUNS = 10_000

    for n in range(2, 8):
        found = missed = bad_instances = 0
        for run in range(RUNS):
            rng = random.Random(42 + run)
            m = rng.randint(0, n * (n - 1))
            k = max(rng.getrandbits(n).bit_count(), 1)  # binomial distribution
            H = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            s, t = rng.sample(range(n), 2)

            got = list(map(list, bcdfs(H, s, t, k)))
            expected = list(map(list, bsdfs_trivial.bsdfs(H, s, t, k)))
            where = f"{s=} {t=} {k=} edges={sorted(H.edges)}"

            # soundness is never relaxed: a spurious path would be a failure
            spurious = [q for q in got if q not in expected]
            assert not spurious, f"spurious path {spurious[0]}: {where}"

            absent = [q for q in expected if q not in got]
            found += len(got)
            missed += len(absent)
            bad_instances += bool(absent)

        print(
            f"n={n}  {RUNS:,} random digraphs sound, {found:,} paths, "
            f"{missed:,} missed on {bad_instances:,} digraphs"
        )


if __name__ == "__main__":
    main()
