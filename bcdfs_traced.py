"""
This code reproduces the execution traces of graph X and Y form the paper.

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
    traces = []
    
    def trace(msg):
        traces.append(f"{''.join(map(str, S)):{len(G.nodes)}} : {' '.join(str(bar[v]) for v in sorted(G.nodes()))} : {'.'*len(S)}{msg}")

    def print_trace():
        print(f"{'\n'.join(traces)}")

    def length(S):
        return len(S) - 1

    def UpdateBarrier(u, l):
        if bar[u] > l:
            trace(f"update({u}) bar[{u}] := {l} (was {bar[u]})")
            bar[u] = l
            for v in G.predecessors(u):
                if v not in S:
                    UpdateBarrier(v, l + 1)

    def search(u):
        trace(f"search({u}) enter")
        F = k + 1
        S.append(u)
        if u == t:
            yield S.copy()
            trace(f"search({u}) output {S}") 
            S.pop()
            F = 0
            return F
        elif length(S) < k:
            for v in G.successors(u):
                if v not in S:
                    if length(S) + bar[v] < k:      # originally: + 1 ... <= k
                        f = yield from search(v)
                        if f != k + 1:
                            F = min(F, f + 1)
                    else:
                        trace(f"search({v}) {v} pruned ({length(S) + bar[v]} >= k)") 
                else:
                    trace(f"search({v}) {v} pruned (in S)") 

        if F == k + 1:
            trace(f"search({u}) fruitless bar[{u}] := {k - length(S) + 1} (was {bar[u]})")
            bar[u] = k - length(S) + 1
        else:
            trace(f"search({u}) fruitful")
            # bar[u] = k + 1
            UpdateBarrier(u, F)
        S.pop()
        trace(f"search({u}) exit") 
        return F

    trace("")
    yield from search(s)
    print_trace()


def main():
    import networkx as nx
    import bsdfs_trivial

    print("### counter-example to BC-DFS's completeness, graph X in the paper ###")
    X = nx.parse_adjlist(
        ["A B C", "B C D E", "C B D", "D B", "E"], create_using=nx.DiGraph
    )
    s = 'A'
    t = 'E'
    k = 4
    got = list(bcdfs(X, s, t, k))
    expected = list(bsdfs_trivial.bsdfs(X, s, t, k))
    assert got == [
        ["A", "B", "E"],
        ["A", "C", "B", "E"],
    ]
    absent = [q for q in expected if q not in got]
    assert absent == [["A", "C", "D", "B", "E"]]
    print(f"counter-example: BC-DFS misses {len(absent)} of {len(expected)} paths: {absent}")

    print("### counter-example to BC-DFS's monotonicity claim, graph Y in the paper ###")
    Y = nx.parse_adjlist(
        ["A D", "B D E F", "C", "D A B C", "E A B D", "F B"], create_using=nx.DiGraph
    )
    s = 'E'
    t = 'C'
    k = 6
    got = list(bcdfs(Y, s, t, k))
    expected = list(bsdfs_trivial.bsdfs(Y, s, t, k))
    assert got == [
        ["E", "A", "D", "C"],
        ["E", "B", "D", "C"],
        ["E", "D", "C"]
    ]
    assert got == expected, "completeness should hold here"
    print("### the end ###")

if __name__ == "__main__":
    main()
