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

import networkx as nx
from bsdfs import bsdfs


def main():
    
    # X: counter-example to completeness 
    X = nx.parse_adjlist(
        ["A B C", "B C D E", "C B D", "D B", "E"], create_using=nx.DiGraph
    )
    s = 'A'
    t = 'E'
    k = 4
    pathsX = list(bcdfs(X, s, t, k))
    assert pathsX == [
        ["A", "B", "E"],
        ["A", "C", "B", "E"],
    ]  # missing ['A', 'C', 'D', 'B', 'E']

    # Z: counter-example to claimed monotonicity (||S2|| > ||S1||)
    Z = nx.DiGraph()
    # Z.add_edges_from([(0, 3), (1, 3), (1, 4), (1, 5), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 3), (5, 1)])
    Z.add_edges_from([('A', 'D'), ('B', 'D'), ('B', 'E'), ('B', 'F'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('E', 'A'), ('E', 'B'), ('E', 'D'), ('F', 'B')])
    s = 'E'
    t = 'C'
    k = 6
    pathsZ = list(bcdfs(Z, s, t, k))
    print(pathsZ)
    nx.nx_pydot.write_dot(Z, "Z.dot")
    
if __name__ == "__main__":
    main()
