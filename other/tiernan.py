"""
Tiernan's elementary circuit algorithm EC, modern Python transcription of

  author  = {James C. Tiernan},
  title   = {An Efficient Search Algorithm to Find the Elementary Circuits of a Graph},
  journal = {Communications of the ACM},
  volume  = {13}, number = {12}, pages = {722--726}, year = {1970},

Vertices are numbered 0..N-1 in G.nodes order (EC's numbering 1..N is arbitrary).
The N x N arrays G and H of the paper become successor lists and closure sets.
Circuits are yielded as vertex lists without repeating the initial vertex (as in EC and nx.simple_cycles).
"""


def tiernan(G):
    nodes = list(G.nodes)
    N = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    Gamma = [[idx[w] for w in G.successors(v)] for v in nodes]

    for p1 in range(N):                       # EC1 (p1 = 0), EC5 (advance initial vertex)
        P = [p1]
        onP = [False] * N
        onP[p1] = True
        H = [set() for _ in range(N)]         # EC5: H <- 0

        while True:
            last = P[-1]
            # EC2 [Path Extension]: first j in Gamma(last) with j > P[1], j not in P, j not in H[last]
            for j in Gamma[last]:
                if j > p1 and not onP[j] and j not in H[last]:
                    P.append(j)
                    onP[j] = True
                    break
            else:
                # EC3 [Circuit Confirmation]
                if p1 in Gamma[last]:
                    yield [nodes[i] for i in P]
                # EC4 [Vertex Closure]
                if len(P) == 1:
                    break                     # line 7: all circuits through P[1] considered -> EC5
                H[last].clear()               # line 8
                P.pop()
                onP[last] = False
                H[P[-1]].add(last)            # close last to its predecessor on P


def main():
    import networkx as nx

    # Fig. 1/2 of the paper, vertices 1..5; expected circuits [1,2,3,5], [1,2,4,3,5], [2]
    G = nx.DiGraph([(1, 2), (2, 2), (2, 3), (2, 4), (3, 5), (4, 3), (5, 1)])
    cycles = list(tiernan(G))
    print(cycles)
    assert cycles == [[1, 2, 3, 5], [1, 2, 4, 3, 5], [2]]


if __name__ == "__main__":
    main()