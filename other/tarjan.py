import networkx as nx



def enum_cycles(graph, start):
    """
    Enumerate all elementary cycles in a directed graph that start at `start`.

    graph: dict {v: [neighbors]}
    start: starting vertex
    """
    mark = {v: False for v in graph}
    point_stack = []
    marked_stack = []
    cycles = []

    def backtrack(v, s):
        """
        Returns True if an elementary circuit containing the
        partial path on the stack has been found.
        """
        f = False

        point_stack.append(v)
        mark[v] = True
        marked_stack.append(v)

        print(f"stack: {' '.join(map(str, point_stack))} Visiting {v}")
        for w in graph[v]:
            # Tarjan's rule: only consider w >= s
            if w < s:
                continue

            if w == s:
                # Found a cycle
                cycles.append(point_stack.copy())
                f = True

            elif not mark[w]:
                # Continue DFS
                g = backtrack(w, s)
                f = f or g

        # If a cycle was found, unmark until v
        if f:
            while marked_stack and marked_stack[-1] != v:
                u = marked_stack.pop()
                mark[u] = False

            marked_stack.pop()
            mark[v] = False

        point_stack.pop()
        return f

    # Reset marks
    for v in graph:
        mark[v] = False
    marked_stack.clear()

    # Only run BACKTRACK for the chosen start node
    backtrack(start, start)

    # Cleanup
    while marked_stack:
        u = marked_stack.pop()
        mark[u] = False

    return cycles


# worst case example for Tarjan's algorithm from Johnson's paper
# adapted to generate k^3 paths
def worst_case_cycles(k):
    G = nx.DiGraph()
    for u in range(1, 3*k+3 + 1):
        G.add_node(u)
    for u in range(2, k+1 + 1):
        G.add_edge(1, u)
        G.add_edge(u, k+2)
    for u in range(k+2, 2*k+1 + 1):
        G.add_edge(u, 2*k+2)
    for u in range(k+2, 2*k + 1):
        G.add_edge(u, u+1)
    for u in range(2*k+3, 3*k+2 + 1):
        G.add_edge(2*k+2, u)
        G.add_edge(u, 3*k+3)

    G.add_edge(3*k+3, 2*k+2)
    G.add_edge(2*k+3, k+2)
    G.add_edge(2*k+1, 1)
    nx.drawing.nx_pydot.write_dot(G, f"tarjan_cycles_{k}.dot")
    s = 1
    bound = k + 3
    return G, s, bound


# worst case example for Tarjan's algorithm from Johnson's paper
# adapted to generate k^3 paths
def worst_case_paths(k):
    G = nx.DiGraph()
    for u in range(1, 3*k+3 + 1):
        G.add_node(u)
    for u in range(2, k+1 + 1):
        G.add_edge(1, u)
        G.add_edge(u, k+2)
    for u in range(k+2, 2*k+1 + 1):
        G.add_edge(u, 2*k+2)
    for u in range(k+2, 2*k + 1):
        G.add_edge(u, u+1)
    for u in range(2*k+3, 3*k+2 + 1):
        G.add_edge(2*k+2, u)
        G.add_edge(u, 3*k+3)
    nx.drawing.nx_pydot.write_dot(G, f"tarjan_paths_{k}.dot")
    s = 1
    t = 3*k+3
    bound = 2 + k + 2
    return G, s, t, bound


if __name__ == "__main__":
    k = 10
    G, s, bound = worst_case_cycles(k)
    graph = {v: list(G.successors(v)) for v in G.nodes()}
    cycles = enum_cycles(graph, s)
    print(f"Found {len(cycles)} cycles starting at {s}")
    for cycle in cycles:
        print(cycle)
