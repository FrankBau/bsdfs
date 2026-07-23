# search core, copilot generated
# is it really Tiernan???

# there is no length bound here

def simple_cycles_tiernan(graph):
    """
    graph: dict {v: [neighbors]}
    yields: each simple directed cycle as a list of vertices
    """

    vertices = list(graph.keys())

    # We iterate over each vertex as a potential cycle start
    for start in vertices:
        stack = [start]          # current DFS path
        blocked = {start}        # vertices currently on the path

        def dfs(v):
            for w in graph[v]:
                # Case 1: we found a cycle back to start
                if w == start:
                    yield stack[:]  # copy of current path
                    continue

                # Case 2: continue DFS if w not already on path
                if w not in blocked:
                    blocked.add(w)
                    stack.append(w)

                    yield from dfs(w)

                    # backtrack
                    stack.pop()
                    blocked.remove(w)

        # run DFS from start
        yield from dfs(start)

        # After finishing all cycles starting at `start`,
        # conceptually remove `start` from the graph
        for v in graph:
            graph[v] = [w for w in graph[v] if w != start]
