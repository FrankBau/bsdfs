"""BS-DFS (tight scheme) -- reference implementation with event callback.

The algorithm is a line-by-line transcription of Algorithms 1-3 of the paper
(bsdfs.tex, alg:bsdfs / alg:search / alg:fruitful).  Instrumentation is one
`emit(...)` call per event and one `steps += ...` per cost-model site; there
are no mode flags and no guards.  The paper's `Output(path)` line *is* the
output event.  Pass `emit=list.append` of a list to record a trace, or any
streaming evaluator (see trace_eval.py); pass `emit=lambda e: None` for a
counter-only run (the function returns the final step count).

Cost model (paper, sec:work-attribution), counted at exactly these sites:
    Search call at v     : 1 (entry) + 1 per scanned successor
    cascade dequeue at q : 1 (dequeue) + 1 per scanned predecessor
    producing an output  : |path| <= k+1

Event schema -- every event carries the step counter *after* the step it
represents, so all step accounting is derivable from the trace:

    ('start',   0)                            o_0, before any step
    ('enter',   step, v, h)                   after the entry step, h = |S|-1
    ('exit',    step, v, h, sd, fruitful)     before the pop
    ('output',  step, path)                   o_tau, path as a tuple
    ('cascade', step, origin, sd)             a Fruitful(...) call begins
    ('dequeue', step, q, d)                   after the dequeue step
    ('write',   step, x, old, new, kind)      kind in {'raise','fruitful','drop'}
    ('term',    step)                         o_{T+1}, after the last step

The events 'start' and 'term' realize the paper's virtual events o_0 and
o_{T+1}: the externally observable events of a run are exactly the 'start',
'output', and 'term' events of the trace.
"""
from collections import deque


def bsdfs_traced(G, s, t, k, emit):
    """Tight scheme.  Emits events; returns the total step count."""
    b = {x: 0 for x in G.nodes}
    S = []
    steps = 0

    def fruitful(v, sd):
        nonlocal steps
        emit(("cascade", steps, v, sd))                             # instr
        emit(("write", steps, v, b[v], sd, "fruitful"))             # instr
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            steps += 1                                              # instr
            emit(("dequeue", steps, q, d))                          # instr
            for p in G.predecessors(q):
                steps += 1                                          # instr
                if p not in S and b[p] > d + 1:
                    emit(("write", steps, p, b[p], d + 1, "drop"))  # instr
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        nonlocal steps
        S.append(v)
        h = len(S) - 1
        steps += 1                                                  # instr
        emit(("enter", steps, v, h))                                # instr
        sd = k + 1
        for w in G.successors(v):
            steps += 1                                              # instr
            if b[w] + h < k:
                if w == t:
                    steps += h + 2                                  # instr
                    emit(("output", steps, tuple(S) + (t,)))
                    sd = 1
                elif w not in S:
                    sd = min(sd, search(w) + 1)
        if sd <= k:
            fruitful(v, sd)
        else:
            emit(("write", steps, v, b[v], k - h + 1, "raise"))     # instr
            b[v] = k - h + 1
        emit(("exit", steps, v, h, sd, sd <= k))                    # instr
        S.pop()
        return sd

    emit(("start", steps))
    search(s)
    emit(("term", steps))
    return steps


if __name__ == "__main__":
    import networkx as nx

    G = nx.DiGraph()
    G.add_edges_from([
        ("s", "a"), ("s", "b"),
        ("a", "b"), ("a", "c"),
        ("b", "c"), ("b", "d"),
        ("c", "a"), ("c", "d"),
        ("d", "b"), ("d", "t"),
    ])
    trace = []
    steps = bsdfs_traced(G, "s", "t", 4, trace.append)
    print(f"graph edges: {list(G.edges)}")
    print("s='s', t='t', k=4")
    for event in trace:
        print(event)
    print(f"steps: {steps}")
