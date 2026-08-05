"""BC-DFS (Peng et al. 2021) -- instrumented with the event callback of
bsdfs_traced.py, so that traces of the two algorithms are directly
comparable ("cost of completeness").

The algorithm is the uploaded bcdfs.py (a faithful translation of the
paper's pseudo-code); `kickstart=True` adds the one-line completeness fix
of bcdfs_kickstart.py: `bar[u] = k+1` immediately before the root
UpdateBarrier call, which forces its guard and thereby makes the origin
assignment unconditional -- the exact mechanism BS-DFS has in Fruitful.
The kick-start line itself is bookkeeping that is immediately overwritten;
it is neither costed nor emitted.

Cost model, aligned with bsdfs_traced (paper, sec:work-attribution):
    search call at u        : 1 (entry) + 1 per scanned successor
                              (u = t and depth-k calls scan nothing)
    UpdateBarrier that fires: 1 + 1 per scanned predecessor
                              (a non-firing call was already paid for by
                              the scan step of its caller)
    producing an output     : |path| <= k+1

Event schema: identical to bsdfs_traced.py.  Correspondences:
    F == sd for every call at u != t (F = min over children of f+1,
        sentinel k+1; the raise value k - h + 1 coincides);
    calls at t are BC-only ('enter'/'output'/'exit' with F = 0) --
        BS-DFS handles t in the parent's scan;
    ('cascade', step, u, F) is emitted whenever the fruitful branch is
        taken; a cascade with NO following root 'dequeue' is a root call
        whose guard bar[u] > F failed -- the omitted origin assignment,
        i.e. the incompleteness mechanism, visible in the trace;
    the root UpdateBarrier's write has kind 'fruitful' (origin
        assignment), recursive writes have kind 'drop'.
"""


def bcdfs_traced(G, s, t, k, emit, kickstart=False):
    """BC-DFS.  Emits events; returns the total step count."""
    bar = {v: 0 for v in G.nodes}
    S = []
    steps = 0

    def update_barrier(u, l, kind):
        nonlocal steps
        if bar[u] > l:
            steps += 1                                              # instr
            emit(("dequeue", steps, u, l))                          # instr
            emit(("write", steps, u, bar[u], l, kind))              # instr
            bar[u] = l
            for v in G.predecessors(u):
                steps += 1                                          # instr
                if v not in S:
                    update_barrier(v, l + 1, "drop")

    def search(u):
        nonlocal steps
        S.append(u)
        h = len(S) - 1
        steps += 1                                                  # instr
        emit(("enter", steps, u, h))                                # instr
        F = k + 1
        if u == t:
            steps += len(S)                                         # instr
            emit(("output", steps, tuple(S)))
            F = 0
        elif h < k:
            for v in G.successors(u):
                steps += 1                                          # instr
                if v not in S:
                    if h + 1 + bar[v] <= k:
                        f = search(v)
                        if f != k + 1:
                            F = min(F, f + 1)
        if u != t:
            if F == k + 1:
                emit(("write", steps, u, bar[u], k - h + 1,         # instr
                      "raise"))                                     # instr
                bar[u] = k - h + 1
            else:
                emit(("cascade", steps, u, F))                      # instr
                if kickstart:
                    bar[u] = k + 1          # kick-start (not costed)
                update_barrier(u, F, "fruitful")
        emit(("exit", steps, u, h, F, F <= k))                      # instr
        S.pop()
        return F

    emit(("start", 0))
    search(s)
    emit(("term", steps))
    return steps
