"""BS-DFS (tight scheme) -- reference implementation with optional instrumentation.

The algorithm below is a line-by-line transcription of Algorithms 1-3 of the
paper (bsdfs.tex, alg:bsdfs / alg:search / alg:fruitful).  Every line that is
NOT part of the algorithm is marked `# instr`.  With `counter=None, trace=None`
the instrumentation is two `is not None` tests per event and nothing else, so
the same function serves the large counter-only campaigns and the traced runs.

Cost model (paper, sec:work-attribution):
    search call at v      : 1 + |suc(v)|      steps
    cascade dequeue at q  : 1 + |pre(q)|      steps
    producing an output   : |path| (<= k+1)   steps

Event schema (all events carry the step counter value *after* the step they
represent, so any step-accounting claim can be re-derived from the trace):

    ('enter',   step, v, h)                   after the push, h = ||S||
    ('exit',    step, v, h, sd, fruitful)     before the pop
    ('output',  step, path)                   path as a tuple
    ('cascade', step, origin, sd)             a Fruitful(...) call begins
    ('dequeue', step, q, d)                   a pair is dequeued
    ('write',   step, x, old, new, kind)      kind in {'raise','fruitful','drop'}

Write kinds: 'raise'    -- fruitless assignment b[v] <- k-h+1
             'fruitful' -- origin assignment  b[v] <- sd  in Fruitful
             'drop'     -- cascade assignment b[p] <- d+1
"""
from collections import deque


class StepCounter:
    """Counts only real-algorithm elementary steps."""
    __slots__ = ("steps",)

    def __init__(self):
        self.steps = 0


def bsdfs(G, s, t, k, counter=None, trace=None):
    """Tight scheme.  Generator yielding each simple s-t path of length <= k."""
    b = {x: 0 for x in G.nodes}
    S = []
    cnt = counter                                                      # instr
    tr = trace                                                         # instr

    def fruitful(v, sd):
        if tr is not None:                                             # instr
            tr.append(("cascade", cnt.steps, v, sd))                   # instr
            tr.append(("write", cnt.steps, v, b[v], sd, "fruitful"))   # instr
        b[v] = sd
        queue = deque([(v, sd)])
        while queue:
            q, d = queue.popleft()
            if cnt is not None:                                        # instr
                cnt.steps += 1                                         # instr
                if tr is not None:                                     # instr
                    tr.append(("dequeue", cnt.steps, q, d))            # instr
            for p in G.predecessors(q):
                if cnt is not None:                                    # instr
                    cnt.steps += 1                                     # instr
                if p not in S and b[p] > d + 1:
                    if tr is not None:                                 # instr
                        tr.append(("write", cnt.steps, p,              # instr
                                   b[p], d + 1, "drop"))               # instr
                    b[p] = d + 1
                    queue.append((p, d + 1))

    def search(v):
        S.append(v)
        h = len(S) - 1                                  # h = ||S||
        if cnt is not None:                                            # instr
            cnt.steps += 1                                             # instr
            if tr is not None:                                         # instr
                tr.append(("enter", cnt.steps, v, h))                  # instr
        sd = k + 1
        for w in G.successors(v):
            if cnt is not None:                                        # instr
                cnt.steps += 1                                         # instr
            if b[w] + h < k:
                if w == t:
                    path = tuple(S) + (t,)
                    if cnt is not None:                                # instr
                        cnt.steps += len(path)                         # instr
                        if tr is not None:                             # instr
                            tr.append(("output", cnt.steps, path))     # instr
                    yield list(path)
                    sd = 1
                elif w not in S:
                    d = yield from search(w)
                    sd = min(sd, d + 1)

        if sd <= k:
            fruitful(v, sd)
        else:
            if tr is not None:                                         # instr
                tr.append(("write", cnt.steps, v,                      # instr
                           b[v], k - h + 1, "raise"))                  # instr
            b[v] = k - h + 1

        if tr is not None:                                             # instr
            tr.append(("exit", cnt.steps, v, h, sd, sd <= k))          # instr
        S.pop()
        return sd

    yield from search(s)


def run_traced(G, s, t, k, limit=None):
    """Run once, returning (paths, trace, counter, capped)."""
    from itertools import islice
    counter = StepCounter()
    trace = []
    gen = bsdfs(G, s, t, k, counter=counter, trace=trace)
    if limit is None:
        paths = list(gen)
        capped = False
    else:
        paths = list(islice(gen, limit))
        capped = len(paths) == limit
    return paths, trace, counter, capped
