"""Trace evaluation for bsdfs_traced: boundary pass and delay statistics.

First pass (`boundaries`): given a recorded trace, locate the period
boundaries beta_tau.  For consecutive spines S_tau, S_{tau+1} (output paths
without t; S_{T+1} = () empty by the terminal convention), let j_tau be the
largest index with S_tau[..j] == S_{tau+1}[..j] (j = -1 if none).  Descent
tau ends with the exit of w = S_tau[j_tau + 1] at depth j_tau + 1 -- the
first such exit after o_tau -- or is empty (beta_tau = o_tau) when
j_tau = |S_tau| - 1.  beta_0 = o_0 = start and beta_{T+1} = o_{T+1} = term.

Any experiment-specific evaluation is then a walk over the trace with the
o_tau ('start'/'output'/'term' events) and beta_tau in hand.

`DelayStats` is a streaming emit-target computing the four statistics of
delay_bound_experiments.py without storing the trace, including the
termination event o_{T+1} in the amortized statistic (thm:amortized-delay
quantifies over events; p = 1 and the no-output run stay excluded, being
wc0 with bound 1).
"""


def boundaries(trace):
    """Return [(tau, step, index)] for beta_1 .. beta_T; indices into trace.

    Single pass: descent tau's closing exit is the first exit of
    w = S_tau[j_tau+1] at depth j_tau+1 after o_tau, and it precedes
    o_{tau+1}, so a linear walk suffices.
    """
    spines = [ev[2][:-1] for ev in trace if ev[0] == "output"] + [()]
    res = []
    tau = 0
    target = None                       # (w, depth) of the pending exit
    for i, ev in enumerate(trace):
        if ev[0] == "output":
            assert target is None, f"beta_{tau} not found before o_{tau+1}"
            tau += 1
            S_tau, S_next = spines[tau - 1], spines[tau]
            j = -1
            while (j + 1 < len(S_tau) and j + 1 < len(S_next)
                   and S_tau[j + 1] == S_next[j + 1]):
                j += 1
            if j == len(S_tau) - 1:     # empty descent
                res.append((tau, ev[1], i))
            else:
                target = (S_tau[j + 1], j + 1)
        elif target is not None and ev[0] == "exit" and (ev[2], ev[3]) == target:
            res.append((tau, ev[1], i))
            target = None
    assert target is None, f"beta_{tau}: exit of {target} not found"
    return res


class DelayStats:
    """Streaming emit-target: (p_out, wc0, wcIn, wcT, amP, n_intervals)."""

    def __init__(self, n, m, k):
        self.unit = (k + 1) * (n + m)
        self.p = 0
        self.prev = 0
        self.wc0 = self.wc_in = self.wc_t = self.am_p = 0.0
        self.result = None

    def __call__(self, ev):
        kind = ev[0]
        if kind == "output":
            self.p += 1
            r = (ev[1] - self.prev) / self.unit
            self.prev = ev[1]
            if self.p == 1:
                self.wc0 = r
            else:
                if r > self.wc_in:
                    self.wc_in = r
                q = ev[1] / (self.p * self.unit)
                if q > self.am_p:
                    self.am_p = q
        elif kind == "term":
            r = (ev[1] - self.prev) / self.unit
            if self.p == 0:
                self.wc0 = r
            else:
                self.wc_t = r
                q = ev[1] / ((self.p + 1) * self.unit)
                if q > self.am_p:
                    self.am_p = q
            self.result = (self.p, self.wc0, self.wc_in, self.wc_t,
                           self.am_p, self.p + 1)


def measure_delays_traced(G, s, t, k):
    """Drop-in for delay_bound_experiments.measure_delays; no trace stored."""
    from bsdfs_traced import bsdfs_traced
    ds = DelayStats(G.number_of_nodes(), G.number_of_edges(), k)
    bsdfs_traced(G, s, t, k, ds)
    return ds.result


class Tee:
    """Compose emit-targets: Tee(trace.append, DelayStats(...))."""

    def __init__(self, *targets):
        self.targets = targets

    def __call__(self, ev):
        for f in self.targets:
            f(ev)


if __name__ == "__main__":
    import networkx as nx

    from bsdfs_traced import bsdfs_traced

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
    boundary_events = {index: (tau, step)
                       for tau, step, index in boundaries(trace)}

    print(f"graph edges: {list(G.edges)}")
    print("s='s', t='t', k=4")
    for index, event in enumerate(trace):
        print(event)
        if index in boundary_events:
            tau, step = boundary_events[index]
            print(("boundary", tau, step))
    print(f"steps: {steps}")
