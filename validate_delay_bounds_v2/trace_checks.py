"""Post-hoc analyzer for BS-DFS event traces (bsdfs_traced.py).

The analyzer never reads algorithm state: it mirrors the search path from
'enter'/'exit' events and the barrier array from 'write' events, and it
re-derives the period/phase/interval segmentation from the recorded outputs.
Every check reports (tested, violations), where `tested` counts the instances
in which the claim's hypothesis was actually exercised -- a check that comes
back 0/0 is vacuous, not confirmed.

Status tags follow the paper: 'proven' (statement in bsdfs.tex),
'derived' (follows from paper lemmas, adversarial review pending),
'conjecture' (no proof), 'self' (analyzer/harness self-consistency).

Segmentation (bsdfs.tex, sec:periods):
    outputs      o_1 .. o_T
    lca          j_tau  = |longest common prefix of S_tau, S_{tau+1}| - 1
    boundary     beta_tau = return of Search(w_tau), w_tau = S_tau[j_tau + 1],
                 or beta_tau = o_tau when j_tau = l_tau - 1 (empty descent);
                 beta_0 = start, beta_T = end of execution
    ascent tau   = (beta_{tau-1}, o_tau]      "Phase A"
    descent tau  = (o_tau, beta_tau]          "Phase B"
    period tau   = ascent tau + descent tau
    interval 0   = (beta_0, o_1];  interval tau = (o_tau, o_{tau+1}];
    interval T   = (o_T, beta_T]

All segments are half-open on the left and closed on the right (bsdfs.tex,
sec:periods; figure caption "the excluded left endpoint").  Consequently the
delay of an interval includes the steps producing its CLOSING output and
excludes those of the output that opens it: the terminal interval has no
closing output and pays none, the initial interval pays for o_1.  This is the
only convention implemented -- bsdfs_traced.py charges |path| steps before
emitting the 'output' event, so step_at(o_tau) already contains them, and
every delay below is a difference of two such step values.
"""
from dataclasses import dataclass, field


# ----------------------------------------------------------------- results

@dataclass
class Check:
    name: str
    label: str
    status: str
    tested: int = 0
    violations: int = 0
    examples: list = field(default_factory=list)

    def fail(self, **info):
        self.violations += 1
        if len(self.examples) < 3:
            self.examples.append(info)

    def __str__(self):
        flag = "FAIL" if self.violations else ("vac " if self.tested == 0 else "ok  ")
        return (f"[{flag}] {self.name:<26} {self.label:<28} {self.status:<10} "
                f"tested={self.tested:<9,} viol={self.violations}")


# ------------------------------------------------------------ segmentation

class Segmentation:
    """Derives periods, phases and intervals from a trace."""

    def __init__(self, trace, capped):
        self.trace = trace
        self.capped = capped
        self.out_pos = [i for i, e in enumerate(trace) if e[0] == "output"]
        self.paths = [list(trace[i][2]) for i in self.out_pos]
        self.T = len(self.out_pos)

        # A capped run is truncated mid-execution: everything after the last
        # output is a fragment of an unfinished interval and is discarded.
        self.end = self.out_pos[-1] if (capped and self.T) else len(trace) - 1

        # lca indices j_tau, tau = 1 .. T-1  (self.j[tau])
        self.j = {}
        for tau in range(1, self.T):
            A, B = self.paths[tau - 1], self.paths[tau]
            jj = 0
            while jj + 1 < len(A) and jj + 1 < len(B) and A[jj + 1] == B[jj + 1]:
                jj += 1
            self.j[tau] = jj

        # beta_tau
        self.beta = {0: -1}
        for tau in range(1, self.T):
            o = self.out_pos[tau - 1]
            spine, jj = self.paths[tau - 1], self.j[tau]
            if jj == len(spine) - 2:           # w_tau = t: empty descent
                self.beta[tau] = o
                continue
            w, hw = spine[jj + 1], jj + 1
            b = None
            for i in range(o + 1, self.end + 1):
                e = trace[i]
                if e[0] == "exit" and e[2] == w and e[3] == hw:
                    b = i
                    break
            if b is None:
                raise AssertionError(f"beta_{tau}: no exit of w={w} at h={hw}")
            self.beta[tau] = b
        # last usable period boundary
        self.last_tau = self.T - 1 if capped else self.T
        if not capped and self.T:
            self.beta[self.T] = self.end

        # phases: list of (kind, tau, lo_exclusive, hi_inclusive)
        self.phases = []
        if self.T == 0:                                   # fruitless run
            self.phases.append(("A", 1, -1, self.end))
        else:
            for tau in range(1, self.last_tau + 1):
                self.phases.append(("A", tau, self.beta[tau - 1], self.out_pos[tau - 1]))
                self.phases.append(("B", tau, self.out_pos[tau - 1], self.beta[tau]))

        # intervals: list of (idx, lo_exclusive, hi_inclusive)
        self.intervals = []
        if self.T == 0:
            self.intervals.append((0, -1, self.end))
        else:
            self.intervals.append((0, -1, self.out_pos[0]))
            for tau in range(1, self.T):
                self.intervals.append((tau, self.out_pos[tau - 1], self.out_pos[tau]))
            if not capped:
                self.intervals.append((self.T, self.out_pos[-1], self.end))

        # per-event lookup tables
        n = len(trace)
        self.ph_of = [None] * n
        self.iv_of = [None] * n
        for p, (_kind, _tau, lo, hi) in enumerate(self.phases):
            for i in range(lo + 1, hi + 1):
                self.ph_of[i] = p
        for idx, lo, hi in self.intervals:
            for i in range(lo + 1, hi + 1):
                self.iv_of[i] = idx

    def spine(self, tau):
        return self.paths[tau - 1]

    def phase_kind(self, p):
        return None if p is None else self.phases[p][0]

    def phase_tau(self, p):
        return None if p is None else self.phases[p][1]


# ---------------------------------------------------------------- analysis

def analyze(G, s, t, k, trace, counter, capped, checks=None):
    seg = Segmentation(trace, capped)
    C = {}

    def mk(name, label, status):
        C[name] = Check(name, label, status)
        return C[name]

    # self-consistency of the harness
    c_stack = mk("stack_nesting", "-", "self")
    c_mirror = mk("barrier_mirror", "-", "self")
    c_cost = mk("cost_model", "sec:work-attribution", "self")
    # conformance of the implementation with the pseudocode
    c_guard = mk("entry_guard", "alg:search", "proven")
    c_wval = mk("write_values", "alg:search/alg:fruitful", "proven")
    c_range = mk("barrier_range", "cor:barrier-upper", "proven")
    c_mono = mk("write_monotonicity", "lem:fruitless-increasing", "proven")
    # structure
    c_purity = mk("ascent_purity", "lem:phase-a-purity", "proven")
    c_cp1 = mk("call_phase_1", "cor:call-phase(1)", "proven")
    c_cp2 = mk("call_phase_2", "cor:call-phase(2)", "proven")
    c_cp3 = mk("call_phase_3", "cor:call-phase(3)", "proven")
    c_orig = mk("descent_origins", "lem:spine-tail-origins", "proven")
    c_seal = mk("descent_seal", "lem:period-seal", "proven")
    c_ds1 = mk("descent_stack_1", "lem:descent-stack(1)", "proven")
    c_ds2 = mk("descent_stack_2", "lem:descent-stack(2)", "proven")
    c_rchain = mk("raise_chain", "lem:raise-bound", "proven")
    c_dchain = mk("drop_chain", "lem:drop-bound", "proven")
    c_block = mk("block_sequence", "cor:block-sequence", "proven")
    c_two = mk("two_entries_interval", "cor:two-entries", "proven")
    c_one = mk("one_entry_phase", "(new)", "derived")
    c_split = mk("two_entries_split", "(new, corollary)", "derived")
    # delay
    c_wcd = mk("worst_case_delay", "thm:worst-case-delay", "proven")
    c_init = mk("initial_interval", "cor:initial-interval", "proven")
    c_amort = mk("amortized_delay", "thm:amortized-delay", "proven")
    c_period = mk("period_budget", "thm:amortized-delay(pf)", "derived")

    n, m = G.number_of_nodes(), G.number_of_edges()
    outdeg = {v: G.out_degree(v) for v in G.nodes}
    indeg = {v: G.in_degree(v) for v in G.nodes}

    # ---- pass 1: calls (enter/exit pairing) ------------------------------
    calls = []            # (enter_idx, exit_idx, v, h, sd, fruitful)
    stack = []
    for i, e in enumerate(trace):
        if e[0] == "enter":
            _, _st, v, h = e
            c_stack.tested += 1
            if h != len(stack):
                c_stack.fail(i=i, v=v, h=h, depth=len(stack))
            stack.append((i, v, h))
        elif e[0] == "exit":
            _, _st, v, h, sd, fr = e
            if not stack or stack[-1][1] != v or stack[-1][2] != h:
                c_stack.fail(i=i, v=v, h=h, top=stack[-1] if stack else None)
                continue
            ei, _, _ = stack.pop()
            calls.append((ei, i, v, h, sd, fr))
    call_of_enter = {c[0]: c for c in calls}

    # ---- pass 2: streaming walk -----------------------------------------
    bmir = {x: 0 for x in G.nodes}
    stack = []                       # (enter_idx, v, h)
    last_d = None                    # d of the most recent dequeue
    pend_cascade = None              # (origin, sd) awaiting its 'fruitful' write
    dropped = set()                  # nodes dropped since the current o_tau
    cur_phase = None
    phase_entries = {}               # (v,h) -> count, per phase
    iv_entries = {}                  # (v,h) -> [phase ids], per interval
    iv_writes = {}                   # node -> "RD..." string, per interval
    cur_iv = None
    fruitful_in_descent = {}         # node -> count, per descent
    chain_r = {}                     # node -> current raise-run length
    chain_d = {}                     # node -> current drop-run length
    raises_at_s = 0
    n_calls = 0
    n_deq = 0
    out_steps = 0

    def close_interval():
        for pair, phs in iv_entries.items():
            if len(phs) >= 2:
                c_two.tested += 1
                if len(phs) > 2:
                    c_two.fail(pair=pair, entries=len(phs))
                c_split.tested += 1
                if len(set(phs)) != len(phs):
                    c_split.fail(pair=pair, phases=phs)
        for node, seq in iv_writes.items():
            c_block.tested += 1
            # R* D* R*  with |R1|<=k, |D|<=k-1, |R2|<=k
            i2 = 0
            r1 = 0
            while i2 < len(seq) and seq[i2] == "R":
                r1 += 1
                i2 += 1
            dd = 0
            while i2 < len(seq) and seq[i2] == "D":
                dd += 1
                i2 += 1
            r2 = 0
            while i2 < len(seq) and seq[i2] == "R":
                r2 += 1
                i2 += 1
            if i2 != len(seq) or r1 > k or dd > k - 1 or r2 > k:
                c_block.fail(node=node, seq=seq[:40], k=k)
        iv_entries.clear()
        iv_writes.clear()

    for i, e in enumerate(trace):
        if i > seg.end:
            break
        ph = seg.ph_of[i]
        iv = seg.iv_of[i]
        kind = seg.phase_kind(ph)
        tau = seg.phase_tau(ph)

        if iv != cur_iv:
            if cur_iv is not None:
                close_interval()
            cur_iv = iv
        if ph != cur_phase:
            phase_entries = {}
            cur_phase = ph
            if kind == "B":
                dropped = set()
                fruitful_in_descent = {}

        typ = e[0]

        if typ == "enter":
            _, _st, v, h = e
            n_calls += 1
            # entry guard: b[v] + (h-1) < k, i.e. b[v] <= k - h
            if h > 0:
                c_guard.tested += 1
                if bmir[v] > k - h:
                    c_guard.fail(i=i, v=v, h=h, b=bmir[v], k=k)
            # one entry per pair per phase
            key = (v, h)
            cnt = phase_entries.get(key, 0) + 1
            phase_entries[key] = cnt
            c_one.tested += 1
            if cnt > 1:
                c_one.fail(i=i, pair=key, phase=(kind, tau), count=cnt)
            iv_entries.setdefault(key, []).append(ph)
            # descent seal
            if kind == "B":
                if dropped:
                    c_seal.tested += 1
                if v in dropped:
                    c_seal.fail(i=i, v=v, tau=tau, h=h)
            stack.append((i, v, h))
            if kind == "B":
                _check_descent_stack(seg, c_ds2, stack, i, tau, trace)

        elif typ == "exit":
            _, _st, v, h, sd, fr = e
            if stack:
                stack.pop()
            if kind == "B":
                if fr:
                    c_orig.tested += 1
                    spine = seg.spine(tau)
                    live = [f[1] for f in stack] + [v]
                    ok = (h < len(spine) - 1 and spine[:h + 1] == list(live)
                          and (tau == seg.T or h > seg.j.get(tau, -1)))
                    if not ok:
                        c_orig.fail(i=i, v=v, h=h, tau=tau, live=live, spine=spine)
                    fruitful_in_descent[v] = fruitful_in_descent.get(v, 0) + 1
                    if fruitful_in_descent[v] > 1:
                        c_orig.fail(i=i, v=v, tau=tau, reason="two fruitful returns")
                # at beta_tau itself w_tau has just been popped and the path
                # is (v_0..v_j); that instant is governed by descent_stack_1
                if stack and seg.beta.get(tau) != i:
                    _check_descent_stack(seg, c_ds2, stack, i, tau, trace)
            if kind == "A" and fr:
                c_purity.fail(i=i, v=v, reason="fruitful return in ascent")
            # descent_stack(1): stack right after beta_tau
            if tau is not None and seg.beta.get(tau) == i and tau < seg.T:
                c_ds1.tested += 1
                spine = seg.spine(tau)
                jj = seg.j.get(tau)
                if jj is not None:
                    want = spine[:jj + 1]
                    got = [f[1] for f in stack]
                    if got != want:
                        c_ds1.fail(tau=tau, got=got, want=want)

        elif typ == "output":
            out_steps += len(e[2])

        elif typ == "cascade":
            _, _st, origin, sd = e
            pend_cascade = (origin, sd)
            if kind == "A":
                c_purity.fail(i=i, reason="cascade in ascent")

        elif typ == "dequeue":
            n_deq += 1
            last_d = e[3]
            if kind == "A":
                c_purity.fail(i=i, reason="dequeue in ascent")

        elif typ == "write":
            _, _st, x, old, new, wkind = e
            c_mirror.tested += 1
            if bmir[x] != old:
                c_mirror.fail(i=i, x=x, recorded=old, mirrored=bmir[x])
            c_wval.tested += 1
            c_mono.tested += 1
            if wkind == "raise":
                h = stack[-1][2] if stack else None
                if new != k - h + 1:
                    c_wval.fail(i=i, x=x, new=new, want=k - h + 1, kind=wkind)
                if new <= old:
                    c_mono.fail(i=i, x=x, old=old, new=new, kind=wkind)
                if x == s:
                    raises_at_s += 1
                chain_r[x] = chain_r.get(x, 0) + 1
                c_rchain.tested += 1
                if chain_r[x] > (1 if x == s else k):
                    c_rchain.fail(x=x, run=chain_r[x], k=k)
                chain_d[x] = 0
                if iv is not None:
                    iv_writes[x] = iv_writes.get(x, "") + "R"
            elif wkind == "fruitful":
                if pend_cascade is None or pend_cascade[0] != x or pend_cascade[1] != new:
                    c_wval.fail(i=i, x=x, new=new, pend=pend_cascade, kind=wkind)
                if new < old:
                    c_mono.fail(i=i, x=x, old=old, new=new, kind=wkind)
                chain_d[x] = 0
            elif wkind == "drop":
                if last_d is None or new != last_d + 1:
                    c_wval.fail(i=i, x=x, new=new, last_d=last_d, kind=wkind)
                if new >= old:
                    c_mono.fail(i=i, x=x, old=old, new=new, kind=wkind)
                if x == s:
                    c_dchain.fail(x=x, reason="drop at s")
                chain_d[x] = chain_d.get(x, 0) + 1
                c_dchain.tested += 1
                if chain_d[x] > k - 1:
                    c_dchain.fail(x=x, run=chain_d[x], k=k)
                chain_r[x] = 0
                dropped.add(x)
                if kind == "A":
                    c_purity.fail(i=i, x=x, reason="drop in ascent")
                if iv is not None:
                    iv_writes[x] = iv_writes.get(x, "") + "D"
            bmir[x] = new
            c_range.tested += 1
            if not (0 <= new <= k) and not (x == s and new == k + 1):
                c_range.fail(i=i, x=x, new=new, k=k)

    if cur_iv is not None:
        close_interval()
    c_purity.tested = sum(1 for p in seg.phases if p[0] == "A")

    # ---- call/phase correspondence --------------------------------------
    for (ei, xi, v, h, sd, fr) in calls:
        if ei > seg.end or xi > seg.end:
            continue
        pe, px = seg.ph_of[ei], seg.ph_of[xi]
        if pe is None or px is None:
            continue
        if seg.phase_kind(pe) == "B":
            c_cp1.tested += 1
            if px != pe or fr:
                c_cp1.fail(v=v, h=h, enter_phase=seg.phases[pe][:2],
                           exit_phase=seg.phases[px][:2], fruitful=fr)
        if not fr:
            c_cp2.tested += 1
            if pe != px:
                c_cp2.fail(v=v, h=h, enter_phase=seg.phases[pe][:2],
                           exit_phase=seg.phases[px][:2])
        else:
            c_cp3.tested += 1
            if seg.phase_kind(pe) != "A" or seg.phase_kind(px) != "B":
                c_cp3.fail(v=v, h=h, enter_phase=seg.phases[pe][:2],
                           exit_phase=seg.phases[px][:2])

    # ---- cost model ------------------------------------------------------
    if not capped:
        recomputed = (sum(1 + outdeg[c[2]] for c in calls)
                      + sum(1 + indeg[e[2]] for e in trace if e[0] == "dequeue")
                      + out_steps)
        c_cost.tested += 1
        if recomputed != counter.steps:
            c_cost.fail(recomputed=recomputed, counted=counter.steps)

    # ---- delay claims ----------------------------------------------------
    wc_bound = 3 * (k + 1) * (n + m)
    init_bound = (k + 1) * (n + m)
    amort_bound = 2 * (k + 1) * (n + m)

    step_at = [0] + [trace[i][1] for i in seg.out_pos]      # step_at[p] at o_p
    final_step = counter.steps if not capped else trace[seg.end][1]

    for idx, lo, hi in seg.intervals:
        start = 0 if lo < 0 else trace[lo][1]
        end_step = trace[hi][1] if hi < len(trace) else final_step
        if idx == seg.T and not capped:
            end_step = final_step
        delay = end_step - start
        if idx == 0:
            c_init.tested += 1
            if delay > init_bound:
                c_init.fail(delay=delay, bound=init_bound, k=k, n=n, m=m)
        c_wcd.tested += 1
        if delay > wc_bound:
            c_wcd.fail(interval=idx, delay=delay, bound=wc_bound, k=k, n=n, m=m)

    for p in range(1, seg.T + 1):
        if capped and p > seg.T:
            continue
        c_amort.tested += 1
        if step_at[p] > amort_bound * p:
            c_amort.fail(p=p, steps=step_at[p], bound=amort_bound * p)

    for tau in range(1, seg.last_tau + 1):
        lo, hi = seg.beta[tau - 1], seg.beta[tau]
        start = 0 if lo < 0 else trace[lo][1]
        end_step = final_step if hi >= len(trace) else trace[hi][1]
        c_period.tested += 1
        if end_step - start > amort_bound:
            c_period.fail(tau=tau, steps=end_step - start, bound=amort_bound)

    if checks is not None:
        return {name: c for name, c in C.items() if name in checks}
    return C


def _check_descent_stack(seg, chk, stack, i, tau, trace):
    """lem:descent-stack(2): during descent tau the search path is a prefix of
    S_tau (up to the deepest frame open at o_tau, index >= j_tau+1) followed by
    frames entered after o_tau."""
    if tau is None or tau > seg.T:
        return
    o = seg.out_pos[tau - 1]
    spine = seg.spine(tau)
    top_old = 0
    while top_old < len(stack) and stack[top_old][0] <= o:
        top_old += 1
    idx = top_old - 1                        # deepest frame open at o_tau
    chk.tested += 1
    if idx < 0:
        if tau < seg.T:
            chk.fail(i=i, tau=tau, reason="no spine frame open")
        return
    if [f[1] for f in stack[:top_old]] != spine[:idx + 1]:
        chk.fail(i=i, tau=tau, got=[f[1] for f in stack[:top_old]],
                 want=spine[:idx + 1])
        return
    if tau < seg.T and idx < seg.j[tau] + 1:
        chk.fail(i=i, tau=tau, i_deepest=idx, j=seg.j[tau])


def merge(total, part):
    for name, c in part.items():
        if name not in total:
            total[name] = Check(c.name, c.label, c.status)
        tc = total[name]
        tc.tested += c.tested
        tc.violations += c.violations
        for ex in c.examples:
            if len(tc.examples) < 3:
                tc.examples.append(ex)
    return total
