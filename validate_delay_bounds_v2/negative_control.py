"""Negative control for the trace checker.

A checker that never fails proves nothing.  Each mutant below changes exactly
one line of the algorithm; the run reports which claims detect it.  The mutants
are deliberately mild -- every one of them still enumerates correctly (mutant
'loose' is the complete loose scheme), so an output-only comparison detects
none of them.
"""
import sys
from collections import defaultdict

from campaign_er import sample_instance
from trace_checks import analyze, merge

BASE = open("bsdfs_traced.py").read()

MUTANTS = {
    # cascade resets to 0 instead of depositing d+1  (= loose scheme)
    "loose": [('b[p], d + 1, "drop"))', 'b[p], 0, "drop"))'),
              ("if p not in S and b[p] > d + 1:", "if p not in S and b[p] > 0:"),
              ("                    b[p] = d + 1\n", "                    b[p] = 0\n")],
    # fruitless write one too small: b[v] <- k-h  (breaks pruning strength)
    "raise_off_by_one": [("b[v], k - h + 1, \"raise\"))", "b[v], k - h, \"raise\"))"),
                         ("            b[v] = k - h + 1\n", "            b[v] = k - h\n")],
    # cascade queue processed LIFO instead of FIFO.  MEASURED NON-INERT (write
    # sequences differ in 75/750 instances) but correctly undetected: a cascade
    # runs atomically, so only its fixpoint is observable, and the relaxation
    # fixpoint is order-independent.  Kept as a control on the controls.
    "cascade_lifo": [("q, d = queue.popleft()", "q, d = queue.pop()")],
    # cascade also writes on-path nodes (drops the p not in S guard).
    # MEASURED INERT on ER samples: over 1.8e5 outputs the condition
    # "p in S and b[p] > d+1" never held, so this mutation changes no run.
    "cascade_on_path": [("if p not in S and b[p] > d + 1:", "if b[p] > d + 1:")],
}


def build(name):
    src = BASE
    for old, new in MUTANTS[name]:
        assert old in src, (name, old)
        src = src.replace(old, new)
    ns = {"__name__": "mutant_" + name}
    exec(compile(src, "mutant_" + name, "exec"), ns)
    return ns


def run(name, ns, runs=120, limit=500):
    total = {}
    outputs = 0
    errors = 0
    for n in ns:
        for r in range(runs):
            G, s, t, k = sample_instance(n, r)
            try:
                paths, trace, counter, capped = MOD["run_traced"](G, s, t, k, limit=limit)
                res = analyze(G, s, t, k, trace, counter, capped)
            except Exception as exc:                 # analyzer refuses the trace
                errors += 1
                continue
            outputs += len(paths)
            merge(total, res)
    fired = {nm: c for nm, c in total.items() if c.violations}
    print(f"\nmutant '{name}':  outputs={outputs:,}  analyzer aborts={errors}")
    if not fired:
        print("  NOT DETECTED")
    for nm, c in sorted(fired.items(), key=lambda kv: -kv[1].violations):
        print(f"  {nm:<24} viol={c.violations:<8,} tested={c.tested:,}")
    return fired


if __name__ == "__main__":
    ns = [7, 9, 11, 13]
    for name in MUTANTS:
        MOD = build(name)
        run(name, ns)
