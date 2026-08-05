# Legacy Code (v1-era)

These modules backed the **v1** submission or were exploratory tooling used
while developing the proofs. They are kept for provenance and reproducibility
of the v1 numbers; they are **not** used by the v2 experiments and are not
maintained.

For v2, the invariant checking they did by hand is done systematically and
claim-by-claim in [`../validate_delay_bounds_v2/`](../validate_delay_bounds_v2/),
which links each check to the lemma, corollary, or theorem it exercises.

| Module | What it was | Superseded by |
| --- | --- | --- |
| `bsdfs_instrumented.py` | BS-DFS with a step/probe counter | `validate_delay_bounds_v2/bsdfs_traced.py` (`StepCounter`) |
| `bsdfs_assert_delay.py` | BS-DFS with inline delay-bound assertions | `validate_delay_bounds_v2/trace_checks.py` (`worst_case_delay`, `amortized_delay`) |
| `bsdfs_asserts.py` | Ad-hoc invariant assertions during the search | `validate_delay_bounds_v2/trace_checks.py` (full claim-linked check set) |
| `bsdfs_loose_instrumented.py` | Hunt for `S*U*S*` event-structure counter-examples in the loose scheme | `../loose_breaker_step_count.py`, `../loose_breaker_x_assignments.py` |
| `bcdfs_monotonicity.py` | Search for counter-examples to BC-DFS's claimed monotonicity | folded into the paper's discussion of BC-DFS |
| `bcdfs_trace.py` | Side-by-side trace of BC-DFS against BS-DFS | folded into the paper's discussion of BC-DFS |

Note: some of these import top-level modules (e.g. `from bsdfs import bsdfs`),
so run them from the repository root, not from inside this directory:

```
python -m legacy.bcdfs_trace
```
