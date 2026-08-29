"""Random graph families shared by the experiments accompanying the paper.

Every experiment (missed_paths, steps, runtime, delay_bounds) draws its
instances from the same families and sweeps the same hop bounds, so the
figures are directly comparable.  Only instance generation and the panel
grid live here; measurement and plotting stay inside each experiment, so
each one can still be read on its own.

A family is described by four things:

    key       short slug, safe for file names and log columns
    name      display name, used as the panel title, e.g. "Erdos-Renyi"
    generate  generate(run) -> (G, s, t), a pure function of ``run``
    note      the parameters, echoed into the .txt log for the record

Adding a family
---------------
Write a ``gen_xy(run) -> (G, s, t)`` that is seeded from ``run`` alone
(so every experiment sees the same instances) and give it a base seed of
its own.  Append a ``Family`` to FAMILIES -- nothing else has to change:
the experiments loop over FAMILIES, and the figures grow a second row of
panels as soon as there are more than two.

Run this module directly to print the size statistics of every family.
"""

import math
import random
from typing import Callable, NamedTuple

import networkx as nx
import matplotlib.pyplot as plt

import sys

# --- sample budget -----------------------------------------------------
# RUNS   instances drawn per family
# ISLICE cap on the enumerated outputs per instance; instances hitting the
#        cap are truncated and excluded by the experiments

if any("pydevd" in m for m in sys.modules):
    print("### debug mode - reduced data set, for preview only ###")
    ISLICE = 10_000
    RUNS = 100
else:
    ISLICE = 100_000
    RUNS = 1_000

# --- shared experiment grid --------------------------------------------

K_VALUES = range(3, 11)
CMAP = plt.get_cmap("plasma", len(K_VALUES))   # one discrete color per k, shared by scatter and colorbar

COLS = 2            # panels per row; axes[::COLS] is the leftmost panel of each row


class Family(NamedTuple):
    key: str
    name: str
    generate: Callable[[int], tuple]
    note: str


def gen_er(run):
    """Erdos-Renyi G(n,m), small and dense enough to be enumerable."""
    rng = random.Random(42 + run)
    n = rng.randint(6, 30)
    m = int(n * math.exp(rng.uniform(0, math.log(n-1))))
    G = nx.gnm_random_graph(n, m, directed=True, seed=rng)
    s, t = rng.sample(list(G.nodes), 2)
    return G, s, t


def gen_ws(run):
    """Watts-Strogatz small world, bidirected, large and sparse."""
    rng = random.Random(73 + run)
    n = 1000
    d = 6
    p = 0.2
    H = nx.watts_strogatz_graph(n, d, p, seed=rng)
    G = nx.DiGraph(H)
    s, t = rng.sample(list(G.nodes), 2)
    return G, s, t


def gen_ba(run):
    rng = random.Random(11 + run)
    H = nx.barabasi_albert_graph(60, 2, seed=rng)
    G = nx.DiGraph(H)
    s, t = rng.sample(list(G.nodes), 2)
    return G, s, t


def gen_ko(run):
    """k-out digraph: out-degree exactly k, in-degree preferential.  Truly
    directed -- unlike ws and ba, which are undirected graphs read as digraphs."""
    rng = random.Random(131 + run)
    # random_k_out_graph is decorated @np_random_state: it rejects a
    # random.Random and wants a numpy-compatible seed, hence the randrange
    H = nx.random_k_out_graph(n=1000, k=5, alpha=1, self_loops=False,
                              seed=rng.randrange(1 << 30))
    G = nx.DiGraph(H)                   # MultiDiGraph -> collapse parallel edges
    # big = max(nx.strongly_connected_components(G), key=len)
    s, t = rng.sample(list(G), 2)
    return G, s, t


FAMILIES = [
    Family("er", "Erdős–Rényi", gen_er,
           "G(n,m) directed, n in [6,30], m = n*exp(U(0,ln(n-1))), seed 42+run"),
    Family("ws", "Watts–Strogatz", gen_ws,
           "n=1000, d=6, p=0.2, bidirected, seed 73+run"),
    Family("ba", "Barabási–Albert", gen_ba, "n=60, m=2, seed=11+run"),
    Family("ko", "k–out", gen_ko,
           "n=1000, k=5 out-edges/node, in-degree preferential (alpha=1), "
           "seed 131+run")
]


def print_families(title):
    """Header for the .txt log: what was run, on which instances."""
    print(f"# {title}: {RUNS=} {ISLICE=} k={min(K_VALUES)}..{max(K_VALUES)}")
    for f in FAMILIES:
        print(f"#   {f.key:>4}  {f.name}  --  {f.note}")


def make_panels(width=5.9, panel_height=3.0, sharey=True):
    """One panel per family, COLS per row, unused cells hidden.

    Returns (fig, axes) with axes[i] belonging to FAMILIES[i].  The
    leftmost panel of each row -- the one that carries the y label -- is
    axes[::COLS].
    """
    n = len(FAMILIES)
    ncols = min(n, COLS)
    nrows = math.ceil(n / ncols)
    fig, grid = plt.subplots(nrows, ncols, figsize=(width, panel_height * nrows),
                             sharey=sharey, constrained_layout=True, squeeze=False)
    axes = list(grid.ravel())
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def main():
    """Size statistics of every family, as a sanity check on the parameters."""
    for f in FAMILIES:
        sizes = [f.generate(run)[0] for run in range(min(RUNS, 100))]
        ns = [G.number_of_nodes() for G in sizes]
        ms = [G.number_of_edges() for G in sizes]
        print(f"{f.key:>4} {f.name:<16} n {min(ns):6,}..{max(ns):<6,} "
              f"m {min(ms):6,}..{max(ms):<6,}   {f.note}")


if __name__ == "__main__":
    main()
