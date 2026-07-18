import networkx as nx
import random
import time
import math
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import islice

import adversarial_graphs

from bsdfs_nx import bsdfs_bounded_cycle_search
from bsdfs_lazy import bsdfs as bsdfs_lazy
from bsdfs_loose import bsdfs as bsdfs_loose
from networkx.algorithms.cycles import _bounded_cycle_search

ALGOS = {
    "bounded_cycle_search":  lambda G, s, k: _bounded_cycle_search(G, [s], k),
    "bsdfs": lambda G, s, k: bsdfs_bounded_cycle_search(G, [s], k),
    "bsdfs_loose": lambda G, s, k: bsdfs_loose(G, s, s, k),
    "bsdfs_lazy": lambda G, s, k: bsdfs_lazy(G, s, s, k),
}

algo1_name = "bsdfs"
algo2_name = "bounded_cycle_search"
# algo2_name = "bsdfs_loose"
# algo2_name = "bsdfs_lazy"


seed = 42
random.seed(seed)


def ring_family(n_values, r, seeds=(0,)):
    """n_values should be multiples of r"""
    for n in n_values:
        G = adversarial_graphs.ring(n, r)
        k_min = n // r
        k_max = n // r + 1
        for k in range(k_min, k_max+1):
            yield f"ring/n={n}/r={r}/k={k}", {"n": n, "r": r}, (G, 0, k)
            
            
def diamond_chain_family(t_values, seeds=(0,)):
    for t in t_values:
        for s in seeds:                 # seeds unused for deterministic gadgets
            G, s, k = adversarial_graphs.diamond_chain(t)
            yield f"diamond_chain/t={t}/seed={s}", {"t": t}, (G, s, k)


def wave_gadget_family(t_values, seeds=(0,)):
    for t in t_values:
        for s in seeds:                 # seeds unused for deterministic gadgets
            G, s, k = adversarial_graphs.wave_gadget(t)
            yield f"wave_gadget/t={t}/seed={s}", {"t": t}, (G, s, k)


def er_gnm_family(n_values, seeds):
    for n in n_values:
        for seed in seeds:
            rng = random.Random(seed)
            m = int(n * math.exp(rng.uniform(0, math.log(n - 1))))
            G = nx.gnm_random_graph(n, m, seed=rng, directed=True)
            s = rng.choice(list(G.nodes))
            k = rng.randrange(1, n)
            yield f"er/n={n}/m={m}/seed={seed}", {"n": n, "m": m}, (G, s, k)
            

def large_er_family(np_values, k_values, seeds):
    for n, p in np_values:
            for seed in seeds:
                rng = random.Random(seed)
                G = nx.erdos_renyi_graph(n, p, seed=rng, directed=True)
                s = rng.choice(list(G.nodes))
                for k in k_values:
                    yield f"large_er/n={n}/p={p}/k={k}/seed={seed}", {"n": n, "p": p}, (G, s, k)


def kleinberg_family(n_values, seeds):
    for n in n_values:
        for seed in seeds:
            rng = random.Random(seed)
            p = 1   # node has a directed edge to every other node within lattice distance — these are its local contacts
            q = 1   # construct q directed edges from to other nodes (the long-range contacts) using independent random trials;
            r = 2   # u has endpoint v with probability proportional to grid_dist(u, v)^{-r}
            dim = 2
            G = nx.navigable_small_world_graph(n, p, q, r, dim, seed=rng)
            # Remove all self-loops
            G.remove_edges_from(nx.selfloop_edges(G))            
            s, t = rng.sample(list(G.nodes), 2)  # target node not used 
            k = rng.randrange(1, n)
            yield f"kleinberg/n={n}/seed={seed}", {"n": n}, (G, s, k)
            

def watts_strogatz_family(n_values, k_values, d=6, p=0.2, seeds=(0,)):
    for n in n_values:
        for seed in seeds:
            for k in k_values:
                rng = random.Random(seed)
                H = nx.watts_strogatz_graph(n, d, p, seed=rng)
                G =nx.DiGraph(H)
                s, t = rng.sample(list(G.nodes), 2)  # target node not used 
                yield f"watts_strogatz/n={n}/k={k}/seed={seed}", {"n": n}, (G, s, k)


def dag_backedge_family(n_values, beta_values, p=None, seeds=range(10)):
    for n in n_values:
        for beta in beta_values:
            for seed in seeds:
                rng = random.Random(seed)
                pp = p if p is not None else 2 / n   # ~2 forward edges/node if unset
                G = adversarial_graphs.dag_plus_backedges(n, pp, beta, rng)
                s = rng.choice(list(G.nodes))
                k = rng.randrange(n // 4, n)           # large enough to see deep cones
                yield (f"dag_back/n={n}/beta={beta}/seed={seed}",
                       {"n": n, "beta": beta}, (G, s, k))


import statistics
from collections import deque
from itertools import islice, zip_longest

_MISSING = object()


def _consume(it, limit):
    """Drain up to limit items at C speed, storing nothing; return elapsed time."""
    t0 = time.perf_counter()
    deque(islice(it, limit) if limit else it, maxlen=0)
    return time.perf_counter() - t0


def _check_lockstep(iters, limit, names, compare=None):
    """Advance all iterators in lockstep with O(1) memory.
    Fails at the first divergence (element or length) with a witness.
    Returns the common output count."""
    its = [islice(it, limit) if limit else it for it in iters]
    count = 0
    for row in zip_longest(*its, fillvalue=_MISSING):
        if _MISSING in row:
            alive = {n: x for n, x in zip(names, row) if x is not _MISSING}
            raise AssertionError(
                f"length divergence at output #{count}: "
                f"exhausted={[n for n, x in zip(names, row) if x is _MISSING]}, "
                f"still producing={alive}")
        if compare is not None:
            compare(row)                 # raises with the row as witness
        count += 1
    return count


def run_family(family_iter, algos, limit=None, repeats=3, compare=None):
    records = []
    total_time = time.perf_counter()
    total_cycles = 0
    names = list(algos)
    for inst_id, params, args in family_iter:
        G, s, k = args

        # --- correctness pass: untimed, lockstep, O(1) memory ---
        n_cycles = _check_lockstep([fn(*args) for fn in algos.values()], limit, names, compare)
        cut = limit is not None and n_cycles == limit
        print(f"{inst_id=} {n_cycles=}")

        # --- timing passes: bulk consumption, median of repeats ---
        for name, fn in algos.items():
            dt = statistics.median(_consume(fn(*args), limit) for _ in range(repeats))
            records.append({
                "instance": inst_id,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "k": k,
                "algo": name,
                "time": dt,
                "cycles": n_cycles,
                "status": "cutoff" if cut else "ok",
                **params,
            })
        total_cycles += n_cycles * len(algos)
    total_time = time.perf_counter() - total_time
    print(f"{len(records):10} samples enumerated {total_cycles:10} cycles in {total_time:10.2f} seconds.")
    return records


def scatter_compare(title, records, algo_x, algo_y, color_by="cycles", log=True):
    by_inst = defaultdict(dict)
    for r in records:
        by_inst[r["instance"]][r["algo"]] = r
    xs, ys, cs = [], [], []
    for inst, d in by_inst.items():
        if algo_x in d and algo_y in d:         # and d[algo_x]["status"] == d[algo_y]["status"] == "ok":
            assert d[algo_x]["cycles"] == d[algo_y]["cycles"]
            xs.append(d[algo_x]["time"]); ys.append(d[algo_y]["time"])
            val = d[algo_x][color_by]
            cs.append(math.log10(val + 1))
            # cs.append(d[algo_x][color_by])
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.set_title(title)
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=18)
    lo, hi = min(xs+ys), max(xs+ys)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)      # identity line
    if log: ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"{algo_x} time [s]"); ax.set_ylabel(f"{algo_y} time [s]")
    fig.colorbar(sc, label=f"log10(1+{color_by})")
    # fig.colorbar(sc, label=color_by)
    return fig


########################## benchmark runs ###########################

if True:
    records = run_family(ring_family(n_values=range(8, 80, 4), r=4), ALGOS)
    fig = scatter_compare("ring_family", records, algo1_name, algo2_name)
    fig.savefig("ring_family.png")

if True:
    records = run_family(diamond_chain_family(t_values=range(5, 25)), ALGOS)
    fig = scatter_compare("diamond_chain_family", records, algo1_name, algo2_name)
    fig.savefig("diamond_chain_family.png")
    
if True:
    records = run_family(wave_gadget_family(t_values=range(10, 100, 2)), ALGOS)
    fig = scatter_compare("wave_gadget_family", records, algo1_name, algo2_name)
    fig.savefig("wave_gadget_family.png")

if True:
    records = run_family(er_gnm_family(n_values=range(10, 100, 10), seeds=range(1, 1000)), ALGOS, limit=1_000)
    fig = scatter_compare("er_gnm_family", records, algo1_name, algo2_name)
    fig.savefig("er_gnm_family.png")

if True:
    records = run_family(kleinberg_family(n_values=range(3, 16), seeds=range(1, 1000)), ALGOS, limit=1_000)
    fig = scatter_compare("kleinberg_family", records, algo1_name, algo2_name)
    fig.savefig("kleinberg_family.png")

if True:
    records = run_family(watts_strogatz_family(n_values=[1000, 2000], k_values=range(3,16), seeds=range(1, 100)), ALGOS, limit=1_000)
    fig = scatter_compare("watts_strogatz_family", records, algo1_name, algo2_name)
    fig.savefig("watts_strogatz_family.png")

if True:
    records = run_family(dag_backedge_family(n_values=[200, 500, 1000], beta_values=[0, 1, 2, 5, 10, 20, 50, 100, 200], seeds=range(1, 100)), ALGOS, limit=1_000)
    fig = scatter_compare("dag_backedge_family", records, algo1_name, algo2_name)
    fig.savefig("dag_backedge_family.png")

if True:
    np_values = [
        (   100, 0.05),
        (   100, 0.1),
        ( 1_000, 0.005),
        ( 1_000, 0.007),
        ( 1_000, 0.01),
        (10_000, 0.0001),
        (10_000, 0.0002),
    ]
    records = run_family(large_er_family(np_values=np_values, k_values=range(5, 15), seeds=range(10)), ALGOS, limit=1_000)
    fig = scatter_compare("large_er_family", records, algo1_name, algo2_name)
    fig.savefig("large_er_family.png")
