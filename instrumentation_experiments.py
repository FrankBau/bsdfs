# from bsdfs_instrumented import bsdfs
from bcdfs_kickstart_instrumented import bcdfs as bsdfs
from bcdfs_instrumented import bcdfs

import math
import random
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

import networkx as nx
import numpy as np

RUNS = 100_000
K_VALUES = range(3, 11)

# ER params
NMAX = 30

# WS params
WS_N = 1000
WS_D = 6
WS_P = 0.2

PROCESSES = None   # None -> os.cpu_count()
CHUNKSIZE = 100

STATS_KEYS = ("search_calls", "successors_considered",
              "predecessors_considered", "bar_updates")


# ------------------------------------------------------------
# Reachability (unchanged)
# ------------------------------------------------------------

def limited_descendants(G, s, k=math.inf):
    """All nodes within BFS distance <= k from s (s itself excluded)."""
    dist = {s: 0}
    q = deque([s])
    result = []
    succ = G.successors

    while q:
        u = q.popleft()
        if dist[u] == k:
            continue
        for v in succ(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                result.append(v)
                q.append(v)
    return result


def pick_target(G, s_pool):
    t = None
    while t is None:
        s = random.choice(s_pool)
        reachable = limited_descendants(G, s)
        if reachable:
            t = random.choice(reachable)
    return s, t


# ------------------------------------------------------------
# Graph factories (unchanged)
# ------------------------------------------------------------

def make_watts_strogatz(n, d, p, seed):
    random.seed(seed)
    np.random.seed(seed)
    H = nx.watts_strogatz_graph(n, d, p, seed=seed)
    return nx.DiGraph(H)


def make_erdos_renyi(n, p_edge, seed):
    random.seed(seed)
    np.random.seed(seed)
    return nx.gnp_random_graph(n, p_edge, seed=seed, directed=True)


# ------------------------------------------------------------
# Sample generators -- deterministic in `run` alone, module-level
# (picklable by reference; each worker regenerates its own graph,
# nothing large is ever broadcast).
# ------------------------------------------------------------

def er_sample(run, nmax=NMAX):
    seed = 42 + run
    random.seed(seed)
    np.random.seed(seed)
    n = random.randint(6, nmax)
    p_edge = random.uniform(2, 5) / (n - 1)
    G = make_erdos_renyi(n, p_edge, seed)
    s_pool = sorted(G.nodes)
    s, t = pick_target(G, s_pool)
    return G, s, t


def ws_sample(run, n=WS_N, d=WS_D, p=WS_P):
    seed = 42 + run
    G = make_watts_strogatz(n, d, p, seed)
    s_pool = sorted(G.nodes)
    s, t = pick_target(G, s_pool)
    return G, s, t


# ------------------------------------------------------------
# Worker -- op-counts only, no timing. Counts are deterministic given
# (G, s, t, k), so there's no measurement noise to counterbalance and
# no reason to care about call order.
# ------------------------------------------------------------

def worker(args):
    sample_fn, k, run = args
    try:
        G, s, t = sample_fn(run)
        stats_bs, stats_bc = {}, {}
        count_bs = sum(1 for _ in bsdfs(G, s, t, k, stats=stats_bs))
        count_bc = sum(1 for _ in bcdfs(G, s, t, k, stats=stats_bc))
        return dict(count_bs=count_bs, stats_bs=stats_bs,
                    count_bc=count_bc, stats_bc=stats_bc)
    except Exception as e:
        print(f"Error in worker (k={k}, run={run}): {e}")
        return None


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

def aggregate(results):
    results = [r for r in results if r is not None]
    if not results:
        return None

    sum_count_bs = sum(r["count_bs"] for r in results)
    sum_count_bc = sum(r["count_bc"] for r in results)

    stats_bs_totals = {key: sum(r["stats_bs"].get(key, 0) for r in results) for key in STATS_KEYS}
    stats_bc_totals = {key: sum(r["stats_bc"].get(key, 0) for r in results) for key in STATS_KEYS}

    return dict(
        n=len(results),
        count_bs=sum_count_bs, count_bc=sum_count_bc,
        stats_bs=stats_bs_totals, stats_bc=stats_bc_totals,
    )


def print_search_header():
    print(
        f"{'k':>4} {'calls_bs':>14} {'calls_bc':>14} {'calls(bc/bs)':>13} "
        f"{'succ_bs':>14} {'succ_bc':>14} {'succ(bc/bs)':>12} {'count(bc/bs)':>13}"
    )


def print_search_row(k, a):
    if a is None:
        print(f"{k:>4} -- no valid runs --")
        return
    cbs = a["stats_bs"]["search_calls"]
    cbc = a["stats_bc"]["search_calls"]
    sbs = a["stats_bs"]["successors_considered"]
    sbc = a["stats_bc"]["successors_considered"]
    calls_ratio = cbc / cbs if cbs else float("nan")
    succ_ratio = sbc / sbs if sbs else float("nan")
    count_ratio = a["count_bc"] / a["count_bs"] if a["count_bs"] else float("nan")
    print(
        f"{k:4} {cbs:14,} {cbc:14,} {calls_ratio:13.3f} "
        f"{sbs:14,} {sbc:14,} {succ_ratio:12.3f} {count_ratio:13.3f}"
    )


def print_propagation_header():
    print(
        f"{'k':>4} {'pred_bs':>14} {'pred_bc':>14} {'pred(bc/bs)':>12} "
        f"{'bar_bs':>14} {'bar_bc':>14} {'bar(bc/bs)':>11}"
    )


def print_propagation_row(k, a):
    if a is None:
        return
    pbs = a["stats_bs"]["predecessors_considered"]
    pbc = a["stats_bc"]["predecessors_considered"]
    barbs = a["stats_bs"]["bar_updates"]
    barbc = a["stats_bc"]["bar_updates"]
    pred_ratio = pbc / pbs if pbs else float("nan")
    bar_ratio = barbc / barbs if barbs else float("nan")
    print(
        f"{k:4} {pbs:14,} {pbc:14,} {pred_ratio:12.3f} "
        f"{barbs:14,} {barbc:14,} {bar_ratio:11.3f}"
    )


# ------------------------------------------------------------
# Experiment driver
# ------------------------------------------------------------

def run_experiment(pool, sample_fn, runs, k_values, desc):
    aggs = {}
    for k in k_values:
        tasks = [(sample_fn, k, run) for run in range(runs)]
        results = list(tqdm(
            pool.imap_unordered(worker, tasks, chunksize=CHUNKSIZE),
            total=runs, desc=f"k={k}", leave=False,
        ))
        aggs[k] = aggregate(results)

    print(f"\n--- {desc}: search-tree size vs completeness ---")
    print_search_header()
    for k in k_values:
        print_search_row(k, aggs[k])

    print(f"\n--- {desc}: propagation cost ---")
    print_propagation_header()
    for k in k_values:
        print_propagation_row(k, aggs[k])


# ------------------------------------------------------------
# Main -- Pool created once, reused across the whole k-sweep and
# both experiments; no corpus ever built or broadcast.
# ------------------------------------------------------------

if __name__ == "__main__":
    with Pool(processes=PROCESSES) as pool:
        run_experiment(pool, er_sample, RUNS, K_VALUES, f"Erdos-Renyi (nmax={NMAX})")
        run_experiment(pool, ws_sample, RUNS, K_VALUES, f"Watts-Strogatz (n={WS_N}, d={WS_D}, p={WS_P})")
        
#################################### bsdfs vs. bcdfs ###########################
                                                                                                                                                                                                                                       
# --- Erdos-Renyi (nmax=30): search-tree size vs completeness ---
#    k       calls_bs       calls_bc  calls(bc/bs)        succ_bs        succ_bc  succ(bc/bs)  count(bc/bs)
#    3      2,017,185      2,539,219         1.259      7,705,776      4,835,349        0.627         1.000
#    4      3,624,388      5,133,471         1.416     14,565,026     12,145,002        0.834         0.995
#    5      7,610,436     11,401,808         1.498     32,925,007     30,478,022        0.926         0.977
#    6     18,865,040     28,044,094         1.487     86,646,750     80,729,376        0.932         0.953
#    7     50,991,004     73,605,503         1.443    243,052,381    222,375,087        0.915         0.932
#    8    141,081,655    197,809,990         1.402    688,051,845    618,062,725        0.898         0.917
#    9    387,665,562    530,357,867         1.368  1,920,478,922  1,699,926,991        0.885         0.904
#   10  1,044,621,438  1,398,541,809         1.339  5,236,162,344  4,576,200,178        0.874         0.894

# --- Erdos-Renyi (nmax=30): propagation cost ---
#    k        pred_bs        pred_bc  pred(bc/bs)         bar_bs         bar_bc  bar(bc/bs)
#    3      2,659,312              0        0.000        601,618              0       0.000
#    4      9,071,455         14,190        0.002      1,945,269          2,826       0.001
#    5     28,407,037        235,348        0.008      5,881,496         47,072       0.008
#    6     85,478,941      1,439,527        0.017     17,230,408        292,617       0.017
#    7    251,033,425      5,745,352        0.023     49,551,528      1,169,633       0.024
#    8    721,046,397     19,290,171        0.027    139,973,460      3,893,344       0.028
#    9  2,021,794,331     59,607,478        0.029    387,214,301     11,881,881       0.031
#   10  5,521,168,086    175,517,463        0.032  1,045,692,156     34,555,721       0.033
                                                                                                                                                                                                                                       
# --- Watts-Strogatz (n=1000, d=6, p=0.2): search-tree size vs completeness ---
#    k       calls_bs       calls_bc  calls(bc/bs)        succ_bs        succ_bc  succ(bc/bs)  count(bc/bs)
#    3     10,204,216     10,228,054         1.002     63,032,163     17,367,098        0.276         1.000
#    4     33,547,198     33,657,857         1.003    206,575,911     63,418,796        0.307         0.982
#    5     90,728,201     91,187,939         1.005    555,268,110    209,238,276        0.377         0.958
#    6    181,021,443    182,770,010         1.010  1,100,216,272    571,175,054        0.519         0.927
#    7    291,082,626    296,814,927         1.020  1,766,658,584  1,185,388,229        0.671         0.883
#    8    473,844,511    485,583,930         1.025  2,896,626,438  2,168,477,911        0.749         0.814
#    9  1,022,555,813    996,062,281         0.974  6,356,795,984  4,602,266,795        0.724         0.731
#   10  3,346,778,447  2,913,712,561         0.871 21,112,874,497 13,413,297,128        0.635         0.661

# --- Watts-Strogatz (n=1000, d=6, p=0.2): propagation cost ---
#    k        pred_bs        pred_bc  pred(bc/bs)         bar_bs         bar_bc  bar(bc/bs)
#    3        237,776              0        0.000         37,416              0       0.000
#    4      1,387,574          5,288        0.004        218,072            869       0.004
#    5      7,311,720         93,377        0.013      1,148,614         15,159       0.013
#    6     36,153,845        733,927        0.020      5,679,154        119,434       0.021
#    7    173,626,587      5,026,508        0.029     27,282,697        817,496       0.030
#    8    824,207,951     34,456,391        0.042    129,552,331      5,587,349       0.043
#    9  3,896,735,563    218,168,563        0.056    612,656,597     35,316,794       0.058
#   10 18,397,355,849  1,151,913,194        0.063  2,892,981,314    186,529,288       0.064



###################### bcdfs_kickstart (bs) vs. bcdfs (bc) #########################################
# --- Erdos-Renyi (nmax=30): search-tree size vs completeness ---
#    k       calls_bs       calls_bc  calls(bc/bs)        succ_bs        succ_bc  succ(bc/bs)  count(bc/bs)
#    3      2,421,662      2,539,219         1.049      4,707,570      4,835,349        1.027         1.000
#    4      4,871,914      5,133,471         1.054     11,675,072     12,145,002        1.040         0.995
#    5     11,222,957     11,401,808         1.016     30,322,581     30,478,022        1.005         0.977
#    6     29,024,510     28,044,094         0.966     84,354,341     80,729,376        0.957         0.953
#    7     79,206,986     73,605,503         0.929    241,060,625    222,375,087        0.922         0.932
#    8    218,432,036    197,809,990         0.906    686,346,957    618,062,725        0.901         0.917
#    9    595,979,735    530,357,867         0.890  1,919,036,251  1,699,926,991        0.886         0.904
#   10  1,593,377,870  1,398,541,809         0.878  5,234,958,507  4,576,200,178        0.874         0.894

# --- Erdos-Renyi (nmax=30): propagation cost ---
#    k        pred_bs        pred_bc  pred(bc/bs)         bar_bs         bar_bc  bar(bc/bs)
#    3      2,659,312              0        0.000        601,618              0       0.000
#    4      9,072,794         14,190        0.002      1,945,623          2,826       0.001
#    5     28,418,495        235,348        0.008      5,884,419         47,072       0.008
#    6     85,539,357      1,439,527        0.017     17,244,964        292,617       0.017
#    7    251,283,405      5,745,352        0.023     49,608,828      1,169,633       0.024
#    8    721,955,252     19,290,171        0.027    140,173,239      3,893,344       0.028
#    9  2,024,841,636     59,607,478        0.029    387,862,054     11,881,881       0.031
#   10  5,530,878,798    175,517,463        0.032  1,047,702,738     34,555,721       0.033
                                                                                                                                                                                                                                                                                                                                  
# --- Watts-Strogatz (n=1000, d=6, p=0.2): search-tree size vs completeness ---
#    k       calls_bs       calls_bc  calls(bc/bs)        succ_bs        succ_bc  succ(bc/bs)  count(bc/bs)
#    3     10,223,081     10,228,054         1.000     17,355,523     17,367,098        1.001         1.000
#    4     33,638,699     33,657,857         1.001     63,346,537     63,418,796        1.001         0.982
#    5     91,164,292     91,187,939         1.000    209,107,663    209,238,276        1.001         0.958
#    6    183,086,496    182,770,010         0.998    572,482,823    571,175,054        0.998         0.927
#    7    300,839,473    296,814,927         0.987  1,204,197,693  1,185,388,229        0.984         0.883
#    8    519,919,237    485,583,930         0.934  2,334,432,194  2,168,477,911        0.929         0.814
#    9  1,240,094,029    996,062,281         0.803  5,793,336,757  4,602,266,795        0.794         0.731
#   10  4,373,545,370  2,913,712,561         0.666 20,549,503,428 13,413,297,128        0.653         0.661

# --- Watts-Strogatz (n=1000, d=6, p=0.2): propagation cost ---
#    k        pred_bs        pred_bc  pred(bc/bs)         bar_bs         bar_bc  bar(bc/bs)
#    3        237,776              0        0.000         37,416              0       0.000
#    4      1,390,021          5,288        0.004        218,470            869       0.004
#    5      7,335,412         93,377        0.013      1,152,428         15,159       0.013
#    6     36,302,198        733,927        0.020      5,703,077        119,434       0.021
#    7    174,398,568      5,026,508        0.029     27,407,582        817,496       0.030
#    8    828,010,280     34,456,391        0.042    130,168,665      5,587,349       0.043
#    9  3,914,960,758    218,168,563        0.056    615,613,199     35,316,794       0.057
#   10 18,483,962,882  1,151,913,194        0.062  2,907,036,442    186,529,288       0.064
  