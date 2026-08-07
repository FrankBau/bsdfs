# Johnson's algorithm
# see https://youtu.be/johyrWospv0

import networkx as nx

def is_simple(path):
    return len(list(path))==len(set(path))


def johnson_cycles(G, s, k=float("inf")):
    cycles = []
    stack = []
    blocked = {v: False for v in G.nodes}
    B = {v: [] for v in G.nodes}

    def unblock(u):
        blocked[u] = False
        while B[u]: # cannot mutate list in for
            w = B[u].pop()
            if blocked[w]:
                unblock(w)

    def circuit(v):
        found = False
        stack.append(v)
        blocked[v] = True
    # phase 1
        for w in G.successors(v):
            if w == s:
                cycles.append(stack + [w])
                found = True
            elif not blocked[w]:
                if circuit(w):
                    found = True
    # phase 2
        if found:
            unblock(v)
        else:
            for w in G.successors(v):
                if v not in B[w]:
                    B[w].append(v)
        stack.pop()
        return found

    circuit(s)
    return cycles

######################################################################

def check_lemma1(v, found, G, s, t, stack, blocked, B):
    stack_set = set(stack)

    def reachable(src, dst, forbidden):
        if src in forbidden:   # ← src is on the path, so path can't avoid forbidden
            return False
        if src == dst:
            return True
        visited = {src}
        frontier = [src]
        while frontier:
            node = frontier.pop()
            for w in G.successors(node):
                if w == dst:
                    return True
                if w not in forbidden and w not in visited:
                    visited.add(w)
                    frontier.append(w)
        return False

    def would_unblock(x):
        """True iff UNBLOCK(v) propagates to x, given that it is called (found=True)."""
        visited = set()
        frontier = [v]
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == x:
                return True
            for w in B[node]:
                if blocked[w] and w not in visited:
                    frontier.append(w)
        return False

    avoid_full    = stack_set          # condition (ii): avoid all stack vertices
    avoid_minus_v = stack_set - {v}    # condition (i):  avoid stack \ {v}

    violations = []
    for x in G.nodes:
        if not blocked[x]:
            continue

        if x == v:
            # UNBLOCK(v) is called iff found=True, and always unblocks v itself
            will_unblock = found
            # (ii) trivially true: v is on the stack, so forbidden in avoid_full
            cond_i  = reachable(v, t, avoid_minus_v)
            cond_ii = True  # v ∈ stack, so no path from v to t avoids full stack
        else:
            # UNBLOCK propagates to x only if found=True and B-graph reaches x
            will_unblock = found and would_unblock(x)
            cond_i  = (reachable(x, v, avoid_minus_v) and
                       reachable(v, t, avoid_minus_v))
            cond_ii = not reachable(x, t, avoid_full)

        expected = cond_i and cond_ii

        if will_unblock != expected:
            violations.append(
                f"  x={x}: will_unblock={will_unblock}, "
                f"cond_i={cond_i}, cond_ii={cond_ii} → expected {expected}"
            )

    if violations:
        raise AssertionError(
            f"Lemma 1 violated at L2 | v={v}, found={found}, stack={list(stack)}\n"
            + "\n".join(violations)
        )

##############################################################

def johnson_paths(G, s, t):
    stack = []
    blocked = {v: False for v in G.nodes}
    B = {v: set() for v in G.nodes}

    def unblock(u):
        blocked[u] = False
        while B[u]:
            w = B[u].pop()
            if blocked[w]:
                unblock(w)

    def search(v):
        found = False
        stack.append(v)

        if v == t:
            yield stack.copy()
            found = True
            
        else:
            blocked[v] = True
            for w in G.successors(v):
                if not blocked[w]:
                    if (yield from search(w)):
                        found = True
        # L2
        check_lemma1(v, found, G, s, t, stack, blocked, B)
        if found:
            unblock(v)
        else:
            for w in G.successors(v):
                if v not in B[w]:
                    B[w].add(v)

        stack.pop()
        return found

    yield from search(s)


import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
from itertools import permutations


def gosper(n, k):
    """Gosper's Hack: an efficient method of computing the next higher integer with the same number of set bits"""
    if k == 0:
        yield 0
        return
    x = (1 << k) - 1
    while x < (1 << n):
        yield x
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r


def by_popcount_from_middle(m):
    mid = m // 2
    for delta in range(mid + 1):
        for k in ([mid - delta, mid + delta] if delta > 0 else [mid]):
            if 0 <= k <= m:
                yield from gosper(m, k)


def by_popcount(m):
    for i in range(m + 1):
        yield from gosper(m, i)


def build_graph_from_mask(n, mask):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for bit, (u, v) in enumerate(permutations(range(n), 2)):
        if (mask >> bit) & 1:
            G.add_edge(u, v)
    return G


def worker(args):
    n, mask, s, t, k = args
    total = 0
    G = build_graph_from_mask(n, mask)
    paths1 = list(johnson_paths(G, s, t))
    paths2 = list(nx.all_simple_paths(G, s, t, k))
    assert sorted(paths1) == sorted(paths2)
    total += len(paths1)
    return total


def validate_layers(n, s, t, k):
    m = n * (n - 1)
    total = 0

    for edges in range(m + 1):
        masks = list(gosper(m, edges))
        print(f"popcount={edges}, {len(masks)} graphs")

        # small layers: run serially (cheap, avoids overhead)
        if len(masks) < 2000:
            for mask in tqdm(masks, desc=f"pc={edges}"):
                total += worker((n, mask, s, t, k))
            continue

        # large layers: parallelize *within* the layer
        with Pool() as pool:
            for r in tqdm(
                pool.imap_unordered(
                    worker, ((n, mask, s, t, k) for mask in masks), chunksize=200
                ),
                total=len(masks),
                desc=f"pc={edges}",
            ):
                total += r

    print(f"total paths found {total}")



import random
from multiprocessing import Pool
from tqdm import tqdm


def worker_er(args):
    n, run = args
    random.seed(42 + run)
    p = random.uniform(0.2, 0.9)
    k = n-1 # Johnson has no k 

    G = nx.gnp_random_graph(n, p, directed=True)
    s, t = random.sample(range(n), 2)
    
    paths1 = list(johnson_paths(G, s, t))
    paths2 = list(nx.all_simple_paths(G, s, t))
    assert sorted(paths1) == sorted(paths2), f"{s=} {t=} {G.edges=} "
    return len(paths1)


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs):
    total = 0
    with Pool() as pool:
        for result in tqdm(
            pool.imap_unordered(
                worker_er, task_er(n, runs), chunksize=200
            )
        ):
            total += result
    return result



if __name__ == "__main__":

    validate_er(8, 1_000_000)
    quit()

    import networkx as nx

    # bcdfs counterexample
    G = nx.parse_adjlist(
        [
            'A B C',
            'B C D E',
            'C B D',
            'D B',
            'E'
        ], create_using=nx.DiGraph)

    s = min(G.nodes())
    t = max(G.nodes())

    paths = johnson_paths(G, s, t)
    paths = list(paths)
    assert paths == [['A', 'B', 'E'], ['A', 'C', 'B', 'E'], ['A', 'C', 'D', 'B', 'E']], "missing: None"

    n = 6
    s = 3
    t = 5
    k = 5
    validate_layers(n, s, t, k)
