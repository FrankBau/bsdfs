# Johnson's cycle finding algorithm
# see https://youtu.be/johyrWospv0

# the orginal has no length bound
# the added length bound implies no linear delay bound
# consider a dead clique as a trap


from math import inf


def johnson_cycles(G, s):
    stack = []
    blocked = {}
    B = {}
    for node in G.nodes:
        blocked[node] = False
        B[node] = []

    def unblock(u):
        blocked[u] = False
        while B[u]:
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
                yield stack.copy()
                found = True
            elif not blocked[w]:
                if (yield from circuit(w)):
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

    yield from circuit(s)
    
    
def johnson_paths_k(G, s, t, k=inf):
    b = {v: inf for v in G.nodes}
    B = {v: set() for v in G.nodes}
    stack = []

    def unblock(v):
        b[v] = inf
        while B[v]:
            w = B[v].pop()
            if b[w] < inf:
                unblock(w)

    def search(v):
        stack.append(v)
        d = inf

        if v == t:
            yield stack.copy()
            d = len(stack) - 1
        elif len(stack) <= k:
            b[v] = len(stack) - 1
            for w in G.successors(v):
                if w not in stack and b[w] > len(stack):
                    d_w = yield from search(w)
                    d = min(d, d_w)

        if d < inf:
            unblock(v)
        else:
            for w in G.successors(v):
                B[w].add(v)

        stack.pop()
        return d

    yield from search(s)

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
        if found:
            unblock(v)
        else:
            for w in G.successors(v):
                if v not in B[w]:
                    B[w].add(v)

        stack.pop()
        return found

    yield from search(s)
    
#############################################################


import networkx as nx
import random
import dfs
from multiprocessing import Pool
from tqdm import tqdm
from itertools import islice


def worker_er(args):
    n, run = args
    random.seed(42 + run)
    p = random.uniform(0, 1)
    k = random.getrandbits(n).bit_count()  # binomial distribution

    G = nx.gnp_random_graph(n, p, directed=True)
    s, t = random.sample(range(n), 2)

    # we cutoff when a number of paths was generated
    paths1 = list(islice(johnson_paths_k(G, s, t, k), 500))
    assert paths1 == sorted(paths1)  # lexicographic order
    paths2 = list(islice(dfs.all_simple_paths(G, s, t, k), 500))
    assert paths2 == sorted(paths2)  # lexicographic order
    if paths1 != paths2:
        ps1 = set(map(tuple, paths1))
        ps2 = set(map(tuple, paths2))
        missing2 = ps1 - ps2
        missing1 = ps2 - ps1
        raise AssertionError(
        # print(
            f"{s=} {t=} {k=} {G.edges=} {list(missing1)[:1]=}  {list(missing2)[:1]=}"
        )
    return len(paths1)


def task_er(n, runs):
    for run in range(runs):
        yield (n, run)


def validate_er(n, runs, processes=None):
    print(n)
    sum_paths = 0
    if processes == 0:
        for run in tqdm(range(runs)):
            sum_paths += worker_er((n, run))
    else:
        with Pool(processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_er, task_er(n, runs), chunksize=200),
                total=runs,
            ):
                sum_paths += result
    return sum_paths


def main():
    seed = 42
    random.seed(seed)

    # quick test for major error and general failure
    for n in range(2, 5):
        validate_er(n, 10_000, processes=0)

    for n in range(5, 15):
        validate_er(n, 1_000_000)


if __name__ == "__main__":
    main()
