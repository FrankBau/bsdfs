# Code for the BS-DFS Algorithm

Companion to

    Frank Bauernöppel and Jörg-Rüdiger Sack
    "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with $O(k(n+m))$ Delay Using Edge-Consistent Node Barriers"

to appear.


# Simple s-t Path Enumeration

`bsdsf(G, s, t, k)` enumerates all length $k$ bounded simple paths 
from node $s$ to node $t$ in the directed graph $G$. 
The delay is $O(k(n+m))$ per output with small constants.

It replaces `BC-DFS` which has issues, see the upcoming paper.

The original `BC-DFS` algorithm was translated from

    author       = {Peng, You and Lin, Xuemin and Zhang, Ying and Zhang, Wenjie and Qin, Lu and Zhou, Jingren},
    title        = {Efficient Hop-constrained s-t Simple Path Enumeration},
    volume       = {30},
    year         = {2021},
    issn         = {0949-877X},
    url          = {https://doi.org/10.1007/s00778-021-00674-5},
    doi          = {10.1007/s00778-021-00674-5},
    pages        = {799--823},
    number       = {5},
    journaltitle = {The {VLDB} Journal},
    date         = {2021-09-01}

and 

    author     = {Peng, You and Zhang, Ying and Lin, Xuemin and Zhang, Wenjie and Qin, Lu and Zhou, Jingren},
    title      = {Towards bridging theory and practice: hop-constrained s-t simple path enumeration},
    year       = {2019},
    issue_date = {December 2019},
    publisher  = {VLDB Endowment},
    volume     = {13},
    number     = {4},
    issn       = {2150-8097},
    url        = {https://doi.org/10.14778/3372716.3372720},
    doi        = {10.14778/3372716.3372720},
    journal    = {Proc. VLDB Endow.},
    month      = dec,
    pages      = {463-476},
    numpages   = {14}

where `BC-DFS` was introduced and appears in pseudocode.

Caution: `BC-DFS` is not complete: it can miss some output; 
and its delay is unknown, as explained in our paper.


# Simple s-Cycle Enumeration

`bsdsf(G, s, s, k)` enumerates all length $k$ bounded simple cycles 
containing node $s$ in the directed graph $G$.
The delay is $O(k(n+m))$ per output with small constants.

It replaces `CYCLE_SEARCH` which has issues, see https://arxiv.org/abs/2512.08392:

    title         = {Finding All Bounded-Length Simple Cycles in a Directed Graph -- Revisited},
    author        = {Frank Bauernöppel and Jörg-Rüdiger Sack},
    year          = {2026},
    eprint        = {2512.08392},
    doi           = {10.48550/arXiv.2512.08392},


The original `CYCLE_SEARCH` algorithm was translated from https://arxiv.org/abs/2105.10094

    title      = {Finding All Bounded-Length Simple Cycles in a Directed Graph},
    author     = {Anshul Gupta and Toyotaro Suzumura},
    year       = {2021},
    doi        = {10.48550/arXiv.2105.10094},

where `CYCLE_SEARCH` was introduced and appears in pseudocode.

Caution: `CYCLE_SEARCH` is not complete: it can miss some output;
and its delay is unknown, as explained in our preprint.
