Code for the BS-DFS algorithm

    Frank Bauernöppel, Jörg-Rüdiger Sack
    "Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with O(k(n + m)) Delay"

to appear.

The original BC-DFS algorithm was translated from

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

where BC-DFS was introduced and appears in pseudo-code.

Caution: BC-DFS is not complete (can miss some output) and its delay is unknown, as explained in our paper.