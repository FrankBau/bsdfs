"""
Empirically evidence for the exact step counter formulae (asserts)
and convergence to the derived bounds.
Using bsdfs on the clique_trap graph family.
"""

import math
import networkx as nx

from bsdfs_traced import bsdfs_traced

def run_clique_trap():

    for c in range(10, 1000, 10):
        k = round(math.sqrt(c))     # some $k \in o(n)$
        N = k*c - k*(k+1)//2 + 1
        G = nx.DiGraph()
        G.add_node("t")
        for u in range(c):
            for v in range(c):
                if u != v: G.add_edge(u, v)
        total = bsdfs_traced(G, 0, "t", k, lambda _: None)
        assert total == c*N, ("G", c, k, total, c*N)
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"{c:5}: {total/(k*(n+m))=:6.4f}", end="\t")

        if k >= 3:
            Gp = nx.DiGraph()
            Gp.add_edge(0, "t")
            for u in range(c):
                for v in range(c):
                    if u != v: Gp.add_edge(u, v)
            tr = []
            total = bsdfs_traced(Gp, 0, "t", k, tr.append)
            ostep = [e[1] for e in tr if e[0] == "output"]
            assert ostep == [4] and total - 4 == c*N + c*c - 1, ("G+", c, k)
            n, m = Gp.number_of_nodes(), Gp.number_of_edges()
            print(f"{c:5}: {(total-ostep[0])/((k+1)*(n+m))=:6.4f}")


if __name__ == "__main__":
    run_clique_trap()


# 2026-08-04
#
#   G_c  fruitless initial interval  G^+_c terminal interval  [steps per unit]
#
#    10: total/(k*(n+m))=0.8251      10: (total-ostep[0])/((k+1)*(n+m))=0.8554
#    20: total/(k*(n+m))=0.8853      20: (total-ostep[0])/((k+1)*(n+m))=0.9050
#    30: total/(k*(n+m))=0.9057      30: (total-ostep[0])/((k+1)*(n+m))=0.9200
#    40: total/(k*(n+m))=0.9161      40: (total-ostep[0])/((k+1)*(n+m))=0.9273
#    50: total/(k*(n+m))=0.9225      50: (total-ostep[0])/((k+1)*(n+m))=0.9317
#    60: total/(k*(n+m))=0.9268      60: (total-ostep[0])/((k+1)*(n+m))=0.9346
#    70: total/(k*(n+m))=0.9373      70: (total-ostep[0])/((k+1)*(n+m))=0.9440
#    80: total/(k*(n+m))=0.9387      80: (total-ostep[0])/((k+1)*(n+m))=0.9447
#    90: total/(k*(n+m))=0.9456      90: (total-ostep[0])/((k+1)*(n+m))=0.9509
#   100: total/(k*(n+m))=0.9459     100: (total-ostep[0])/((k+1)*(n+m))=0.9507
#   110: total/(k*(n+m))=0.9508     110: (total-ostep[0])/((k+1)*(n+m))=0.9552
#   120: total/(k*(n+m))=0.9507     120: (total-ostep[0])/((k+1)*(n+m))=0.9547
#   130: total/(k*(n+m))=0.9545     130: (total-ostep[0])/((k+1)*(n+m))=0.9582
#   140: total/(k*(n+m))=0.9541     140: (total-ostep[0])/((k+1)*(n+m))=0.9576
#   150: total/(k*(n+m))=0.9572     150: (total-ostep[0])/((k+1)*(n+m))=0.9604
#   160: total/(k*(n+m))=0.9567     160: (total-ostep[0])/((k+1)*(n+m))=0.9597
#   170: total/(k*(n+m))=0.9592     170: (total-ostep[0])/((k+1)*(n+m))=0.9621
#   180: total/(k*(n+m))=0.9615     180: (total-ostep[0])/((k+1)*(n+m))=0.9642
#   190: total/(k*(n+m))=0.9609     190: (total-ostep[0])/((k+1)*(n+m))=0.9635
#   200: total/(k*(n+m))=0.9628     200: (total-ostep[0])/((k+1)*(n+m))=0.9653
#   210: total/(k*(n+m))=0.9646     210: (total-ostep[0])/((k+1)*(n+m))=0.9669
#   220: total/(k*(n+m))=0.9639     220: (total-ostep[0])/((k+1)*(n+m))=0.9662
#   230: total/(k*(n+m))=0.9655     230: (total-ostep[0])/((k+1)*(n+m))=0.9676
#   240: total/(k*(n+m))=0.9669     240: (total-ostep[0])/((k+1)*(n+m))=0.9690
#   250: total/(k*(n+m))=0.9662     250: (total-ostep[0])/((k+1)*(n+m))=0.9682
#   260: total/(k*(n+m))=0.9675     260: (total-ostep[0])/((k+1)*(n+m))=0.9694
#   270: total/(k*(n+m))=0.9687     270: (total-ostep[0])/((k+1)*(n+m))=0.9706
#   280: total/(k*(n+m))=0.9681     280: (total-ostep[0])/((k+1)*(n+m))=0.9698
#   290: total/(k*(n+m))=0.9692     290: (total-ostep[0])/((k+1)*(n+m))=0.9709
#   300: total/(k*(n+m))=0.9702     300: (total-ostep[0])/((k+1)*(n+m))=0.9718
#   310: total/(k*(n+m))=0.9695     310: (total-ostep[0])/((k+1)*(n+m))=0.9711
#   320: total/(k*(n+m))=0.9705     320: (total-ostep[0])/((k+1)*(n+m))=0.9720
#   330: total/(k*(n+m))=0.9714     330: (total-ostep[0])/((k+1)*(n+m))=0.9729
#   340: total/(k*(n+m))=0.9722     340: (total-ostep[0])/((k+1)*(n+m))=0.9737
#   350: total/(k*(n+m))=0.9716     350: (total-ostep[0])/((k+1)*(n+m))=0.9730
#   360: total/(k*(n+m))=0.9724     360: (total-ostep[0])/((k+1)*(n+m))=0.9737
#   370: total/(k*(n+m))=0.9731     370: (total-ostep[0])/((k+1)*(n+m))=0.9744
#   380: total/(k*(n+m))=0.9738     380: (total-ostep[0])/((k+1)*(n+m))=0.9751
#   390: total/(k*(n+m))=0.9732     390: (total-ostep[0])/((k+1)*(n+m))=0.9745
#   400: total/(k*(n+m))=0.9739     400: (total-ostep[0])/((k+1)*(n+m))=0.9751
#   410: total/(k*(n+m))=0.9745     410: (total-ostep[0])/((k+1)*(n+m))=0.9757
#   420: total/(k*(n+m))=0.9751     420: (total-ostep[0])/((k+1)*(n+m))=0.9763
#   430: total/(k*(n+m))=0.9745     430: (total-ostep[0])/((k+1)*(n+m))=0.9757
#   440: total/(k*(n+m))=0.9751     440: (total-ostep[0])/((k+1)*(n+m))=0.9762
#   450: total/(k*(n+m))=0.9757     450: (total-ostep[0])/((k+1)*(n+m))=0.9768
#   460: total/(k*(n+m))=0.9762     460: (total-ostep[0])/((k+1)*(n+m))=0.9773
#   470: total/(k*(n+m))=0.9756     470: (total-ostep[0])/((k+1)*(n+m))=0.9767
#   480: total/(k*(n+m))=0.9761     480: (total-ostep[0])/((k+1)*(n+m))=0.9772
#   490: total/(k*(n+m))=0.9766     490: (total-ostep[0])/((k+1)*(n+m))=0.9776
#   500: total/(k*(n+m))=0.9771     500: (total-ostep[0])/((k+1)*(n+m))=0.9781
#   510: total/(k*(n+m))=0.9766     510: (total-ostep[0])/((k+1)*(n+m))=0.9775
#   520: total/(k*(n+m))=0.9770     520: (total-ostep[0])/((k+1)*(n+m))=0.9780
#   530: total/(k*(n+m))=0.9774     530: (total-ostep[0])/((k+1)*(n+m))=0.9784
#   540: total/(k*(n+m))=0.9779     540: (total-ostep[0])/((k+1)*(n+m))=0.9788
#   550: total/(k*(n+m))=0.9783     550: (total-ostep[0])/((k+1)*(n+m))=0.9792
#   560: total/(k*(n+m))=0.9777     560: (total-ostep[0])/((k+1)*(n+m))=0.9786
#   570: total/(k*(n+m))=0.9781     570: (total-ostep[0])/((k+1)*(n+m))=0.9790
#   580: total/(k*(n+m))=0.9785     580: (total-ostep[0])/((k+1)*(n+m))=0.9794
#   590: total/(k*(n+m))=0.9789     590: (total-ostep[0])/((k+1)*(n+m))=0.9797
#   600: total/(k*(n+m))=0.9792     600: (total-ostep[0])/((k+1)*(n+m))=0.9801
#   610: total/(k*(n+m))=0.9788     610: (total-ostep[0])/((k+1)*(n+m))=0.9796
#   620: total/(k*(n+m))=0.9791     620: (total-ostep[0])/((k+1)*(n+m))=0.9799
#   630: total/(k*(n+m))=0.9794     630: (total-ostep[0])/((k+1)*(n+m))=0.9802
#   640: total/(k*(n+m))=0.9797     640: (total-ostep[0])/((k+1)*(n+m))=0.9805
#   650: total/(k*(n+m))=0.9801     650: (total-ostep[0])/((k+1)*(n+m))=0.9808
#   660: total/(k*(n+m))=0.9796     660: (total-ostep[0])/((k+1)*(n+m))=0.9804
#   670: total/(k*(n+m))=0.9799     670: (total-ostep[0])/((k+1)*(n+m))=0.9806
#   680: total/(k*(n+m))=0.9802     680: (total-ostep[0])/((k+1)*(n+m))=0.9809
#   690: total/(k*(n+m))=0.9805     690: (total-ostep[0])/((k+1)*(n+m))=0.9812
#   700: total/(k*(n+m))=0.9808     700: (total-ostep[0])/((k+1)*(n+m))=0.9815
#   710: total/(k*(n+m))=0.9803     710: (total-ostep[0])/((k+1)*(n+m))=0.9810
#   720: total/(k*(n+m))=0.9806     720: (total-ostep[0])/((k+1)*(n+m))=0.9813
#   730: total/(k*(n+m))=0.9809     730: (total-ostep[0])/((k+1)*(n+m))=0.9816
#   740: total/(k*(n+m))=0.9811     740: (total-ostep[0])/((k+1)*(n+m))=0.9818
#   750: total/(k*(n+m))=0.9814     750: (total-ostep[0])/((k+1)*(n+m))=0.9820
#   760: total/(k*(n+m))=0.9810     760: (total-ostep[0])/((k+1)*(n+m))=0.9816
#   770: total/(k*(n+m))=0.9812     770: (total-ostep[0])/((k+1)*(n+m))=0.9819
#   780: total/(k*(n+m))=0.9815     780: (total-ostep[0])/((k+1)*(n+m))=0.9821
#   790: total/(k*(n+m))=0.9817     790: (total-ostep[0])/((k+1)*(n+m))=0.9823
#   800: total/(k*(n+m))=0.9819     800: (total-ostep[0])/((k+1)*(n+m))=0.9825
#   810: total/(k*(n+m))=0.9821     810: (total-ostep[0])/((k+1)*(n+m))=0.9828
#   820: total/(k*(n+m))=0.9817     820: (total-ostep[0])/((k+1)*(n+m))=0.9824
#   830: total/(k*(n+m))=0.9820     830: (total-ostep[0])/((k+1)*(n+m))=0.9826
#   840: total/(k*(n+m))=0.9822     840: (total-ostep[0])/((k+1)*(n+m))=0.9828
#   850: total/(k*(n+m))=0.9824     850: (total-ostep[0])/((k+1)*(n+m))=0.9830
#   860: total/(k*(n+m))=0.9826     860: (total-ostep[0])/((k+1)*(n+m))=0.9832
#   870: total/(k*(n+m))=0.9828     870: (total-ostep[0])/((k+1)*(n+m))=0.9834
#   880: total/(k*(n+m))=0.9824     880: (total-ostep[0])/((k+1)*(n+m))=0.9830
#   890: total/(k*(n+m))=0.9826     890: (total-ostep[0])/((k+1)*(n+m))=0.9832
#   900: total/(k*(n+m))=0.9828     900: (total-ostep[0])/((k+1)*(n+m))=0.9834
#   910: total/(k*(n+m))=0.9830     910: (total-ostep[0])/((k+1)*(n+m))=0.9835
#   920: total/(k*(n+m))=0.9832     920: (total-ostep[0])/((k+1)*(n+m))=0.9837
#   930: total/(k*(n+m))=0.9834     930: (total-ostep[0])/((k+1)*(n+m))=0.9839
#   940: total/(k*(n+m))=0.9830     940: (total-ostep[0])/((k+1)*(n+m))=0.9835
#   950: total/(k*(n+m))=0.9832     950: (total-ostep[0])/((k+1)*(n+m))=0.9837
#   960: total/(k*(n+m))=0.9834     960: (total-ostep[0])/((k+1)*(n+m))=0.9839
#   970: total/(k*(n+m))=0.9835     970: (total-ostep[0])/((k+1)*(n+m))=0.9841
#   980: total/(k*(n+m))=0.9837     980: (total-ostep[0])/((k+1)*(n+m))=0.9842
#   990: total/(k*(n+m))=0.9839     990: (total-ostep[0])/((k+1)*(n+m))=0.9844
