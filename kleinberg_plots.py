#!/usr/bin/env python3
"""Parse bsdfs/bcdfs comparison logs and plot miss ratio + per-interval timing.

Usage:
    python plot_ratios.py results.txt
    python kleinberg_test.py | python plot_ratios.py

Expected line format (one line per (s,t) run, one block per k):
    iv=       8 iv2=       8 us/iv=    271.44 us2/iv2=    214.00 | ...
"""

import re
import sys
import matplotlib.pyplot as plt

KS = [6, 8, 10, 12, 14]  # k values, in block order per line
OUT = "kleinberg_ratios.png"

pat = re.compile(r"iv=\s*(\d+)\s+iv2=\s*(\d+)\s+us/iv=\s*([\d.]+)\s+us2/iv2=\s*([\d.]+)")

src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
rows = []  # (k, iv, iv2, us_per_iv, us2_per_iv2)
for line in src:
    blocks = pat.findall(line)
    if not blocks:
        continue
    assert len(blocks) == len(KS), f"expected {len(KS)} blocks: {line!r}"
    for k, (iv, iv2, a, b) in zip(KS, blocks):
        rows.append((k, int(iv), int(iv2), float(a), float(b)))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))


# --- Panel 1: fraction of paths missed by BC-DFS ---
ax = axes[0]
for k in KS:
    pts = [(iv - 1, 1 - (iv2 - 1) / (iv - 1))
           for kk, iv, iv2, _, _ in rows if kk == k and iv > 1]
    if pts:
        x, y = zip(*pts)
        ax.scatter(x, y, s=12, alpha=0.7, marker='.', label=f"k={k}")
ax.set_xscale("log")
ax.set_xlabel("paths enumerated by BS-DFS")
ax.set_ylabel("fraction missed by BC-DFS")
ax.set_title("BC-DFS incompleteness")
ax.grid(True, alpha=0.3)
ax.legend()


# --- Panel 3: total time normalized by the SAME (BS-DFS) interval count ---
# us2/iv vs us/iv, i.e. (us2/iv2)*(iv2/iv) / (us/iv). Removes the flattery
# BC-DFS gets from dividing by its smaller interval count.
ax = axes[1]
for k in KS:
    pts = [(iv, (b * iv2 / iv) / a) for kk, iv, iv2, a, b in rows if kk == k]
    x, y = zip(*pts)
    ax.scatter(x, y, s=12, alpha=0.7, marker='.', label=f"k={k}")
ax.axhline(1.0, color="gray", lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlabel("intervals (BS-DFS)")
ax.set_ylabel("t_BSDFS / t_BCDFS = total-time ratio")
ax.set_title("Runtime Ratio, Common Denominator i_BSDFS")
ax.grid(True, alpha=0.3)
ax.legend()


# --- Panel 2: per-interval time, each normalized by its own interval count ---
ax = axes[2]
for k in KS:
    pts = [(iv, b / a) for kk, iv, iv2, a, b in rows if kk == k]
    x, y = zip(*pts)
    ax.scatter(x, y, s=12, alpha=0.7, marker='.', label=f"k={k}")
ax.axhline(1.0, color="gray", lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlabel("intervals (BS-DFS)")
ax.set_ylabel("(t_BSDFS/i_BSDFS) / (t_BCDFS/i_BCDFS)")
ax.set_title("Runtime Ratio, per-interval found, own denominators")
ax.grid(True, alpha=0.3)
ax.legend()


plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"wrote {OUT}", file=sys.stderr)

# --- summary table ---
def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

print(f"{'k':>3} {'#pts':>5} {'#paths>0':>8} {'miss med':>9} {'miss max':>9} "
      f"{'t-own med':>9} {'t-tot med':>9}")
for k in KS:
    sub = [r for r in rows if r[0] == k]
    withp = [r for r in sub if r[1] > 1]
    miss = [1 - (iv2 - 1) / (iv - 1) for _, iv, iv2, _, _ in withp]
    town = [b / a for _, _, _, a, b in sub]
    ttot = [(b * iv2 / iv) / a for _, iv, iv2, a, b in sub]
    print(f"{k:>3} {len(sub):>5} {len(withp):>8} "
          f"{med(miss) if miss else float('nan'):>9.3f} "
          f"{max(miss) if miss else float('nan'):>9.3f} "
          f"{med(town):>9.3f} {med(ttot):>9.3f}")
