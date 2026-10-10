"""Speed vs area for the N_MAC sweep.

Reads results/n_mac_sweep.csv (written by quartus/sweep_n_mac.tcl) and
  - prints a Markdown table for the README
  - writes docs/n_mac_sweep.png (needs matplotlib; the table works without it)

Throughput and latency are given twice: at the board's 50 MHz clock, and at
each design's own Fmax (what the same RTL would do with a PLL at that rate).

    python3 python/plot_sweep.py [path/to/n_mac_sweep.csv]
"""
import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV = os.path.join(ROOT, "results", "n_mac_sweep.csv")
PNG = os.path.join(ROOT, "docs", "n_mac_sweep.png")
BOARD_MHZ = 50.0

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def num(s):
    return float(s) if s not in ("", None) else None


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["logic_elements"]:
                print("skipping N_MAC=%s (%s)" % (r["n_mac"], r.get("setup_slack_ns_at_50mhz", "")))
                continue
            d = {k: num(v) for k, v in r.items()}
            d["n_mac"] = int(d["n_mac"])
            d["cycles"] = int(d["cycles"])
            rows.append(d)
    rows.sort(key=lambda d: d["n_mac"])
    for d in rows:
        d["us_50"] = d["cycles"] / BOARD_MHZ
        d["inf_s_50"] = BOARD_MHZ * 1e6 / d["cycles"]
        d["us_fmax"] = d["cycles"] / d["fmax_mhz"]
        d["inf_s_fmax"] = d["fmax_mhz"] * 1e6 / d["cycles"]
    return rows


def table(rows):
    base = rows[0]
    out = ["| MACs | Cycles | Latency @ 50 MHz | Speedup | Logic elements | M9Ks | 9-bit mults | Fmax | Inferences/s @ Fmax | Inf/s per LE @ 50 MHz |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in rows:
        out.append("| %d | %d | %.2f us | %.1fx | %d | %d | %d | %.1f MHz | %s | %.1f |" % (
            d["n_mac"], d["cycles"], d["us_50"], base["cycles"] / d["cycles"],
            d["logic_elements"], d["m9k"], d["mult_9bit"], d["fmax_mhz"],
            "{:,.0f}".format(d["inf_s_fmax"]), d["inf_s_50"] / d["logic_elements"]))
    return "\n".join(out)


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    # left: the tradeoff itself, throughput against area, one point per N
    les = [d["logic_elements"] for d in rows]
    thr = [d["inf_s_50"] for d in rows]
    ax1.plot(les, thr, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    for d in rows:
        ax1.annotate("N=%d" % d["n_mac"], (d["logic_elements"], d["inf_s_50"]),
                     textcoords="offset points", xytext=(8, -4), color=INK2, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Logic elements (Quartus fit, whole board design)")
    ax1.set_ylabel("MNIST inferences per second @ 50 MHz")
    ax1.set_title("Throughput vs area", loc="left", fontweight="bold")

    # right: what parallelism costs in clock rate
    ns = [d["n_mac"] for d in rows]
    ax2.plot(ns, [d["fmax_mhz"] for d in rows], color=BLUE, lw=2, marker="o", ms=8,
             mec=SURFACE, mew=2, zorder=3)
    ax2.axhline(BOARD_MHZ, color=ORANGE, lw=1.5, ls="--", zorder=2)
    ax2.annotate("board clock, 50 MHz", (ns[0], BOARD_MHZ), textcoords="offset points",
                 xytext=(0, 6), color=INK2, fontsize=9)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_ylim(0, max(d["fmax_mhz"] for d in rows) * 1.15)
    ax2.set_xlabel("Parallel MACs (N)")
    ax2.set_ylabel("Fmax, slow 85 C corner (MHz)")
    ax2.set_title("Clock rate vs parallelism", loc="left", fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=SURFACE)
    print("wrote %s" % os.path.relpath(PNG, ROOT))


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else CSV)
    if not rows:
        sys.exit("no compiled rows in the CSV")
    print(table(rows))
    print()
    plot(rows)


if __name__ == "__main__":
    main()
