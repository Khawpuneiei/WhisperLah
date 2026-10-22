"""Growth chart: WER of each WhisperLah version on the headline benchmarks.

    python scripts/chart_growth.py        # reads results/table.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FFFFFF"

# x axis: base model, then each released WhisperLah version
VERSIONS = [("whisper-large-v3-turbo", "Base\nWhisper"), ("whisperlah-v1", "v1"),
            ("whisperlah-v2", "v2"), ("whisperlah-v3", "v3")]
# (set, label, colour); each point is the better of room and near-mic mode
SERIES = [
    ("ami_sdm", "AMI meetings, table mic (English)", "#3B7DD8", 0),
    ("mnsc_part3", "NSC Part 3, two-person conversation", "#2A9D6F", 10),
    ("sasrbench", "SASRBench-v1", "#E2803B", -14),
]


def main(out=ROOT / "results" / "whisperlah_growth_chart.png"):
    t = {(r["set"], r["model"]): r["wer"] for r in json.load(open(ROOT / "results" / "table.json"))
         if r["view"] == "full"}
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for s, label, color, nudge in SERIES:
        xs, ys = [], []
        for i, (m, _) in enumerate(VERSIONS):
            vals = [t[(s, k)] for k in (m, m + "-near") if (s, k) in t]
            if vals:
                xs.append(i)
                ys.append(min(vals))
        # solid between neighbouring versions, dashed across a version that was not measured
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.5, linestyle="-" if x1 - x0 == 1 else (0, (3, 3)))
        ax.plot(xs, ys, color=color, linestyle="none", marker="o", markersize=7, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=10, color=INK, fontweight="bold" if x == xs[-1] else "normal")
        drop = ys[0] - ys[-1]
        ax.annotate(f"{label}\n−{drop:.1f} pt vs base", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(16, nudge), va="center", fontsize=10, color=color)
    ax.set_xticks(range(len(VERSIONS)), [v for _, v in VERSIONS], fontsize=11, color=INK2)
    ax.set_xlim(-0.3, len(VERSIONS) - 0.3)
    ax.set_ylim(0, 38)
    ax.set_ylabel("word error rate, %  (lower is better)", fontsize=10, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("WhisperLah, version by version", loc="left", fontsize=14, color=INK, pad=14)
    fig.text(0.01, 0.01, "Better of room and near-mic mode per point. v1 was not run on AMI. "
             "Same clips and normaliser throughout; contaminated clips excluded.", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.72, top=0.88, bottom=0.14)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
