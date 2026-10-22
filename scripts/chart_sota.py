"""Field chart: every model scored on both public Singlish benchmarks, one point each.

    python scripts/chart_sota.py          # reads results/table.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FAFAF8"
OTHER, OURS, OURS_OLD = "#9AA3B2", "#E2803B", "#F0BE96"

# (result name, label, label position in data coordinates); outside models first
OTHERS = [
    ("whisper-large-v3-turbo", "Whisper large-v3-turbo\n(OpenAI, the base model)", (76.6, 70.6)),
    ("mjwong-turbo-singlish", "Public Singlish fine-tune (mjwong)", (77.0, 75.6)),
    ("MERaLiON-2-3B", "MERaLiON-2-3B (A*STAR)", (84.6, 71.3)),
    ("gpt-4o-transcribe-diarize", "GPT-4o Transcribe\n(OpenAI API)†", (69.8, 70.8)),
]
# (result name, label, label position, is the release)
OURS_PTS = [
    ("whisperlah-v1", "WhisperLah v1", (74.0, 78.8), False),
    ("whisperlah-v2.1", "v2.1 room", (78.6, 79.4), False),
    ("whisperlah-v2", "v2 room", (81.2, 78.6), False),
    ("whisperlah-v2.1-near", "v2.1 near-mic", (83.2, 77.9), False),
    ("whisperlah-v2-near", "v2 near-mic", (86.0, 78.9), False),
    ("whisperlah-v3", "WhisperLah v3\nroom mode", (78.6, 84.0), True),
    ("whisperlah-v3-near", "WhisperLah v3\nnear-mic mode", (85.6, 84.6), True),
]
PATH = ["whisperlah-v1", "whisperlah-v2-near", "whisperlah-v3-near"]  # best mode of each release
LEADER = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=5)


def main(out=ROOT / "results" / "whisperlah_sota_chart.png"):
    t = {(r["set"], r["model"]): r["wer"] for r in json.load(open(ROOT / "results" / "table.json"))
         if r["view"] == "full"}
    # plotted as word accuracy (100 - WER) so that up and right are better, as on a leaderboard
    xy = {m: (100 - t[("sasrbench", m)], 100 - t[("mnsc_part3", m)]) for m, *_ in OTHERS + OURS_PTS
          if ("sasrbench", m) in t and ("mnsc_part3", m) in t}
    plt.rcParams["font.family"] = ["Inter", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10, 7.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xlo, xhi, ylo, yhi = 67, 90, 67, 86
    # dashed lines of equal average accuracy across the two benchmarks
    g = np.linspace(xlo, xhi, 50)
    for avg in (70, 75, 80, 85):
        ax.plot(g, 2 * avg - g, color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        lx = min(xhi - 0.2, 2 * avg - ylo - 0.4)
        ax.text(lx, 2 * avg - lx + 0.25, f"avg {avg}", fontsize=8, color=MUTED, ha="right", va="bottom")
    # the WhisperLah release path
    pts = [xy[m] for m in PATH if m in xy]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.4,
                                                             shrinkA=7, shrinkB=7, alpha=0.7), zorder=2)
    for m, label, pos in OTHERS:
        if m in xy:
            ax.scatter(*xy[m], s=70, color=OTHER, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(label, xy[m], xytext=pos, fontsize=9.5, color=INK2, va="center",
                        ha="left" if pos[0] >= xy[m][0] else "right", arrowprops=LEADER)
    for m, label, pos, release in OURS_PTS:
        if m in xy:
            near = m.endswith("-near")
            ax.scatter(*xy[m], s=130 if release else 60, marker="D" if near else "o",
                       color=OURS if release else OURS_OLD, zorder=4, edgecolor="white", linewidth=0.8)
            w = 100 - xy[m][0], 100 - xy[m][1]
            ax.annotate(f"{label}  ({w[0]:.1f} / {w[1]:.1f})" if release else label, xy[m], xytext=pos,
                        fontsize=10 if release else 8.5, color=INK if release else INK2,
                        fontweight="bold" if release else "normal", va="center",
                        ha="left" if pos[0] >= xy[m][0] else "right", arrowprops=LEADER)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("SASRBench-v1 word accuracy, % (100 − WER): close-talk Singlish", fontsize=10.5, color=INK2)
    ax.set_ylabel("NSC Part 3 word accuracy, % (100 − WER): two-person conversation", fontsize=10.5, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.07, 0.955, "WhisperLah against the field on Singlish", fontsize=16, color=INK, fontweight="bold")
    fig.text(0.07, 0.915, "Every model we ran on both public Singlish benchmarks, same clips and text normaliser. "
             "Top right is best.\nDashed: equal average accuracy. Bracketed on v3: its WER, SASRBench / NSC Part 3.\n"
             "Orange: WhisperLah (diamonds: near-mic mode); the arrows trace each release's best mode.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.07, 0.015, "† GPT-4o's NSC Part 3 score is on a 299-clip subset to limit API cost. "
             "Contaminated clips excluded for every model.", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.09)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
