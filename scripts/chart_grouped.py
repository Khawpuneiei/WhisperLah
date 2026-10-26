"""Grouped bars: WhisperLah v3 and its closest rivals on all 7 Singlish test sets, maker logos under each model.

    python scripts/chart_grouped.py [out.png]      # reads results/table.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logos import logo_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FAFAF8"
OURS_BAND = "#FBE7D7"

# (result name, label): WhisperLah first, then the rivals by their 7-set average
MODELS = [
    ("whisperlah-v3", "WhisperLah v3"),
    ("MERaLiON-2-3B", "MERaLiON-2-3B"),
    ("SenseVoice-Small", "SenseVoice Small"),
    ("gpt-transcribe", "GPT Transcribe"),
    ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3"),
    ("whisper-large-v3-turbo", "Whisper large-v3-turbo\n(the base model)"),
]
# (set, legend label, colour, read speech); colours validated as an adjacent categorical set
SETS = [
    ("sasrbench", "SASRBench-v1", "#2a78d6", False),
    ("mnsc_part3", "NSC Part 3", "#1baf7a", False),
    ("mnsc_part4", "NSC Part 4", "#eda100", False),
    ("mnsc_part5", "NSC Part 5", "#e87ba4", False),
    ("mnsc_part6", "NSC Part 6", "#008300", False),
    ("mnsc_part1", "NSC Part 1 (read)", "#6250d6", True),
    ("mnsc_part2", "NSC Part 2 (read)", "#e34948", True),
]


def wer_of(t, s, m):
    # WhisperLah in the mode each set calls for: near-mic on close-talk SASRBench, room mode on the rest
    if m == "whisperlah-v3" and s == "sasrbench":
        m = "whisperlah-v3-near"
    return t[(s, m)]["wer"]


def main(out=ROOT / "results" / "whisperlah_grouped_chart.png"):
    rows = [r for r in json.load(open(ROOT / "results" / "table.json")) if r["view"] == "full"]
    t = {(r["set"], r["model"]): r for r in rows}
    # the star goes to the lowest WER among WhisperLah v3 and every outside model run on the full set
    others = {m for _, m in t if not m.startswith(("whisperlah", "probe", "serve", "ct2")) and "-autolang" not in m
              and m not in ("gpt-4o-transcribe-nochunk", "gpt-4o-mini-transcribe")}
    best = {}
    for s, *_ in SETS:
        cands = [(wer_of(t, s, "whisperlah-v3"), "whisperlah-v3")]
        cands += [(t[(s, m)]["wer"], m) for m in others if (s, m) in t and t[(s, m)]["clips"] == t[(s, "whisperlah-v3")]["clips"]]
        best[s] = min(cands)[1]

    plt.rcParams["font.family"] = ["Inter", "Arial", "DejaVu Sans"]
    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n, width = len(SETS), 0.11
    for g, (m, label) in enumerate(MODELS):
        if m.startswith("whisperlah"):
            ax.axvspan(g - 0.47, g + 0.47, color=OURS_BAND, zorder=0, linewidth=0)
        for i, (s, _, colour, read) in enumerate(SETS):
            x = g + (i - (n - 1) / 2) * width
            v = wer_of(t, s, m)
            ax.bar(x, v, width * 0.86, color=colour, hatch="////" if read else None, zorder=3, linewidth=0)
            won = best[s] == m
            ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, zorder=4,
                    color=INK if won else INK2, fontweight="bold" if won else "normal")
            if won:
                ax.text(x, v + 2.6, "★", ha="center", va="bottom", fontsize=10, color=INK, zorder=4)
        ours = m.startswith("whisperlah")
        logo_label(ax, m, label, (g, -0.03), height=15, fontsize=10.5, color=INK if ours else INK2,
                   weight="bold" if ours else "normal")
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 52)
    ax.set_xticks([])
    ax.set_ylabel("Word error rate, %  (lower is better)", fontsize=10.5, color=INK2)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch="////" if read else None, linewidth=0) for _, _, c, read in SETS]
    ax.legend(handles, [lab for _, lab, _, _ in SETS], ncol=len(SETS), loc="lower left", bbox_to_anchor=(0, 1.0),
              frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.4, labelcolor=INK2)

    wins = sum(best[s] == "whisperlah-v3" for s, *_ in SETS)
    losses = [(lab, best[s]) for s, lab, _, _ in SETS if best[s] != "whisperlah-v3"]
    fig.text(0.055, 0.955, f"WhisperLah v3 has the lowest word error rate on {wins} of {len(SETS)} Singlish test sets",
             fontsize=16, color=INK, fontweight="bold")
    lost = " and ".join(lab.replace(" (read)", "") for lab, _ in losses)
    rival = {m: lab for m, lab in MODELS}[losses[0][1]] if losses else ""
    fig.text(0.055, 0.915, f"★ lowest of WhisperLah v3 and all {len(others)} outside models we ran, scored on the same clips. "
             + (f"It is behind on {lost}, read-speech sets (hatched), where {rival} is best." if losses else ""),
             fontsize=10, color=INK2)
    fig.text(0.055, 0.02, "WhisperLah v3 in near-mic mode on SASRBench (close-talk) and room mode elsewhere. "
             "The read-speech prompts recur in NSC training data, which a model trained on the full corpus has seen "
             f"(see Benchmark protocol).\nThe other {len(others) - len(MODELS) + 1} models are in the README table. Logos are trademarks of their "
             "owners and only show who made each model.", fontsize=8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.17)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
