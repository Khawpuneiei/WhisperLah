"""Leaderboard bars: every model run on all 7 Singlish test sets, ranked by average word accuracy, maker logos below.

    python scripts/chart_ranked.py [out.png]      # reads results/table.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logos import logo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FAFAF8"
OTHER, OURS, OURS_OLD = "#B4BCC8", "#E2803B", "#F0BE96"
SETS = ["sasrbench", "mnsc_part3", "mnsc_part4", "mnsc_part5", "mnsc_part6", "mnsc_part1", "mnsc_part2"]
LABEL = {
    "whisperlah-v3": "WhisperLah v3", "whisperlah-v2": "WhisperLah v2",
    "MERaLiON-2-3B": "MERaLiON-2-3B", "SenseVoice-Small": "SenseVoice Small", "gpt-transcribe": "GPT Transcribe",
    "mjwong-turbo-singlish": "Singlish fine-tune, mjwong", "parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
    "whisper-large-v3": "Whisper large-v3", "distil-large-v3.5": "Distil-Whisper large-v3.5",
    "whisper-large-v3-turbo": "Whisper large-v3-turbo", "gpt-4o-mini-transcribe-nochunk": "GPT-4o mini Transcribe",
    "Voxtral-Mini-3B": "Voxtral Mini 3B, 4-bit", "gpt-4o-transcribe-diarize": "GPT-4o Transcribe diarize",
    "gpt-4o-transcribe": "GPT-4o Transcribe", "granite-speech-3.3-2b": "Granite Speech 3.3 2B",
    "Qwen2.5-Omni-3B": "Qwen2.5-Omni 3B, 4-bit",
    "MERaLiON-2-10B-ASR": "MERaLiON-2-10B-ASR", "whisper-1": "Whisper-1", "canary-1b-v2": "Canary 1B v2",
}
BASE = "whisper-large-v3-turbo"  # the model WhisperLah adapts


def wer_of(t, s, m):
    # WhisperLah in the mode each set calls for: near-mic on close-talk SASRBench, room mode on the rest
    if m.startswith("whisperlah") and s == "sasrbench" and (s, m + "-near") in t:
        m += "-near"
    return t[(s, m)]["wer"]


def main(out=ROOT / "results" / "whisperlah_ranked_chart.png"):
    rows = [r for r in json.load(open(ROOT / "results" / "table.json")) if r["view"] == "full"]
    t = {(r["set"], r["model"]): r for r in rows}
    full = {s: t[(s, "whisperlah-v3")]["clips"] for s in SETS}
    ranked, unranked = [], []
    for m in LABEL:
        if all((s, m) in t and t[(s, m)]["clips"] == full[s] for s in SETS):
            ranked.append((100 - sum(wer_of(t, s, m) for s in SETS) / len(SETS), m))
        else:
            unranked.append(LABEL[m])
    ranked.sort(reverse=True)

    plt.rcParams["font.family"] = ["Inter", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(14, 8.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x_of = {}
    for x, (v, m) in enumerate(ranked):
        x_of[m] = (x, v)
        colour = OURS if m == "whisperlah-v3" else OURS_OLD if m.startswith("whisperlah") else OTHER
        ax.bar(x, v, 0.74, color=colour, zorder=3, linewidth=0)
        ax.text(x, v - 2.2, f"{v:.1f}", ha="center", va="top", fontsize=11.5, color=INK, zorder=4,
                fontweight="bold" if m == "whisperlah-v3" else "normal")
        img = logo(m, 18)
        if img is not None:
            ax.add_artist(AnnotationBbox(img, (x, -0.028), xycoords=("data", "axes fraction"), box_alignment=(0.5, 1),
                                         frameon=False, annotation_clip=False))
        ours = m.startswith("whisperlah")
        ax.text(x + 0.25, -0.1, LABEL[m], transform=ax.get_xaxis_transform(), rotation=45, rotation_mode="anchor",
                ha="right", va="top", fontsize=10, color=INK if ours else INK2, fontweight="bold" if ours else "normal")
    # the arrow: what the adapter adds on top of the model it starts from
    (xb, vb), (xo, vo) = x_of[BASE], x_of["whisperlah-v3"]
    ax.annotate("", (xo + 0.1, vo + 1.0), (xb, vb + 1.0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=2, mutation_scale=16, shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=0.17"))
    ax.text((xb + xo) / 2, max(vb, vo) + 11.5, f"+{vo - vb:.1f} points over its base model, "
            "Whisper large-v3-turbo", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
    # the honest part, in the space above the short bars
    p1, p2 = [(t[(s, "whisperlah-v3")]["wer"], t[(s, "MERaLiON-2-3B")]["wer"]) for s in ("mnsc_part1", "mnsc_part2")]
    ax.text(len(ranked) - 0.6, 97, "Not first on every set.", ha="right", va="top", fontsize=10.5, color=INK,
            fontweight="bold")
    ax.text(len(ranked) - 0.6, 92.5, f"On the read-speech sets MERaLiON-2-3B is ahead:\nNSC Part 1 {p1[1]:.1f}% vs "
            f"{p1[0]:.1f}% WER, NSC Part 2 {p2[1]:.1f}% vs {p2[0]:.1f}%.\nWhisperLah v3 leads the other five sets.",
            ha="right", va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    ax.set_xlim(-0.6, len(ranked) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.055, 0.955, "Singlish speech recognition: accuracy index", fontsize=16, color=INK, fontweight="bold")
    fig.text(0.055, 0.915, f"Average word accuracy (100 − WER) across all {len(SETS)} Singlish test sets: SASRBench-v1 "
             "and NSC Parts 1 to 6. Same clips and text normaliser for every model. Higher is better.",
             fontsize=10, color=INK2)
    fig.text(0.055, 0.02, "WhisperLah in near-mic mode on SASRBench (close-talk) and room mode elsewhere. Not ranked, "
             f"because not run on all {len(SETS)} sets: {', '.join(unranked)} (see the README table).\n"
             "Logos are trademarks of their owners and only show who made each model.", fontsize=8, color=MUTED,
             linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.885, bottom=0.3)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
