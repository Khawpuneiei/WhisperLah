"""Deck chart: WER (lower is better) and Singlish particle recall (higher is better).

Two panels, one measure each, shared model order. Baselines in a warm neutral,
our model in the deck accent; every bar carries its value so identity and
magnitude never depend on colour alone.

    python scripts/chart.py            # reads results/table.json
"""
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].as_posix()
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FFFFFF"
BASE, OURS = "#CFC9BE", "#E2803B"

LABELS = {
    "whisper-large-v3-turbo": "Whisper large-v3-turbo\n(open, no adaptation)",
    "mjwong-turbo-singlish": "Public Singlish fine-tune\n(mjwong, full fine-tune)",
    "gpt-4o-transcribe-diarize": "GPT-4o Transcribe\n(current production API)",
}


def label(name):
    return LABELS.get(name, "Rapport LoRA (ours)\n52 h Singlish, 1 laptop GPU")


def is_ours(name):
    return name not in LABELS


def rounded_barh(ax, y, w, h, color):
    """4px-style rounded data end, square at the baseline."""
    r = min(h * 0.18, w * 0.5) if w > 0 else 0
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=1, linewidth=0, facecolor=color))
    if w > r:  # square off the baseline end
        ax.add_patch(plt.Rectangle((0, y - h / 2), r, h, linewidth=0, facecolor=color))


def panel(ax, rows, key, title, fmt, xmax):
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        v = r[key] or 0
        rounded_barh(ax, y, v, 0.42, OURS if is_ours(r["model"]) else BASE)
        ax.text(v + xmax * 0.015, y, fmt(v), va="center", ha="left", fontsize=13,
                color=INK, fontweight="bold" if is_ours(r["model"]) else "normal")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label(r["model"]) for r in reversed(rows)], fontsize=11, color=INK2)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def main(set_name="mnsc_part3", out=f"{ROOT}/results/singlish_asr_chart.png"):
    table = json.load(open(f"{ROOT}/results/table.json"))
    rows = [t for t in table if t["set"] == set_name and t["view"] == "full"]
    order = ["gpt-4o-transcribe-diarize", "whisper-large-v3-turbo", "mjwong-turbo-singlish"]
    rows.sort(key=lambda t: order.index(t["model"]) if t["model"] in order else 99)
    plt.rcParams["font.family"] = "Arial"
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.95 * len(rows) + 1.3), dpi=200,
                               gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    panel(a, rows, "wer", "Word error rate  (lower is better)", lambda v: f"{v:.1f}%",
          max(t["wer"] for t in rows) * 1.22)
    panel(b, rows, "particle_recall", "Singlish particles caught  (lah, leh, lor …)",
          lambda v: f"{v:.0f}%", 100)
    b.set_yticklabels([])
    clips = rows[0]["clips"]
    fig.text(0.01, 0.01, f"IMDA National Speech Corpus Part 3, held-out MNSC test split: {clips} two-speaker "
             "conversation clips (≈7.7 h). Same normaliser and decoding for every model.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:])
