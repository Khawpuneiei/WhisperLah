"""Figures for the WhisperLah technical report, drawn from results/table.json only.

    python docs/technical-report/figures.py      # writes docs/technical-report/fig_*.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INK, MUTED, GRID, SURFACE = "#1F1E1C", "#8A857C", "#ECE8E1", "#FFFFFF"
OURS, OTHER = "#E2803B", "#B4BCC8"
SUB, DEL, INS = "#5F5B54", "#E2803B", "#B4BCC8"

rows = [r for r in json.load(open(ROOT / "results" / "table.json")) if r["view"] == "full"]
T = {(r["set"], r["model"]): r for r in rows}

plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color=GRID, zorder=0)
    ax.set_axisbelow(True)


def errors():
    """Substitutions, deletions and insertions on NSC Part 3, per 100 reference words."""
    models = [("whisper-large-v3-turbo", "Whisper large-v3-turbo (base)"), ("MERaLiON-2-3B", "MERaLiON-2-3B"),
              ("gpt-transcribe", "GPT Transcribe"), ("distil-large-v3.5", "Distil-Whisper large-v3.5"),
              ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3"), ("whisperlah-v2", "WhisperLah v2"),
              ("whisperlah-v3", "WhisperLah v3")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=220)
    for i, (m, label) in enumerate(models):
        r = T[("mnsc_part3", m)]
        n = r["sub"] + r["del"] + r["ins"]
        scale = r["wer"] / n  # errors -> WER points
        left = 0
        for k, c, name in (("sub", SUB, "Substitutions"), ("del", DEL, "Deletions"), ("ins", INS, "Insertions")):
            w = r[k] * scale
            ax.barh(i, w, left=left, color=c, height=0.62, label=name if i == 0 else None, zorder=3)
            if w > 2.2:
                ax.text(left + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if c != INS else INK)
            left += w
        ax.text(left + 0.4, i, f"{r['wer']:.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(models)), [l for _, l in models])
    ax.set_xlabel("WER points on NSC Part 3 (997 clips)")
    ax.set_xlim(0, 32)
    style(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_errors.png", facecolor=SURFACE)


def probes():
    """v3 checkpoints on the fixed probe clips (300 NSC Part 3, 300 SASRBench before the overlap filter)."""
    steps = list(range(1000, 7001, 1000))
    p3 = [T[("mnsc_part3", f"probe-step{s}")]["wer"] for s in steps]
    sa = [T[("sasrbench", f"probe-step{s}")]["wer"] for s in steps]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=220)
    ax.plot(steps, p3, "o-", color=OURS, label=f"NSC Part 3 probe ({T[('mnsc_part3', 'probe-step1000')]['clips']} clips)", zorder=3)
    ax.plot(steps, sa, "s-", color=SUB, label=f"SASRBench probe, room mode ({T[('sasrbench', 'probe-step1000')]['clips']} clips)", zorder=3)
    for s, a, b in zip(steps, p3, sa):
        ax.annotate(f"{a:.1f}", (s, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=OURS)
        ax.annotate(f"{b:.1f}", (s, b), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5, color=SUB)
    ax.set_xlabel("v3 training step (batch 16)")
    ax.set_ylabel("WER, %")
    ax.set_ylim(15, 22)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_probes.png", facecolor=SURFACE)


def particles():
    """Discourse-particle recall on NSC Part 3."""
    models = [("whisperlah-v3", "WhisperLah v3"), ("gpt-transcribe", "GPT Transcribe"),
              ("whisper-large-v3", "Whisper large-v3"), ("gpt-4o-mini-transcribe-nochunk", "GPT-4o mini Transcribe"),
              ("Voxtral-Mini-3B", "Voxtral Mini 3B"), ("SenseVoice-Small", "SenseVoice Small"),
              ("whisper-large-v3-turbo", "Whisper large-v3-turbo (base)"), ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3"),
              ("mjwong-turbo-singlish", "Singlish fine-tune (mjwong)"), ("MERaLiON-2-3B", "MERaLiON-2-3B")]
    vals = [T[("mnsc_part3", m)]["particle_recall"] for m, _ in models]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=220)
    y = range(len(models))[::-1]
    ax.barh(list(y), vals, color=[OURS if m.startswith("whisperlah") else OTHER for m, _ in models], height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(list(y), [l for _, l in models])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of reference discourse particles written in the output, NSC Part 3")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_particles.png", facecolor=SURFACE)


if __name__ == "__main__":
    errors()
    probes()
    particles()
