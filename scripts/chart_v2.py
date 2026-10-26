"""Headline chart: WER on the two Singlish benchmarks, every model scored alike.

    python scripts/chart_v2.py            # reads results/table.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1].as_posix()
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#ECE8E1", "#FFFFFF"
BASE, OURS = "#CFC9BE", "#E2803B"

# (result name, label): every model in the results table, open models then the OpenAI API;
# WhisperLah rows are drawn in the accent colour
ROWS = [
    ("whisper-large-v3-turbo", "Whisper large-v3-turbo (base)"),
    ("whisper-large-v3", "Whisper large-v3"),
    ("distil-large-v3.5", "Distil-Whisper large-v3.5"),
    ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3 (NVIDIA)"),
    ("canary-1b-v2", "Canary 1B v2 (NVIDIA)"),
    ("Voxtral-Mini-3B", "Voxtral Mini 3B, 4-bit (Mistral)"),
    ("granite-speech-3.3-2b", "Granite Speech 3.3 2B (IBM)"),
    ("SenseVoice-Small", "SenseVoice Small (Alibaba)"),
    ("Qwen2.5-Omni-3B", "Qwen2.5-Omni 3B, 4-bit (Alibaba)"),
    ("mjwong-turbo-singlish", "Public Singlish fine-tune"),
    ("MERaLiON-2-3B", "MERaLiON-2-3B (A*STAR)"),
    ("MERaLiON-2-10B-ASR", "MERaLiON-2-10B-ASR, 4-bit (A*STAR)"),
    ("gpt-transcribe", "GPT Transcribe (OpenAI)"),
    ("gpt-4o-transcribe", "GPT-4o Transcribe (OpenAI)"),
    ("gpt-4o-mini-transcribe-nochunk", "GPT-4o mini Transcribe (OpenAI)"),
    ("gpt-4o-transcribe-diarize", "GPT-4o Transcribe, diarize (OpenAI)"),
    ("whisper-1", "Whisper-1 (OpenAI)"),
    ("whisperlah-v2", "WhisperLah v2 · room mode"),
    ("whisperlah-v2-near", "WhisperLah v2 · near-mic mode"),
]
CUT = 45  # longer bars are cut here (marked with a break) so the close race stays readable
PANELS = [
    ("sasrbench", "SASRBench-v1  ·  single speaker, close-talk mic", ROWS),
    ("mnsc_part3", "NSC Part 3  ·  two-person conversation", ROWS),
]


def bar(ax, y, w, h, color):
    r = min(h * 0.18, w * 0.5) if w > 0 else 0
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0, facecolor=color))
    if w > r:
        ax.add_patch(plt.Rectangle((0, y - h / 2), r, h, linewidth=0, facecolor=color))


def main(version="whisperlah-v2", out=f"{ROOT}/results/whisperlah_v2_chart.png"):
    table = json.load(open(f"{ROOT}/results/table.json"))
    wer = {(t["set"], t["model"]): t for t in table if t["view"] == "full"}
    tag = version.replace("whisperlah-", "WhisperLah ")
    panels = [(s, title, [(k.replace("whisperlah-v2", version), lab.replace("WhisperLah v2", tag)) for k, lab in rows])
              for s, title, rows in PANELS]
    panels = [(s, title, [(k, lab) for k, lab in rows if (s, k) in wer]) for s, title, rows in panels]
    panels = [p for p in panels if p[2]]
    xmax = min(CUT, max(wer[(s, k)]["wer"] for s, _, rows in panels for k, _ in rows)) * 1.2
    plt.rcParams["font.family"] = "Arial"
    n = max(len(rows) for _, _, rows in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 0.62 * n + 1.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, (s, title, rows) in zip(axes if len(panels) > 1 else [axes], panels):
        best = min(wer[(s, k)]["wer"] for k, _ in rows)
        for i, (k, lab) in enumerate(rows):
            y = len(rows) - 1 - i
            v = wer[(s, k)]["wer"]
            ours = k.startswith("whisperlah")
            bar(ax, y, min(v, CUT), 0.5, OURS if ours else BASE)
            if v > CUT:  # a white double slash across the end of a cut bar
                for dx in (-0.012, 0.0):
                    ax.plot([CUT * (0.93 + dx) - 0.4, CUT * (0.93 + dx) + 0.4], [y - 0.32, y + 0.32],
                            color=SURFACE, linewidth=2.2, solid_capstyle="butt", zorder=3)
            ax.text(min(v, CUT) + xmax * 0.015, y, f"{v:.1f}%", va="center", ha="left", fontsize=12, color=INK,
                    fontweight="bold" if v == best else "normal")
        ax.set_xlim(0, xmax)
        ax.set_xticks(range(0, int(min(xmax, CUT)) + 1, 10))  # no scale past the cut
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        clips = wer[(s, rows[0][0])]["clips"]
        # † marks a model scored on a sample of the clips (MERaLiON-2-10B on NSC Part 3)
        ax.set_yticklabels([lab + ("†" if wer[(s, k)]["clips"] != clips else "") for k, lab in reversed(rows)],
                           fontsize=10.5, color=INK2)
        ax.set_title(f"{title}\nword error rate, lower is better  ·  {clips} clips", loc="left",
                     fontsize=11.5, color=INK, pad=10, linespacing=1.5)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.01, 0.01, "Same clips and text normaliser for every model; each run with its own recommended settings "
             "(see Benchmark protocol). Clips whose text overlaps training data excluded for all.\n"
             f"† scored on a 300-clip sample. Bars over {CUT}% are cut short (//); the number is the full WER.",
             fontsize=8.5, color=MUTED, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=4)
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:])
