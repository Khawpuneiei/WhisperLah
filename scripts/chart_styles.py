"""Draft chart formats for review: A grouped by model, B leaderboard, C the field chart with logos.

    python scripts/chart_styles.py        # reads results/table.json, writes results/drafts/style_*.png
Logos are in results/drafts/logos (sources in its SOURCES.md).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
LOGOS, OUT = REPO / "results" / "drafts" / "logos", REPO / "results" / "drafts"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Inter", "Arial", "DejaVu Sans"]
INK, INK2, MUTED, GRID, SURFACE = "#1F1E1C", "#5F5B54", "#8A857C", "#E4E0D8", "#FFFFFF"
OURS = "#E2803B"

T = {(r["set"], r["model"]): r["wer"] for r in json.load(open(REPO / "results" / "table.json")) if r["view"] == "full"}
SETS = [("sasrbench", "SASRBench-v1", "close-talk"), ("mnsc_part3", "NSC Part 3", "conversation"),
        ("mnsc_part4", "NSC Part 4", "conversation"), ("mnsc_part5", "NSC Part 5", "conversation"),
        ("mnsc_part6", "NSC Part 6", "conversation"), ("mnsc_part1", "NSC Part 1", "read speech"),
        ("mnsc_part2", "NSC Part 2", "read speech")]
LOGO = {"openai": "openai", "astar": "astar", "alibaba": "qwen", "nvidia": "nvidia", "mistral": "mistral",
        "ibm": "ibm", "hf": "hf", "ours": "lepaq", "indie": "user"}
COLOR = {"ours": OURS, "openai": "#1F1E1C", "astar": "#1D3C8F", "alibaba": "#615CED", "nvidia": "#76B900",
         "mistral": "#F5A623", "ibm": "#0F62FE", "hf": "#9AA3B2", "indie": "#9AA3B2"}


def wer(s, m):
    """WhisperLah: near-mic mode on SASRBench (close-talk, one speaker), room mode everywhere else."""
    if m in ("whisperlah-v3", "whisperlah-v2"):
        return T.get((s, m + "-near")) if s == "sasrbench" else T.get((s, m))
    return T.get((s, m))


def acc(w):
    return None if w is None else max(0.0, 100 - w)


def logo(name, zoom):
    im = Image.open(LOGOS / f"{LOGO[name]}.png").convert("RGBA")
    return OffsetImage(np.asarray(im.resize((96, 96), Image.LANCZOS)), zoom=zoom)


def put_logo(ax, name, xy, zoom, coords=("data", "axes fraction")):
    if name in LOGO:
        ax.add_artist(AnnotationBbox(logo(name, zoom), xy, xycoords=coords, frameon=False, annotation_clip=False))


# ---------------------------------------------------------------- A: grouped by model, one bar per test set
def chart_a():
    models = [("whisperlah-v3", "WhisperLah v3", "ours"), ("gpt-transcribe", "GPT Transcribe", "openai"),
              ("whisper-large-v3", "Whisper large-v3", "openai"), ("MERaLiON-2-3B", "MERaLiON-2-3B", "astar"),
              ("MERaLiON-2-10B-ASR", "MERaLiON-2-10B†", "astar"), ("SenseVoice-Small", "SenseVoice Small", "alibaba"),
              ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B", "nvidia"), ("Voxtral-Mini-3B", "Voxtral Mini 3B", "mistral")]
    conv = ["#7A2E12", "#B4471F", "#E0703F", "#F2A07A", "#F8CDB5"]  # five conversational / close-talk sets
    shades = conv + ["#FFFFFF", "#FFFFFF"]
    hatch = [None] * 5 + ["////", "...."]
    fig, ax = plt.subplots(figsize=(15, 7.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    n, gw = len(SETS), 0.86
    bw = gw / n
    best = {s: max(acc(wer(s, m)) for m, *_ in models if wer(s, m) is not None) for s, *_ in SETS}
    ax.axvspan(-0.5, 0.5, color="#FDF0E6", zorder=0)  # WhisperLah's group
    for gi, (m, label, org) in enumerate(models):
        for si, (s, *_rest) in enumerate(SETS):
            x = gi - gw / 2 + bw * (si + 0.5)
            a = acc(wer(s, m))
            if a is None:
                ax.text(x, 2, "TBD", rotation=90, ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            ax.bar(x, a, bw * 0.92, color=shades[si], hatch=hatch[si], edgecolor="#6B6660" if hatch[si] else "none",
                   linewidth=0.6, zorder=2)
            if a == best[s]:  # most accurate model on this test set
                ax.text(x, a + 0.8, "★", ha="center", va="bottom", fontsize=11,
                        color=OURS if m == "whisperlah-v3" else INK2, zorder=3)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Word accuracy, % (100 − WER)", fontsize=12, color=INK2)
    ax.yaxis.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10, length=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([""] * len(models))
    for gi, (m, label, org) in enumerate(models):
        put_logo(ax, org, (gi, -0.06), 0.24)
        ours = org == "ours"
        ax.text(gi, -0.115, label, transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=11.5 if ours else 10, fontweight="bold" if ours else "normal", color=OURS if ours else INK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=shades[i], hatch=hatch[i],
                             edgecolor="#6B6660" if hatch[i] else "none") for i in range(n)]
    leg = ax.legend(handles, [f"{lab} · {kind}" for _, lab, kind in SETS], title="Test sets", loc="upper left",
                    bbox_to_anchor=(1.005, 1.0), frameon=True, fontsize=9.5, title_fontsize=11, borderpad=1)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.8")
    fig.text(0.06, 0.955, "WhisperLah v3 is the most accurate model on 5 of 7 Singlish test sets",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.93, "★ marks the most accurate model on each test set. WhisperLah leads every close-talk and conversational set;\n"
             "on the two read-speech sets (hatched) it does not: MERaLiON-2-3B (A*STAR) leads NSC Part 1 and Part 2.", fontsize=11, color=INK2, va="top")
    fig.text(0.06, 0.02, "Same clips and text normaliser for every model; clips overlapping any training transcript "
             "excluded. WhisperLah: near-mic mode on SASRBench-v1 (one close-talk speaker), room mode elsewhere. "
             "† NSC Part 3 on a 300-clip sample; other sets not run yet (TBD). IBM Granite and Qwen2.5-Omni score far "
             "lower and are left out here.", fontsize=8.5, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.84, bottom=0.17)
    fig.savefig(OUT / "style_A_grouped.png", facecolor=SURFACE)


# ---------------------------------------------------------------- B: leaderboard, one number per model
def chart_b():
    pool = [("whisperlah-v3", "WhisperLah v3", "ours"), ("whisperlah-v2", "WhisperLah v2", "ours"),
            ("gpt-transcribe", "GPT Transcribe", "openai"), ("gpt-4o-mini-transcribe", "GPT-4o mini Transcribe", "openai"),
            ("gpt-4o-transcribe-diarize", "GPT-4o Transcribe diarize", "openai"), ("gpt-4o-transcribe", "GPT-4o Transcribe", "openai"),
            ("whisper-large-v3", "Whisper large-v3", "openai"), ("whisper-large-v3-turbo", "Whisper large-v3-turbo", "openai"),
            ("distil-large-v3.5", "Distil-Whisper large-v3.5", "hf"), ("MERaLiON-2-3B", "MERaLiON-2-3B", "astar"),
            ("SenseVoice-Small", "SenseVoice Small", "alibaba"), ("Qwen2.5-Omni-3B", "Qwen2.5-Omni 3B", "alibaba"),
            ("parakeet-tdt-0.6b-v3", "Parakeet TDT 0.6B v3", "nvidia"), ("Voxtral-Mini-3B", "Voxtral Mini 3B", "mistral"),
            ("granite-speech-3.3-2b", "Granite Speech 3.3 2B", "ibm"), ("mjwong-turbo-singlish", "mjwong Singlish tune", "indie")]
    rows = []
    for m, label, org in pool:
        a = [acc(wer(s, m)) for s, *_ in SETS]
        if None not in a:
            rows.append((sum(a) / len(a), m, label, org))
    rows.sort(reverse=True)
    fig, ax = plt.subplots(figsize=(15, 7.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(rows))
    for x, (v, m, label, org) in zip(xs, rows):
        c = COLOR[org] if not (org == "ours" and m != "whisperlah-v3") else "#F0BE96"
        ax.bar(x, v, 0.78, color=c, zorder=2)
        ax.text(x, v - 3, f"{v:.1f}", ha="center", va="top", fontsize=11, color="white", fontweight="bold", zorder=3)
        put_logo(ax, org, (x, -0.075), 0.24)
        ax.text(x, -0.15, label, transform=ax.get_xaxis_transform(), ha="right", va="top", rotation=55,
                rotation_mode="anchor", fontsize=10, color=OURS if org == "ours" else INK,
                fontweight="bold" if m == "whisperlah-v3" else "normal")
    i3 = next(i for i, r in enumerate(rows) if r[1] == "whisperlah-v3")
    i2 = next(i for i, r in enumerate(rows) if r[1] == "whisperlah-v2")
    ax.annotate("", (i3, rows[i3][0] + 1.5), (i2, rows[i2][0] + 1.5),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=2.2, connectionstyle="arc3,rad=0.35",
                                mutation_scale=18), zorder=4)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.yaxis.grid(color=GRID, linewidth=1, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.text(0.05, 0.955, "Singlish ASR accuracy index", fontsize=20, color=INK, fontfamily="serif")
    fig.text(0.05, 0.905, "Average word accuracy (100 − WER) over all 7 Singlish test sets: SASRBench-v1 and NSC Parts 1–6, "
             "same clips and text normaliser for every model.\nWhisperLah v3 ranks first overall, but not on every set: "
             "on the read-speech sets (NSC Parts 1 and 2) MERaLiON-2-3B is more accurate.",
             fontsize=11, color=INK2, va="top")
    fig.text(0.05, 0.015, "WhisperLah: near-mic mode on SASRBench-v1, room mode elsewhere. Accuracy floored at 0 where WER "
             "exceeds 100% (Granite, Qwen2.5-Omni on NSC Part 4).\nNot ranked because not yet run on all 7 sets: "
             "MERaLiON-2-10B (A*STAR), Canary 1B v2 (NVIDIA), Whisper-1 (OpenAI API).", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.33)
    fig.savefig(OUT / "style_B_leaderboard.png", facecolor=SURFACE)


# ---------------------------------------------------------------- C: the field chart, logos on the points
def chart_c():
    src = (REPO / "scripts" / "chart_sota.py").read_text(encoding="utf-8")
    org = {"whisper-large-v3-turbo": "openai", "whisper-large-v3": "openai", "distil-large-v3.5": "hf",
           "parakeet-tdt-0.6b-v3": "nvidia", "canary-1b-v2": "nvidia", "Voxtral-Mini-3B": "mistral",
           "granite-speech-3.3-2b": "ibm", "SenseVoice-Small": "alibaba", "Qwen2.5-Omni-3B": "alibaba",
           "MERaLiON-2-3B": "astar", "MERaLiON-2-10B-ASR": "astar", "gpt-transcribe": "openai",
           "gpt-4o-transcribe": "openai", "gpt-4o-mini-transcribe-nochunk": "openai",
           "gpt-4o-transcribe-diarize": "openai", "whisper-1": "openai", "mjwong-turbo-singlish": "indie"}
    dot = 'ax.scatter(x, y, s=60, color=OTHER, zorder=3, edgecolor="white", linewidth=0.8)'
    assert dot in src
    src = src.replace(dot, 'ax.scatter(x, y, s=60, color=OTHER, zorder=3, edgecolor="white", linewidth=0.8) '
                           'if m not in ORG else put_logo(ax, ORG[m], (x, y), 0.38, "data")')
    ns = {"__name__": "chart_sota_logos", "__file__": str(REPO / "scripts" / "chart_sota.py"),
          "ORG": org, "put_logo": put_logo}
    exec(compile(src, "chart_sota.py", "exec"), ns)
    ns["main"](out=OUT / "style_C_field_logos.png")


chart_a()
chart_b()
chart_c()
print("done", OUT)
