"""Print the README results tables (markdown) from results/table.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].as_posix()

MODELS = [
    ("whisper-large-v3-turbo", "Whisper large-v3-turbo"),
    ("mjwong-turbo-singlish", "Public Singlish fine-tune (mjwong)"),
    ("MERaLiON-2-3B", "MERaLiON-2-3B (A*STAR)"),
    ("whisperlah-v1", "WhisperLah v1"),
    ("whisperlah-v2", "**WhisperLah v2** · room mode"),
    ("whisperlah-v2-near", "**WhisperLah v2** · near-mic mode"),
    ("whisperlah-v2.1", "**WhisperLah v2.1** · room mode"),
    ("whisperlah-v2.1-near", "**WhisperLah v2.1** · near-mic mode"),
]
SETS = [
    ("kopichat", "KopiChat café chats"),
    ("sasrbench", "SASRBench-v1"),
    ("mnsc_part3", "NSC P3 conversation"),
    ("mnsc_part1", "NSC P1 read"),
    ("mnsc_part2", "NSC P2 read, local names"),
    ("mnsc_part4", "NSC P4 conversation"),
    ("mnsc_part5", "NSC P5 conversation"),
    ("mnsc_part6", "NSC P6 conversation"),
    ("ami", "AMI meetings, headset mic (English)"),
    ("ami_sdm", "AMI meetings, table mic (English)"),
    ("librispeech_clean", "LibriSpeech clean (English)"),
    ("librispeech_other", "LibriSpeech other (English)"),
]


def main():
    t = json.load(open(f"{ROOT}/results/table.json"))
    full = {(r["set"], r["model"]): r for r in t if r["view"] == "full"}
    sets = [(s, lab) for s, lab in SETS if any((s, m) in full for m, _ in MODELS)]
    clips = {s: next(full[(s, m)]["clips"] for m, _ in MODELS if (s, m) in full) for s, _ in sets}
    best = {s: min(full[(s, m)]["wer"] for m, _ in MODELS if (s, m) in full) for s, _ in sets}

    print("| Model | " + " | ".join(f"{lab}<br>{clips[s]} clips" for s, lab in sets) + " |")
    print("|---|" + "---|" * len(sets))
    for m, lab in MODELS:
        if not any((s, m) in full for s, _ in sets):
            continue
        cells = []
        for s, _ in sets:
            r = full.get((s, m))
            if r is None:
                cells.append("·")
            else:
                v = f"{r['wer']:.1f}"
                cells.append(f"**{v}**" if r["wer"] == best[s] else v)
        print(f"| {lab} | " + " | ".join(cells) + " |")

    print("\nSingapore discourse particles caught (lah, leh, lor, meh, sia, hor, mah, liao …), share of those in the reference:\n")
    print("| Model | " + " | ".join(lab for s, lab in sets[:2]) + " |")
    print("|---|---|---|")
    for m, lab in MODELS:
        cells = [f"{full[(s, m)]['particle_recall']:.0f}%" if (s, m) in full and full[(s, m)]["particle_recall"] is not None
                 else "·" for s, _ in sets[:2]]
        if any(c != "·" for c in cells):
            print(f"| {lab} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
