"""Draft a meeting transcript for hand correction: WhisperLah (CTranslate2,
word timestamps) + community-1 with our fine-tuned segmentation.

    .venv-serve/Scripts/python.exe scripts/diar/draft_meeting.py data/meetings/audio/KP-meet1.wav \
        --model models/whisperlah-v3-ct2 --params data/diar/pipeline_best.json --out data/meetings/drafts

Each word takes the speaker whose (exclusive) diarization turn overlaps it
most, or the nearest turn if none does. Segments are cut at every speaker
change and at pauses over --max-pause, and are never longer than --max-len
(a long one is split at its longest inner pause, recursively).

Writes <out>/<name>.json:
    {"audio", "duration", "asr", "diarization", "speakers", "segments": [{id, start, end, speaker, text}]}
Meeting audio, drafts and gold labels live under data/ (gitignored) and are
never committed. Needs the serving code in scripts/serve/ (fallback.py, server.py).

Clustering: pass --params data/diar/none.json (any missing file) for pyannote's
defaults. The dev-tuned pipeline_best.json over-splits long meetings (30-50
speakers on a 30-min recording), because the synthetic dev clips are ~12 s.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, (ROOT / "scripts" / "serve").as_posix())
sys.path.insert(0, (ROOT / "scripts" / "diar").as_posix())
import server  # noqa: E402  (sets PATH for CTranslate2's CUDA libs on Windows)
import fallback  # noqa: E402
from eval_der import hf_token  # noqa: E402


def label_words(words, turns):
    """[(start, end, text)] x [(start, end, speaker)] -> [(start, end, text, speaker)]"""
    turns = sorted(turns)
    out, j = [], 0
    for ws, we, text in words:
        while j < len(turns) and turns[j][1] < ws:
            j += 1
        best, best_ov, k = None, 0.0, j
        while k < len(turns) and turns[k][0] <= we:
            ov = min(we, turns[k][1]) - max(ws, turns[k][0])
            if ov > best_ov:
                best, best_ov = turns[k][2], ov
            k += 1
        if best is None and turns:
            mid = (ws + we) / 2
            best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        out.append((ws, we, text, best or "SPEAKER_00"))
    return out


def split_long(words, max_len):
    if len(words) < 2 or words[-1][1] - words[0][0] <= max_len:
        return [words]
    gaps = [words[i + 1][0] - words[i][1] for i in range(len(words) - 1)]
    i = int(np.argmax(gaps)) + 1
    return split_long(words[:i], max_len) + split_long(words[i:], max_len)


def segment(labelled, max_pause, max_len):
    groups = []
    for w in labelled:
        if groups and groups[-1][-1][3] == w[3] and w[0] - groups[-1][-1][1] <= max_pause:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [p for g in groups for p in split_long(g, max_len)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="16 kHz mono WAVs")
    ap.add_argument("--model", default="models/whisperlah-v3-ct2")
    ap.add_argument("--asr-name", default="whisperlah-v3-room")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--params", default="data/diar/pipeline_best.json",
                    help="tune_clustering.py output; missing file = default clustering")
    ap.add_argument("--segmentation", default="runs/diar/seg-ft/epoch=14.ckpt")
    ap.add_argument("--max-pause", type=float, default=0.6)
    ap.add_argument("--max-len", type=float, default=25.0)
    ap.add_argument("--out", default="data/meetings/drafts")
    args = ap.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio
    from pyannote.audio import Model, Pipeline

    model = WhisperModel(str(ROOT / args.model), device="cuda", compute_type=args.compute)
    fallback.install(model)
    batched = BatchedInferencePipeline(model=model)

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token())
    params_path = ROOT / args.params
    if params_path.exists():
        tuned = json.loads(params_path.read_text())
        args.segmentation = tuned.get("segmentation", args.segmentation)
        pipe.instantiate(tuned["params"])
    pipe._segmentation.model = Model.from_pretrained(str(ROOT / args.segmentation))
    pipe.to(torch.device("cuda"))
    p = pipe.parameters(instantiated=True)
    epoch = Path(args.segmentation).stem.split("=")[-1].lstrip("0") or "0"
    dia_name = (f"community-1 + seg-ft epoch {epoch}, threshold={p['clustering']['threshold']:g}, "
                f"Fa={p['clustering']['Fa']:g}, Fb={p['clustering']['Fb']:g}, "
                f"min_duration_off={p['segmentation']['min_duration_off']:g}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.audio):
        t0 = time.time()
        wav = decode_audio(str(path), sampling_rate=16000)
        duration = len(wav) / 16000

        common = dict(language=None, task="transcribe", beam_size=1, best_of=1, word_timestamps=True)
        segs, _ = batched.transcribe(wav, temperature=0.0, condition_on_previous_text=False,
                                     batch_size=args.batch_size, **common)
        pieces = fallback.pieces_from_segments(list(segs))
        pieces, redecoded = fallback.redecode_loops(model, wav, pieces, **common)
        words = sorted(w for pc in pieces for w in pc.words if w[2])
        asr_s = time.time() - t0

        t1 = time.time()
        out = pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": 16000, "uri": path.stem})
        ann = getattr(out, "exclusive_speaker_diarization", None) or out.speaker_diarization
        turns = [(s.start, s.end, spk) for s, _, spk in ann.itertracks(yield_label=True)]
        dia_s = time.time() - t1

        groups = segment(label_words(words, turns), args.max_pause, args.max_len)
        names = {}
        segments = []
        for g in sorted(groups, key=lambda g: g[0][0]):
            spk = names.setdefault(g[0][3], f"S{len(names) + 1}")
            segments.append({"id": len(segments) + 1, "start": round(g[0][0], 2), "end": round(g[-1][1], 2),
                             "speaker": spk, "text": " ".join(w[2] for w in g).strip()})
        doc = {"audio": path.name, "duration": round(duration, 2), "asr": args.asr_name,
               "diarization": dia_name, "speakers": list(names.values()), "segments": segments}
        (out_dir / f"{path.stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"meeting": path.stem, "duration_s": round(duration, 1), "segments": len(segments),
                          "speakers": len(names), "words": len(words), "redecoded": redecoded,
                          "asr_s": round(asr_s, 1), "diarize_s": round(dia_s, 1),
                          "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
