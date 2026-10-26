"""Diarization error rate of a pyannote pipeline on one of our RTTM sets.

    .venv-diar/Scripts/python.exe scripts/diar/eval_der.py --split test --name community1 [--device cuda]
    .venv-diar/Scripts/python.exe scripts/diar/eval_der.py --split test --name ft --segmentation runs/diar/seg-ft/epoch=09.ckpt
    .venv-diar/Scripts/python.exe scripts/diar/eval_der.py --split test --name ft-tuned --params data/diar/pipeline_best.json

Reads HF_TOKEN from .env.hf.txt (or .env.hf.txt.txt). Audio is passed in memory
because torchcodec cannot decode files on this Windows setup. Scores with
pyannote.metrics, no forgiveness collar and overlap counted, both as reported
for community-1, plus the 0.25 s collar figure that older papers use.
Writes results/diar_<split>__<name>.json and the hypothesis RTTM next to it.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).resolve().parents[2]


def hf_token():
    for name in (".env.hf.txt", ".env.hf.txt.txt"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def load_rttm(path):
    refs = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            ann = refs.setdefault(f[1], Annotation(uri=f[1]))
            ann[Segment(float(f[3]), float(f[3]) + float(f[4]))] = f[7]
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--segmentation", help="fine-tuned segmentation checkpoint to swap into community-1")
    ap.add_argument("--params", help="pipeline_best.json from tune_clustering.py (also sets --segmentation)")
    ap.add_argument("--oracle-speakers", action="store_true", help="pass the reference speaker count")
    args = ap.parse_args()

    d = ROOT / "data" / "diar" / args.split
    refs = load_rttm(d / f"{args.split}.rttm")
    names = (d / f"{args.split}.lst").read_text().split()[: args.limit]

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token())
    if args.params:
        tuned = json.loads((ROOT / args.params).read_text())
        args.segmentation = args.segmentation or str(ROOT / tuned["segmentation"])
        pipe.instantiate(tuned["params"])
    if args.segmentation:
        from pyannote.audio import Model
        pipe._segmentation.model = Model.from_pretrained(args.segmentation)
    pipe.to(torch.device(args.device))

    der0, der25 = DiarizationErrorRate(collar=0.0), DiarizationErrorRate(collar=0.25)
    hyp_lines, audio_s, t0 = [], 0.0, time.time()
    for k, name in enumerate(names):
        sr, wav = wavfile.read(d / "wav" / f"{name}.wav")  # .venv-diar has no soundfile
        wav = wav.astype(np.float32) / 32768  # make_synth writes 16-bit PCM
        audio_s += len(wav) / sr
        k = len(refs[name].labels()) if args.oracle_speakers else None
        try:
            out = pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr}, num_speakers=k)
        except ValueError:  # fewer embeddings than reference speakers: KMeans cannot force k
            out = pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr})
        hyp = getattr(out, "speaker_diarization", out)  # pyannote 4 returns an output object
        hyp.uri = name
        der0(refs[name], hyp)
        der25(refs[name], hyp)
        hyp_lines += [f"SPEAKER {name} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> {spk} <NA> <NA>"
                      for s, _, spk in hyp.itertracks(yield_label=True)]
        if k % 25 == 0:
            print(f"{k}/{len(names)} DER so far {100 * abs(der0):.2f}%", flush=True)

    comp = der0.accumulated_  # seconds per error type, summed over files
    total = comp.get("total", 0.0) or 1.0
    res = {"split": args.split, "name": args.name, "files": len(names),
           "der": round(100 * abs(der0), 2), "der_collar_0.25": round(100 * abs(der25), 2),
           "missed": round(100 * comp.get("missed detection", 0) / total, 2),
           "false_alarm": round(100 * comp.get("false alarm", 0) / total, 2),
           "confusion": round(100 * comp.get("confusion", 0) / total, 2),
           "rtf": round((time.time() - t0) / audio_s, 4), "device": args.device,
           "params": pipe.parameters(instantiated=True), "oracle_speakers": args.oracle_speakers}
    out = ROOT / "results" / f"diar_{args.split}__{args.name}"
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".rttm").write_text("\n".join(hyp_lines) + "\n")
    print(json.dumps(res), "\nDER_DONE", flush=True)


if __name__ == "__main__":
    main()
