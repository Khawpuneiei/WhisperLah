"""Alibaba SenseVoice Small on the same clips, through funasr.

English, inverse text normalisation on (digits like the other models), and
SenseVoice's <|...|> language/emotion/event tags stripped (by regex: funasr's
rich_transcription_postprocess turns them into emoji). Lives in its own
environment:
    .venv-bench/Scripts/python.exe scripts/eval_sensevoice.py --set sasrbench --limit 1200
Writes results/<set>__<name>.jsonl in the same format as eval_hf.py, so
report.py scores it with the same normaliser as every other model.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
from funasr import AutoModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, decode_audio, load_set, save_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FunAudioLLM/SenseVoiceSmall")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--name", default="SenseVoice-Small")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = AutoModel(model=args.model, hub="hf", device=args.device, disable_update=True)

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b).astype(np.float32), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=args.name):
        batch = audio[s:s + args.bs]
        res = model.generate(input=[w for _, w, _ in batch], fs=16000, language="en", use_itn=True,
                             batch_size=len(batch), disable_pbar=True)
        for (i, _, ref), r in zip(batch, res):
            out.append({"idx": i, "ref": ref, "hyp": re.sub(r"<\|[^|]*\|>", "", r["text"]).strip()})
    secs = time.time() - t0
    save_jsonl(f"{ROOT}/results/{args.set}__{args.name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    print(json.dumps({"model": args.model, "set": args.set, "n": len(out), "audio_hours": round(hours, 2),
                      "runtime_s": round(secs), "rtf": round(secs / (hours * 3600), 4)}), flush=True)


if __name__ == "__main__":
    main()
