"""NVIDIA NeMo ASR models (Parakeet, Canary) on the same clips, through onnx-asr.

NeMo itself is painful on Windows; onnx-asr runs NVIDIA's exported ONNX graphs
on onnxruntime-gpu with no NeMo install. Lives in its own environment:
    .venv-bench/Scripts/python.exe scripts/eval_onnxasr.py --model nemo-parakeet-tdt-0.6b-v3 --set sasrbench --limit 1200 --name parakeet-tdt-0.6b-v3
Writes results/<set>__<name>.jsonl in the same format as eval_hf.py, so
report.py scores it with the same normaliser as every other model.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import onnx_asr
import onnxruntime
from tqdm import tqdm

onnxruntime.preload_dlls()  # CUDA/cuDNN DLLs from the torch wheel in this venv

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, decode_audio, load_set, save_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nemo-parakeet-tdt-0.6b-v3", help="onnx-asr model name")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--name", required=True)
    ap.add_argument("--language", help="source language for multilingual models such as Canary")
    ap.add_argument("--path", help="local model folder (its config.json overrides onnx-asr's defaults)")
    args = ap.parse_args()

    model = onnx_asr.load_model(args.model, args.path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    kw = {"language": args.language} if args.language else {}

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b).astype(np.float32), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=args.name):
        batch = audio[s:s + args.bs]
        texts = model.recognize([w for _, w, _ in batch], sample_rate=16000, **kw)
        for (i, _, ref), hyp in zip(batch, texts):
            out.append({"idx": i, "ref": ref, "hyp": hyp.strip()})
    secs = time.time() - t0
    save_jsonl(f"{ROOT}/results/{args.set}__{args.name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    print(json.dumps({"model": args.model, "set": args.set, "n": len(out), "audio_hours": round(hours, 2),
                      "runtime_s": round(secs), "rtf": round(secs / (hours * 3600), 4)}), flush=True)


if __name__ == "__main__":
    main()
