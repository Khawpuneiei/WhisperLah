"""Mistral Voxtral on the same clips, through transformers' transcription request API.

Voxtral Mini 3B is ~9 GB in bf16, more than the 4060's 8 GB, so the language
model is loaded in 4-bit NF4 (audio encoder and projector stay bf16). Runs in
its own environment because it needs transformers >= 4.54:
    .venv-bench/Scripts/python.exe scripts/eval_voxtral.py --set sasrbench --limit 1200
Writes results/<set>__<name>.jsonl in the same format as eval_hf.py, so
report.py scores it with the same normaliser as every other model.
"""
import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, decode_audio, load_set, save_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--name", default="Voxtral-Mini-3B")
    ap.add_argument("--bf16", action="store_true", help="no quantisation (needs more than 8 GB)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
    if not args.bf16:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"])
    model = VoxtralForConditionalGeneration.from_pretrained(args.model, **kw).eval()

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=args.name):
        batch = audio[s:s + args.bs]
        inputs = processor.apply_transcription_request(language="en", audio=[w for _, w, _ in batch],
                                                       model_id=args.model, sampling_rate=16000,
                                                       format=["wav"] * len(batch))
        inputs = inputs.to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        texts = processor.batch_decode(ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        for (i, _, ref), hyp in zip(batch, texts):
            out.append({"idx": i, "ref": ref, "hyp": hyp.strip()})
    secs = time.time() - t0
    save_jsonl(f"{ROOT}/results/{args.set}__{args.name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    print(json.dumps({"model": args.model, "set": args.set, "n": len(out), "audio_hours": round(hours, 2),
                      "runtime_s": round(secs), "rtf": round(secs / (hours * 3600), 4),
                      "quant": "bf16" if args.bf16 else "nf4"}), flush=True)


if __name__ == "__main__":
    main()
