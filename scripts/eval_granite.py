"""IBM Granite Speech on the same clips, using the model card's transcription prompt.

Runs in its own environment because it needs transformers >= 4.52:
    .venv-bench/Scripts/python.exe scripts/eval_granite.py --set sasrbench --limit 1200
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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, decode_audio, load_set, save_jsonl  # noqa: E402

SYSTEM = ("Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\n"
          "You are Granite, developed by IBM. You are a helpful AI assistant")
USER = "<|audio|>can you transcribe the speech into a written format?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-speech-3.3-2b")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--name")
    args = ap.parse_args()
    name = args.name or args.model.split("/")[-1]

    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    tok.padding_side = "left"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
                                     tokenize=False, add_generation_prompt=True)

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=name):
        batch = audio[s:s + args.bs]
        inputs = processor([prompt] * len(batch), [torch.from_numpy(w).float() for _, w, _ in batch],
                           device="cuda", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)
        texts = tok.batch_decode(ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        for (i, _, ref), hyp in zip(batch, texts):
            out.append({"idx": i, "ref": ref, "hyp": hyp.strip()})
    secs = time.time() - t0
    save_jsonl(f"{ROOT}/results/{args.set}__{name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    print(json.dumps({"model": args.model, "set": args.set, "n": len(out), "audio_hours": round(hours, 2),
                      "runtime_s": round(secs), "rtf": round(secs / (hours * 3600), 4)}), flush=True)


if __name__ == "__main__":
    main()
