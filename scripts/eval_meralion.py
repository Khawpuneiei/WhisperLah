"""MERaLiON-2 (A*STAR) on the same clips, using the authors' recommended prompt.

Runs in its own environment because MERaLiON-2 pins transformers==4.50.1:
    .venv-meralion/Scripts/python.exe scripts/eval_meralion.py --set sasrbench --limit 1200
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

PROMPT = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MERaLiON/MERaLiON-2-3B")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--name")
    ap.add_argument("--load-in-4bit", action="store_true", help="NF4 weights via bitsandbytes: the 10B fits in 8 GB")
    args = ap.parse_args()
    name = args.name or args.model.split("/")[-1]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    kw = dict(use_safetensors=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
        model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, **kw).eval()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, **kw).to("cuda").eval()

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    conv = [{"role": "user", "content": PROMPT.format(query="Please transcribe this speech.")}]
    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=name):
        batch = audio[s:s + args.bs]
        chat = processor.tokenizer.apply_chat_template([conv] * len(batch), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=chat, audios=[w for _, w, _ in batch])
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
                if v.dtype == torch.float32:
                    inputs[k] = inputs[k].to(torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        texts = processor.batch_decode(ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        for (i, _, ref), hyp in zip(batch, texts):
            out.append({"idx": i, "ref": ref, "hyp": hyp.strip()})
    secs = time.time() - t0
    save_jsonl(f"{ROOT}/results/{args.set}__{name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    print(json.dumps({"model": args.model, "set": args.set, "n": len(out), "audio_hours": round(hours, 2),
                      "runtime_s": round(secs), "rtf": round(secs / (hours * 3600), 4)}), flush=True)


if __name__ == "__main__":
    main()
