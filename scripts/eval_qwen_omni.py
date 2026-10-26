"""Alibaba Qwen2.5-Omni on the same clips, text output only.

Only the "thinker" (audio/vision encoders + language model) is loaded; the
speech talker is never built. The thinker is ~9 GB in bf16, more than the
4060's 8 GB, so its language model is loaded in 4-bit NF4 by default. Uses
the model card's required system prompt and its English ASR instruction.
Runs in its own environment because it needs
transformers >= 4.52 and torchvision:
    .venv-bench/Scripts/python.exe scripts/eval_qwen_omni.py --set sasrbench --limit 1200
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
from transformers import BitsAndBytesConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, decode_audio, load_set, save_jsonl  # noqa: E402

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")
PROMPT = "Transcribe the English audio into text without any punctuation marks."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--name", default="Qwen2.5-Omni-3B")
    ap.add_argument("--bf16", action="store_true", help="no quantisation (the 3B thinker is ~9 GB, more than 8 GB)")
    args = ap.parse_args()

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
    if not args.bf16:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["audio_tower", "visual", "lm_head"])
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model, **kw).eval()
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    # the thinker's generation config has no stop token: without this it writes on past <|im_end|>
    tok = processor.tokenizer
    stop = dict(eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")],
                pad_token_id=tok.pad_token_id)

    rows = load_set(args.set, args.limit)
    audio = sorted(((i, decode_audio(b), ref) for i, b, ref in rows), key=lambda r: -len(r[1]))
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=args.name):
        batch = audio[s:s + args.bs]
        inputs = processor(text=[text] * len(batch), audio=[w for _, w, _ in batch], return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda", torch.bfloat16) if v.dtype == torch.float32 else v.to("cuda")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=500, do_sample=False, **stop)
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
