"""WER of a Whisper checkpoint (optionally + LoRA adapter) on a Singlish test set.

    python scripts/eval_hf.py --model openai/whisper-large-v3-turbo --set mnsc_part3 --name whisper-turbo
    python scripts/eval_hf.py --model openai/whisper-large-v3-turbo --adapter runs/lora/final --set mnsc_part3 --name ours-lora
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import ROOT, Normalizer, decode_audio, load_set, save_jsonl, wer_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--set", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--greedy", action="store_true",
                    help="plain greedy decoding, no temperature fallback")
    ap.add_argument("--prompt", help='Whisper text prompt, e.g. "Near speaker only." for WhisperLah near-mic mode')
    ap.add_argument("--len-cap", action="store_true",
                    help="cap output at 8 tokens per second of audio (+24): stops run-on text past the end of a clip")
    args = ap.parse_args()
    torch.manual_seed(0)  # fallback samples at T>0; keep reruns identical

    rows = load_set(args.set, args.limit)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    audio = [(i, decode_audio(b), ref) for i, b, ref in rows]
    # longest first, so the first batch tells us right away if memory is short
    audio.sort(key=lambda r: -len(r[1]))
    long = sum(len(w) > 30 * 16000 for _, w, _ in audio)
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h, {long} over 30 s", flush=True)

    out, t0 = [], time.time()
    for s in tqdm(range(0, len(audio), args.bs), desc=args.name):
        batch = audio[s:s + args.bs]
        waves = [w for _, w, _ in batch]
        is_long = any(len(w) > 30 * 16000 for w in waves)
        feats = processor(waves, sampling_rate=16000, return_tensors="pt",
                          truncation=not is_long, padding="longest" if is_long else "max_length",
                          return_attention_mask=is_long)
        kw = dict(language="en", task="transcribe", num_beams=1)
        if not args.greedy:
            # Whisper's standard guard against repetition loops: re-decode at a
            # higher temperature when the output compresses too well (a loop).
            # Applied to every model in the table alike.
            kw.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35, do_sample=False)
        if is_long:  # Whisper long-form: sequential 30 s windows
            kw.update(attention_mask=feats.attention_mask.to("cuda"), return_timestamps=True,
                      condition_on_prev_tokens=False)
        else:
            cap = 440
            if args.len_cap:
                cap = min(440, int(8 * max(len(w) for w in waves) / 16000) + 24)
            kw.update(max_new_tokens=cap)
        if args.prompt:
            kw["prompt_ids"] = processor.get_prompt_ids(args.prompt, return_tensors="pt").to("cuda")
            # prompt + 4 start tokens + output must fit Whisper's 448 decoder positions
            if "max_new_tokens" in kw:
                kw["max_new_tokens"] = min(kw["max_new_tokens"], 448 - 4 - len(kw["prompt_ids"]) - 1)
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to("cuda", torch.float16), **kw)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for (i, _, ref), hyp in zip(batch, texts):
            out.append({"idx": i, "ref": ref, "hyp": hyp.strip()})
    secs = time.time() - t0

    norm = Normalizer()
    pairs = [(norm(r["ref"]), norm(r["hyp"])) for r in out]
    pairs = [(r, h) for r, h in pairs if r]
    rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
    rep.update(model=args.model, adapter=args.adapter, set=args.set, name=args.name,
               decoding=("greedy" if args.greedy else "greedy+temperature_fallback") + ("+len_cap" if args.len_cap else ""),
               prompt=args.prompt,
               audio_hours=round(hours, 2), runtime_s=round(secs), rtf=round(secs / (hours * 3600), 4))
    print(json.dumps(rep, indent=1), flush=True)

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    save_jsonl(f"{ROOT}/results/{args.set}__{args.name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    path = f"{ROOT}/results/summary.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()
