"""LoRA fine-tune of Whisper large-v3-turbo on Singlish conversation (MNSC Part 3).

Sized for one 8 GB laptop GPU: frozen bf16 base, fp32 LoRA weights on every
attention and MLP projection in encoder and decoder, gradient checkpointing.
The encoder is adapted on purpose: the gap on Singlish is mostly acoustic
(accent, particles, fast turn-taking), not vocabulary.

    python scripts/train_lora.py --steps 20 --name smoke        # throughput check
    python scripts/train_lora.py --steps 1200 --name lora-r32   # real run
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

from common import ROOT

BASE = "openai/whisper-large-v3-turbo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = f"{ROOT}/runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    feats = np.load(f"{ROOT}/data/train_feats.npy", mmap_mode="r")
    texts = json.load(open(f"{ROOT}/data/train_text.json", encoding="utf-8"))
    tok = WhisperTokenizer.from_pretrained(BASE)
    tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    labels = []
    for t in texts:
        ids = tok(" " + t).input_ids[:448]
        if ids[0] == tok.convert_tokens_to_ids("<|startoftranscript|>"):
            ids = ids[1:]  # the model prepends the start token itself when shifting labels
        labels.append(ids)
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= 3]
    print(f"{len(keep)} usable clips of {len(texts)}", flush=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda").train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = max(20, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    def batch(ix):
        x = torch.from_numpy(np.stack([feats[i] for i in ix])).to("cuda", torch.bfloat16)
        L = max(len(labels[i]) for i in ix)
        y = torch.full((len(ix), L), -100, dtype=torch.long)
        for r, i in enumerate(ix):
            y[r, :len(labels[i])] = torch.tensor(labels[i])
        return x, y.to("cuda")

    order, pos = rng.permutation(keep), 0
    log = open(f"{out_dir}/train_log.jsonl", "a")
    t0, run_loss = time.time(), []
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if pos + args.bs > len(order):
                order, pos = rng.permutation(keep), 0
            ix = order[pos:pos + args.bs]
            pos += args.bs
            x, y = batch(ix)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=x, labels=y).loss / args.accum
            loss.backward()
            run_loss.append(loss.item() * args.accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.steps:
            el = time.time() - t0
            rec = {"step": step, "loss": round(float(np.mean(run_loss)), 4), "lr": sched.get_last_lr()[0],
                   "sec_per_step": round(el / step, 2), "eta_min": round(el / step * (args.steps - step) / 60, 1),
                   "epoch": round(step * args.bs * args.accum / len(keep), 2),
                   "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss = []
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(f"{out_dir}/step{step}")
    model.save_pretrained(f"{out_dir}/final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
