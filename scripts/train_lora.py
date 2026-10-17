"""LoRA fine-tune of Whisper large-v3-turbo on Singlish conversation (MNSC Part 3).

Sized for one 8 GB laptop GPU: frozen bf16 base, fp32 LoRA weights on every
attention and MLP projection in encoder and decoder, gradient checkpointing.
The encoder is adapted on purpose: the gap on Singlish is mostly acoustic
(accent, particles, fast turn-taking), not vocabulary.

    python scripts/train_lora.py --steps 20 --name smoke        # throughput check
    python scripts/train_lora.py --steps 500 --name lora-r32    # v1
    python scripts/train_lora.py --steps 1300 --data v1,v2 --init ../runs/lora-r32/final         --lr 2e-4 --specaug --name v2                           # v2, warm start from v1
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
NEAR_PROMPT = "Near speaker only."


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
    ap.add_argument("--data", default="v1", help="comma list of feature stores: v1 (Part 3 slice), v2 (all parts)")
    ap.add_argument("--init", help="LoRA adapter to continue training from")
    ap.add_argument("--specaug", action="store_true", help="SpecAugment time + frequency masking")
    args = ap.parse_args()
    out_dir = f"{ROOT}/runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # every clip is (getter, text); getters return a [128, 3000] float array
    clips = []
    if "v1" in args.data.split(","):
        v1 = np.load(f"{ROOT}/data/train_feats.npy", mmap_mode="r")
        for i, t in enumerate(json.load(open(f"{ROOT}/data/train_text.json", encoding="utf-8"))):
            clips.append((("v1", i), t, "PART3", "room"))
    if "v2" in args.data.split(","):
        idx = json.load(open(f"{ROOT}/data/v2_index.json", encoding="utf-8"))
        v2 = np.memmap(f"{ROOT}/data/v2_feats.bin", dtype=np.float16, mode="r", shape=(idx["n_values"],))
        for c in idx["clips"]:
            clips.append((("v2", c), c["text"], c["src"], c.get("mode", "room")))

    def feature(ref):
        kind, key = ref
        if kind == "v1":
            return v1[key]
        out = np.full((128, 3000), key["pad"], dtype=np.float16)  # silence, exactly as Whisper pads
        out[:, :key["T"]] = v2[key["off"]:key["off"] + 128 * key["T"]].reshape(128, key["T"])
        return out

    texts = [t for _, t, _, _ in clips]
    from collections import Counter
    print("training mix:", dict(Counter(src for _, _, src, _ in clips)), flush=True)
    tok = WhisperTokenizer.from_pretrained(BASE)
    tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    # Explicit decoder input / label pairs. Room mode is the standard Whisper
    # target. Near-mic mode puts a Whisper prompt in front ("<|startofprev|>
    # Near speaker only."), and the prompt itself is never a training target.
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    prefix = [sot] + tok.convert_tokens_to_ids(["<|en|>", "<|transcribe|>", "<|notimestamps|>"])
    eot = tok.eos_token_id
    near_prompt = [tok.convert_tokens_to_ids("<|startofprev|>")] + tok(" " + NEAR_PROMPT, add_special_tokens=False).input_ids
    seqs = []
    for _, t, _, mode in clips:
        body = tok(" " + t, add_special_tokens=False).input_ids
        head = (near_prompt if mode == "near" else []) + prefix
        full = head + body[:447 - len(head)] + [eot]
        n_masked = len(head) - len(prefix)  # predictions of the prompt and of <|startoftranscript|>
        lab = full[1:]
        if mode == "near":
            lab = [-100] * n_masked + lab[n_masked:]
        seqs.append((full[:-1], lab))
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= 3]
    print(f"{len(keep)} usable clips of {len(texts)}", flush=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    if args.specaug:
        # time and frequency masking (HF Whisper applies it only in training)
        model.config.apply_spec_augment = True
        model.config.mask_time_prob, model.config.mask_time_length = 0.05, 10
        model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 10
    if args.init:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
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
        x = torch.from_numpy(np.stack([feature(clips[i][0]) for i in ix]).astype(np.float32)).to("cuda", torch.bfloat16)
        L = max(len(seqs[i][0]) for i in ix)
        dec = torch.full((len(ix), L), eot, dtype=torch.long)
        lab = torch.full((len(ix), L), -100, dtype=torch.long)
        for r, i in enumerate(ix):
            d, l = seqs[i]
            dec[r, :len(d)] = torch.tensor(d)
            lab[r, :len(l)] = torch.tensor(l)
        return x, dec.to("cuda"), lab.to("cuda")

    order, pos = rng.permutation(keep), 0
    log = open(f"{out_dir}/train_log.jsonl", "a")
    t0, run_loss = time.time(), []
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if pos + args.bs > len(order):
                order, pos = rng.permutation(keep), 0
            ix = order[pos:pos + args.bs]
            pos += args.bs
            x, dec, y = batch(ix)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=x, decoder_input_ids=dec, labels=y).loss / args.accum
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
