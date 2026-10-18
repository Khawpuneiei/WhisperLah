"""Noisy twins: NSC Part 3 conversation re-mixed with café ambience.

The product scenario is a phone on a café table. NSC is studio-quiet, so a
subset of Part 3 training clips is re-mixed with real café ambience at
0–15 dB SNR, with a touch of synthetic reverb, and stored again with the
same transcript. Labels are unchanged: the model learns that the words are
the same under noise.

Noise source: data/noise/*.wav|m4a|mp3, ambience recorded by the team in
real cafés with nobody at the table speaking (scripts/kopichat/README.md).
These files are training material only and never part of any test set.

    python scripts/prep_noisy.py --clips 12000
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, clean_nsc, decode_audio, iter_parquet

MNSC = "MERaLiON/Multitask-National-Speech-Corpus-v1"
LOCAL = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1"
INDEX = f"{ROOT}/data/v2_index.json"
STORE = f"{ROOT}/data/v2_feats.bin"


def load_any(path):
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                          "-f", "wav", "-"], capture_output=True, check=True).stdout
    return sf.read(io.BytesIO(out), dtype="float32")[0]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def reverb(x, rng):
    """Small-room reverb: exponentially decaying noise impulse, RT60 0.2–0.5 s."""
    rt60 = rng.uniform(0.2, 0.5)
    n = int(rt60 * 16000)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    y = np.convolve(x, ir)[:len(x)]
    return y * (rms(x) / max(rms(y), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12000)
    ap.add_argument("--shards", type=int, default=22, help="Part 3 train shards to draw clean clips from")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    noise_files = sorted(glob.glob(f"{ROOT}/data/noise/*"))
    if not noise_files:
        raise SystemExit("no files in data/noise/: record café ambience first (see scripts/kopichat/README.md)")
    noise = [load_any(p) for p in noise_files]
    print(f"{len(noise)} ambience files, {sum(len(n) for n in noise) / 16000 / 60:.1f} min", flush=True)

    index = json.load(open(INDEX, encoding="utf-8"))
    if any(c["src"] == "NOISY-PART3" for c in index["clips"]):
        raise SystemExit("noisy twins already in the store")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    flat = open(STORE, "ab")
    pos = index["n_values"]
    assert pos * 2 == os.path.getsize(STORE), "store and index disagree; refusing to append"

    from huggingface_hub import hf_hub_download
    # shards not already used for clean training, spread across the part
    used = set(index.get("shards", [])) | {f"train-{i:05d}-of-00171.parquet" for i in range(0, 171, 15)}
    cands = [f"train-{i:05d}-of-00171.parquet" for i in range(171) if f"train-{i:05d}-of-00171.parquet" not in used]
    shards = cands[::max(1, len(cands) // args.shards)][:args.shards]

    def store(waves, texts):
        nonlocal pos
        f = fe(waves, sampling_rate=16000, return_tensors="np").input_features
        for w, x, t in zip(waves, f, texts):
            T = min(3000, int(np.ceil(len(w) / 160)) + 3)
            part = np.ascontiguousarray(x[:, :T].astype(np.float16))
            flat.write(part.tobytes())
            index["clips"].append({"src": "NOISY-PART3", "off": pos, "T": T, "pad": float(x.max() - 2.0), "text": t})
            pos += part.size

    n_done = 0
    for name in tqdm(shards, desc="noisy shards"):
        if n_done >= args.clips:
            break
        path = hf_hub_download(MNSC, f"ASR-PART3-Train/{name}", repo_type="dataset", local_dir=LOCAL)
        waves, texts = [], []
        for _, b, t in iter_parquet(path, "context", "answer"):
            t = clean_nsc(t)
            if not t or n_done >= args.clips:
                continue
            clean = decode_audio(b).astype(np.float32)
            bg = noise[rng.integers(len(noise))]
            if len(bg) > len(clean):
                s0 = rng.integers(len(bg) - len(clean))
                bg = bg[s0:s0 + len(clean)]
            else:
                bg = np.tile(bg, int(np.ceil(len(clean) / len(bg))))[:len(clean)]
            snr = rng.uniform(0, 15)
            x = reverb(clean, rng) if rng.random() < 0.5 else clean
            mix = x + bg * (rms(x) / max(rms(bg), 1e-6)) * 10 ** (-snr / 20)
            mix /= max(1.0, float(np.abs(mix).max()) / 0.99)
            waves.append(mix)
            texts.append(t)
            n_done += 1
            if len(waves) == 32:
                store(waves, texts)
                waves, texts = [], []
        if waves:
            store(waves, texts)
        flat.flush()
        index["n_values"] = pos
        json.dump(index, open(INDEX, "w", encoding="utf-8"), ensure_ascii=False)
        os.remove(path)
    flat.close()
    print(f"added {n_done} noisy twins; store now {len(index['clips'])} clips, {pos * 2 / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
