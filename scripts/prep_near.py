"""Near-mic mode training data: one speaker up front, another conversation behind.

SASRBench (and NSC's per-speaker close-talk transcripts) label only the person
wearing the microphone, even though the other person in the room is audible.
A model trained on MNSC's two-speaker transcripts transcribes both, which that
benchmark scores as insertions.

To teach "transcribe only the near speaker" on request, each single-speaker
read-speech clip from NSC Parts 1 and 2 is mixed with a random stretch of a
Part 3 conversation 6-20 dB quieter, and labelled with the near speaker's
words only. The trainer prefixes these with a Whisper prompt ("Near speaker
only."), so room mode (no prompt) is unchanged.

Appends to the same feature store as prep_v2.py, tagged src="NEAR-PARTn".
"""
import glob
import json
import os

import numpy as np
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, clean_nsc, decode_audio, iter_parquet

MNSC = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1"
INDEX = f"{ROOT}/data/v2_index.json"
rng = np.random.default_rng(0)
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

prev = json.load(open(INDEX, encoding="utf-8"))
if any(c["src"].startswith("NEAR") for c in prev["clips"]):
    raise SystemExit("near-mic clips already stored")

# background pool: 400 clips of Part 3 conversation (their words never appear in a near-mode label)
pool = []
for _, b, _ in iter_parquet(sorted(glob.glob(f"{MNSC}/ASR-PART3-Train/*.parquet"))[-1], "context", "answer"):
    pool.append(decode_audio(b).astype(np.float32))
    if len(pool) == 400:
        break


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


flat = open(f"{ROOT}/data/v2_feats.bin", "ab")
index, pos, n_new = prev["clips"], prev["n_values"], 0


def flush(src, waves, texts):
    global pos, n_new
    f = fe(waves, sampling_rate=16000, return_tensors="np").input_features
    for w, x, t in zip(waves, f, texts):
        T = min(3000, int(np.ceil(len(w) / 160)) + 3)
        feats = np.ascontiguousarray(x[:, :T].astype(np.float16))
        flat.write(feats.tobytes())
        index.append({"src": src, "off": pos, "T": T, "pad": float(x.max() - 2.0), "text": t, "mode": "near"})
        pos += feats.size
        n_new += 1


for part in ("ASR-PART1-Train", "ASR-PART2-Train"):
    src = f"NEAR-{part.split('-')[1]}"
    for path in sorted(glob.glob(f"{MNSC}/{part}/*.parquet")):
        waves, texts = [], []
        for _, b, t in tqdm(iter_parquet(path, "context", "answer"), desc=src):
            t = clean_nsc(t)
            if not t:
                continue
            fg = decode_audio(b).astype(np.float32)
            bg = pool[rng.integers(len(pool))]
            if len(bg) > len(fg):
                s0 = rng.integers(len(bg) - len(fg))
                bg = bg[s0:s0 + len(fg)]
            else:
                bg = np.pad(bg, (0, len(fg) - len(bg)))
            snr_db = rng.uniform(6, 20)
            mix = fg + bg * (rms(fg) / max(rms(bg), 1e-6)) * 10 ** (-snr_db / 20)
            mix /= max(1.0, float(np.abs(mix).max()) / 0.99)
            waves.append(mix)
            texts.append(t)
            if len(waves) == 32:
                flush(src, waves, texts)
                waves, texts = [], []
        if waves:
            flush(src, waves, texts)
flat.close()
prev.update(n_values=pos, clips=index)
json.dump(prev, open(INDEX, "w", encoding="utf-8"), ensure_ascii=False)
print(f"added {n_new} near-mic mixtures; store now {len(index)} clips, {pos * 2 / 2**30:.1f} GB", flush=True)
