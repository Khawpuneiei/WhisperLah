"""v2 training data: every downloaded MNSC train shard not already in v1.

Features are stored unpadded (a short read-speech clip is ~500 frames, not
3000) in one flat fp16 file with an index, then padded back at batch time.
Whisper pads with silence, which log-mel normalisation turns into a single
per-clip constant (max - 2.0), so padding can be rebuilt exactly.

Also writes, for every test set, the clips whose reference shares >=30% of its
8-grams with any training transcript (v1 + v2). Those clips are excluded from
every model's score, so no model gets credit for text it trained on.
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, TEST_SETS, Normalizer, clean_nsc, decode_audio, iter_parquet

MNSC = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1"
SOURCES = ["ASR-PART1-Train", "ASR-PART2-Train", "ASR-PART3-Train", "ASR-PART4-Train",
           "ASR-PART5-Train", "ASR-PART6-Train"]

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
shards = [(src, p) for src in SOURCES for p in sorted(glob.glob(f"{MNSC}/{src}/*.parquet"))]
N = sum(pq.ParquetFile(p).metadata.num_rows for _, p in shards)
print(f"{len(shards)} shards, {N} clips", flush=True)

# appended clip by clip: exact size on disk, no preallocation
flat = open(f"{ROOT}/data/v2_feats.bin", "wb")
index, pos, secs = [], 0, {}


def store(src, waves, texts):
    global pos
    f = fe(waves, sampling_rate=16000, return_tensors="np").input_features  # [b,128,3000]
    for w, x, t in zip(waves, f, texts):
        T = min(3000, int(np.ceil(len(w) / 160)) + 3)  # +2 is exact (STFT window leakage); +3 for margin
        part = np.ascontiguousarray(x[:, :T].astype(np.float16))
        n = part.size
        flat.write(part.tobytes())
        index.append({"src": src, "off": pos, "T": T, "pad": float(x.max() - 2.0), "text": t})
        pos += n
        secs[src] = secs.get(src, 0.0) + len(w) / 16000


for src, path in tqdm(shards, desc="shards"):
    batch_w, batch_t = [], []
    for _, b, t in iter_parquet(path, "context", "answer"):
        t = clean_nsc(t)
        if not t:
            continue
        batch_w.append(decode_audio(b))
        batch_t.append(t)
        if len(batch_w) == 32:
            store(src, batch_w, batch_t)
            batch_w, batch_t = [], []
    if batch_w:
        store(src, batch_w, batch_t)

flat.close()
json.dump({"n_values": pos, "clips": index}, open(f"{ROOT}/data/v2_index.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print({k: f"{v / 3600:.1f} h" for k, v in secs.items()}, flush=True)
print(f"stored {len(index)} clips, {pos * 2 / 2**30:.1f} GB", flush=True)

# --- contamination check: all training text vs every test set --------------
norm = Normalizer()


def grams(t, n=8):
    w = norm(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


train_text = json.load(open(f"{ROOT}/data/train_text.json", encoding="utf-8")) + [c["text"] for c in index]
train_grams = set()
for t in tqdm(train_text, desc="train 8-grams"):
    train_grams |= grams(t)
report = {}
for name in TEST_SETS:
    pattern, ac, tc = TEST_SETS[name]
    if not glob.glob(pattern):
        continue
    seen, total = [], 0
    for i, _, ref in iter_parquet(pattern, ac, tc):
        total += 1
        g = grams(ref)
        if g and len(g & train_grams) / len(g) >= 0.3:
            seen.append(i)
    json.dump(seen, open(f"{ROOT}/results/{name}_overlap_with_train.json", "w"))
    report[name] = f"{len(seen)}/{total}"
print("test clips overlapping training text:", report, flush=True)
