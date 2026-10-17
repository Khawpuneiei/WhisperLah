"""Turn the MNSC Part 3 training slice into a log-mel memmap + cleaned targets.

Precomputing on the GPU once means the training loop only reads arrays, which
matters on Windows where DataLoader workers are spawned processes that would
each re-open the parquet files.

Also runs the contamination check against SASRBench: both benchmarks are cut
from NSC Part 3, so some of their audio may be in this training slice. We
measure 8-gram overlap per SASRBench clip and write the list of clips that
look seen, so the SASRBench result for our model can exclude them.
"""
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, TRAIN_GLOB, Normalizer, clean_nsc, decode_audio, iter_parquet, load_set

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
# streamed, never materialised: the slice is ~6 GB of WAV bytes
N = sum(pq.ParquetFile(p).metadata.num_rows for p in sorted(glob.glob(TRAIN_GLOB)))
print(f"{N} training clips", flush=True)

feats = np.lib.format.open_memmap(f"{ROOT}/data/train_feats.npy", mode="w+", dtype=np.float16,
                                  shape=(N, fe.feature_size, fe.nb_max_frames))
texts, secs, k, chunk = [], 0.0, 0, []

def flush(chunk, k):
    waves = [decode_audio(b) for _, b, _ in chunk]
    f = fe(waves, sampling_rate=16000, return_tensors="np", device="cuda")
    feats[k:k + len(chunk)] = f.input_features.astype(np.float16)
    return sum(len(w) for w in waves) / 16000

for row in tqdm(iter_parquet(TRAIN_GLOB, "context", "answer"), total=N, desc="features"):
    chunk.append(row)
    texts.append(clean_nsc(row[2]))
    if len(chunk) == 32:
        secs += flush(chunk, k)
        k += len(chunk)
        chunk = []
if chunk:
    secs += flush(chunk, k)
feats.flush()
with open(f"{ROOT}/data/train_text.json", "w", encoding="utf-8") as fh:
    json.dump(texts, fh, ensure_ascii=False)
print(f"train audio: {secs / 3600:.1f} h", flush=True)

# --- contamination check vs SASRBench -------------------------------------
norm = Normalizer()
def grams(t, n=8):
    w = norm(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

train_grams = set()
for t in texts:
    train_grams |= grams(t)
seen = []
for i, _, ref in load_set("sasrbench"):
    g = grams(ref)
    if g and len(g & train_grams) / len(g) >= 0.3:
        seen.append(i)
json.dump(seen, open(f"{ROOT}/results/sasrbench_overlap_with_train.json", "w"))
print(f"SASRBench clips overlapping the training slice: {len(seen)}", flush=True)
