"""A slice of LibriSpeech train-clean-100 (CC-BY-4.0) into the training store.

Insurance against forgetting standard English: the v2 adapter lost 0.8 pt on
LibriSpeech "other" after Singlish-only training. A few thousand read-English
utterances in the mix is enough to hold the line; this is not an attempt to
win LibriSpeech.

    python scripts/prep_libri.py --shards 5      # ~10k utterances, ~2.3 GB streamed
"""
import argparse
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, decode_audio, iter_parquet

R = "openslr/librispeech_asr"
LOCAL = f"{ROOT}/data/librispeech"
INDEX = f"{ROOT}/data/v2_index.json"
STORE = f"{ROOT}/data/v2_feats.bin"
N_SHARDS = 14  # all/train.clean.100/0000..0013


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=5)
    args = ap.parse_args()

    index = json.load(open(INDEX, encoding="utf-8"))
    index.setdefault("shards", [])
    names = [f"all/train.clean.100/{i:04d}.parquet" for i in range(N_SHARDS)]
    names = [n for n in names if f"libri:{n}" not in index["shards"]]
    step = max(1, len(names) // args.shards)
    todo = names[::step][:args.shards]
    print(f"{len(todo)} LibriSpeech shards to add; store has {len(index['clips'])} clips", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    flat = open(STORE, "ab")
    pos = index["n_values"]
    size = os.path.getsize(STORE)
    assert size >= pos * 2, "store is shorter than the index; refusing to append"
    if size > pos * 2:
        flat.close()
        with open(STORE, "r+b") as fh:
            fh.truncate(pos * 2)
        print(f"discarded {(size - pos * 2) / 2**20:.0f} MB of partial shard past the index", flush=True)
        flat = open(STORE, "ab")

    def store(waves, texts):
        nonlocal pos
        f = fe(waves, sampling_rate=16000, return_tensors="np").input_features
        for w, x, t in zip(waves, f, texts):
            T = min(3000, int(np.ceil(len(w) / 160)) + 3)
            part = np.ascontiguousarray(x[:, :T].astype(np.float16))
            flat.write(part.tobytes())
            index["clips"].append({"src": "LIBRI", "off": pos, "T": T, "pad": float(x.max() - 2.0), "text": t})
            pos += part.size

    for name in tqdm(todo, desc="librispeech shards"):
        path = hf_hub_download(R, name, repo_type="dataset", local_dir=LOCAL)
        waves, texts, n0 = [], [], len(index["clips"])
        for _, b, t in iter_parquet(path, "audio", "text"):
            t = " ".join((t or "").lower().split())
            w = decode_audio(b)
            if not t or len(w) > 30 * 16000:
                continue
            waves.append(w)
            texts.append(t)
            if len(waves) == 32:
                store(waves, texts)
                waves, texts = [], []
        if waves:
            store(waves, texts)
        flat.flush()
        index["shards"].append(f"libri:{name}")
        index["n_values"] = pos
        json.dump(index, open(INDEX, "w", encoding="utf-8"), ensure_ascii=False)
        os.remove(path)
        tqdm.write(f"{name}: +{len(index['clips']) - n0} clips, store {pos * 2 / 2**30:.1f} GB")
    flat.close()
    print(f"done: {len(index['clips'])} clips, {pos * 2 / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
