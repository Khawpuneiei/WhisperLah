"""v3 training data, streamed: download one MNSC shard, featurise it into the
store, delete the parquet, move on. Disk never holds more than a few shards.

The store format is prep_v2.py's: a flat fp16 file of unpadded log-mel
features plus v2_index.json. This script only adds to it and is safe to
re-run: shards already in the index are skipped.

    python scripts/prep_stream.py            # the v3 plan below
    python scripts/prep_stream.py --dry-run  # list shards and sizes only
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, clean_nsc, decode_audio, iter_parquet

MNSC = "MERaLiON/Multitask-National-Speech-Corpus-v1"
LOCAL = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1"
INDEX = f"{ROOT}/data/v2_index.json"
STORE = f"{ROOT}/data/v2_feats.bin"

# part: (shards in the part, how many to add). Chosen evenly across each
# part, skipping anything already in the index, so speakers are spread.
PLAN = {
    "ASR-PART3-Train": (171, 60),   # two-person conversation: the core
    "ASR-PART4-Train": (17, 13),    # all of it: our weakest conversation set
    "ASR-PART5-Train": (39, 10),
    "ASR-PART6-Train": (161, 10),
    "ASR-PART2-Train": (727, 10),   # Singapore names and places, vocabulary only
}


def shard_list(index):
    done = set(index.get("shards", []))
    v1 = {f"train-{i:05d}-of-00171.parquet" for i in range(0, 171, 15)}
    out = []
    for part, (n, k) in PLAN.items():
        names = [f"train-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        if part == "ASR-PART3-Train":
            names = [x for x in names if x not in v1]
        names = [x for x in names if x not in done]
        step = max(1, len(names) // k)
        out += [(part, x) for x in names[::step][:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = json.load(open(INDEX, encoding="utf-8"))
    index.setdefault("shards", [])
    todo = shard_list(index)
    print(f"{len(todo)} shards to add; store has {len(index['clips'])} clips", flush=True)
    if args.dry_run:
        for part, name in todo:
            print(f"  {part}/{name}")
        return

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    flat = open(STORE, "ab")
    pos = index["n_values"]
    # the index is the source of truth: a run killed mid-shard leaves bytes past
    # the index's end, which are discarded; anything shorter is a real fault
    size = os.path.getsize(STORE)
    assert size >= pos * 2, "store is shorter than the index; refusing to append"
    if size > pos * 2:
        flat.close()
        with open(STORE, "r+b") as fh:
            fh.truncate(pos * 2)
        print(f"discarded {(size - pos * 2) / 2**20:.0f} MB of partial shard past the index", flush=True)
        flat = open(STORE, "ab")

    def store(src, waves, texts):
        nonlocal pos
        f = fe(waves, sampling_rate=16000, return_tensors="np").input_features
        for w, x, t in zip(waves, f, texts):
            T = min(3000, int(np.ceil(len(w) / 160)) + 3)
            part = np.ascontiguousarray(x[:, :T].astype(np.float16))
            flat.write(part.tobytes())
            index["clips"].append({"src": src, "off": pos, "T": T, "pad": float(x.max() - 2.0), "text": t})
            pos += part.size

    def fetch(item):
        part, name = item
        return hf_hub_download(MNSC, f"{part}/{name}", repo_type="dataset", local_dir=LOCAL)

    # one download ahead of the featuriser, never more: disk is the constraint
    with ThreadPoolExecutor(1) as ex:
        futures = [ex.submit(fetch, todo[0])] if todo else []
        for k, (part, name) in enumerate(tqdm(todo, desc="shards")):
            path = futures[k].result()
            if k + 1 < len(todo):
                futures.append(ex.submit(fetch, todo[k + 1]))
            waves, texts, n_before = [], [], len(index["clips"])
            for _, b, t in iter_parquet(path, "context", "answer"):
                t = clean_nsc(t)
                if not t:
                    continue
                waves.append(decode_audio(b))
                texts.append(t)
                if len(waves) == 32:
                    store(part, waves, texts)
                    waves, texts = [], []
            if waves:
                store(part, waves, texts)
            flat.flush()
            index["shards"].append(name)
            index["n_values"] = pos
            json.dump(index, open(INDEX, "w", encoding="utf-8"), ensure_ascii=False)
            os.remove(path)
            tqdm.write(f"{part}/{name}: +{len(index['clips']) - n_before} clips, store {pos * 2 / 2**30:.1f} GB")
    flat.close()
    print(f"done: {len(index['clips'])} clips, {pos * 2 / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
