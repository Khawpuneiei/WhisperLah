"""AMI meeting corpus (CC-BY-4.0) into the training store, streamed.

Two microphone conditions, both useful:
  ihm  individual headset mic: what the Open ASR Leaderboard's AMI test uses
  sdm  single distant mic on the table: the closest public match to a phone
       lying on a café table

Utterances are stored as they come (AMI is already segmented per speaker
turn, mostly 1–15 s). Text is uppercased without punctuation in AMI; it is
lowercased here so the decoder sees ordinary casing like the rest of the mix.

    python scripts/prep_ami.py --mics ihm sdm --shards 20
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from common import ROOT, decode_audio, iter_parquet

R, REV = "edinburghcstr/ami", "refs/convert/parquet"
LOCAL = f"{ROOT}/data/ami"
INDEX = f"{ROOT}/data/v2_index.json"
STORE = f"{ROOT}/data/v2_feats.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mics", nargs="+", default=["ihm", "sdm"])
    ap.add_argument("--shards", type=int, default=20, help="train shards per mic condition (42 ihm, 27 sdm exist)")
    ap.add_argument("--min-sec", type=float, default=0.5)
    args = ap.parse_args()

    index = json.load(open(INDEX, encoding="utf-8"))
    index.setdefault("shards", [])
    files = [f.path for f in HfApi().list_repo_tree(R, repo_type="dataset", revision=REV, recursive=True) if hasattr(f, "size")]
    todo = []
    for mic in args.mics:
        tr = sorted(p for p in files if p.startswith(f"{mic}/train/"))
        tr = [p for p in tr if f"ami:{p}" not in index["shards"]]
        step = max(1, len(tr) // args.shards)
        todo += [(f"AMI-{mic.upper()}", p) for p in tr[::step][:args.shards]]
    print(f"{len(todo)} AMI shards to add; store has {len(index['clips'])} clips", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    flat = open(STORE, "ab")
    pos = index["n_values"]
    assert pos * 2 == os.path.getsize(STORE), "store and index disagree; refusing to append"

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
        return hf_hub_download(R, item[1], repo_type="dataset", revision=REV, local_dir=LOCAL)

    with ThreadPoolExecutor(1) as ex:
        futures = [ex.submit(fetch, todo[0])] if todo else []
        for k, (src, name) in enumerate(tqdm(todo, desc="ami shards")):
            path = futures[k].result()
            if k + 1 < len(todo):
                futures.append(ex.submit(fetch, todo[k + 1]))
            waves, texts, n0 = [], [], len(index["clips"])
            for _, b, t in iter_parquet(path, "audio", "text"):
                t = " ".join((t or "").lower().split())
                if not t:
                    continue
                w = decode_audio(b)
                if len(w) < args.min_sec * 16000 or len(w) > 30 * 16000:
                    continue
                waves.append(w)
                texts.append(t)
                if len(waves) == 32:
                    store(src, waves, texts)
                    waves, texts = [], []
            if waves:
                store(src, waves, texts)
            flat.flush()
            index["shards"].append(f"ami:{name}")
            index["n_values"] = pos
            json.dump(index, open(INDEX, "w", encoding="utf-8"), ensure_ascii=False)
            os.remove(path)
            tqdm.write(f"{name}: +{len(index['clips']) - n0} clips, store {pos * 2 / 2**30:.1f} GB")
    flat.close()
    print(f"done: {len(index['clips'])} clips, {pos * 2 / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
