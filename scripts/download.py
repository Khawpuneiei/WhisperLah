"""Fetch the benchmark sets, a speaker-spread slice of MNSC Part 3 train, and the two reference models."""
import time
from huggingface_hub import hf_hub_download, snapshot_download

MNSC = "MERaLiON/Multitask-National-Speech-Corpus-v1"
files = [("mjwong/SASRBench-v1", f"data/test-0000{i}-of-00004.parquet") for i in range(4)]
files += [(MNSC, f"ASR-PART3-Test/train-0000{i}-of-00002.parquet") for i in range(2)]
# 12 of 171 shards, spread across the corpus so the slice covers many speakers
files += [(MNSC, f"ASR-PART3-Train/train-{i:05d}-of-00171.parquet") for i in range(0, 171, 15)]

t0 = time.time()
for repo, f in files:
    p = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"data/{repo.split('/')[1]}")
    print(f"{time.time()-t0:7.0f}s  {p}", flush=True)
for m in ["openai/whisper-large-v3-turbo", "mjwong/whisper-large-v3-turbo-singlish"]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken"])
    print(f"{time.time()-t0:7.0f}s  model {m}", flush=True)
print("DOWNLOAD_DONE", flush=True)
