"""v2 data: more Part 3 conversation, plus Parts 1/2 (short read speech, local
names) and 4/5/6 (code-switching and other conversational settings), and the
held-out test split of every part. Shards spread across each part."""
import time
from huggingface_hub import hf_hub_download

MNSC = "MERaLiON/Multitask-National-Speech-Corpus-v1"
plan = {  # part: (total shards, how many to take)
    "ASR-PART1-Train": (771, 1), "ASR-PART2-Train": (727, 1),
    "ASR-PART3-Train": (171, 8), "ASR-PART4-Train": (17, 4),
    "ASR-PART5-Train": (39, 3), "ASR-PART6-Train": (161, 4),
}
files = []
for d, (n, k) in plan.items():
    step = n // k
    # offset so the new Part 3 shards don't repeat v1's 0,15,30,...
    off = 7 if d == "ASR-PART3-Train" else step // 2
    files += [f"{d}/train-{(off + i * step) % n:05d}-of-{n:05d}.parquet" for i in range(k)]
tests = {"ASR-PART1-Test": 2, "ASR-PART2-Test": 1, "ASR-PART4-Test": 2, "ASR-PART5-Test": 2, "ASR-PART6-Test": 2}
files += [f"{d}/train-{i:05d}-of-{n:05d}.parquet" for d, n in tests.items() for i in range(n)]
t0 = time.time()
for f in files:
    hf_hub_download(MNSC, f, repo_type="dataset", local_dir="data/Multitask-National-Speech-Corpus-v1")
    print(f"{time.time()-t0:6.0f}s {f}", flush=True)
print("DOWNLOAD_V2_DONE", flush=True)
