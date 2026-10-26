"""Tune community-1's clustering (with a fine-tuned segmentation) on dev.

    .venv-diar/Scripts/python.exe scripts/diar/tune_clustering.py --segmentation runs/diar/seg-ft/epoch=14.ckpt --device cuda

Segmentation and embeddings are computed once per file (cached under
data/diar/cache/), then only the clustering and reconstruction steps re-run
for each setting, so the whole sweep takes minutes instead of hours.

community-1 clusters with VBx (AHC initialisation, then variational Bayes):
`threshold` is the AHC distance cut, `Fa`/`Fb` are VBx's acoustic scaling and
speaker-regularisation factors. It has no min_cluster_size. The sweep covers
threshold x Fa x Fb, then min_duration_off on the best of those, and also
scores the best setting with the oracle number of speakers (num_speakers taken
from the reference) to show how much confusion is just miscounting.

Writes results/diar_tune_<split>.jsonl (one row per setting) and the chosen
parameters to data/diar/pipeline_best.json, which eval_der.py --params reads.
"""
import argparse
import copy
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from pyannote.audio import Model, Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate

from eval_der import ROOT, hf_token, load_rttm

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
FAS = [0.03, 0.07, 0.15, 0.3, 0.5]  # 0.5 is the top of pyannote's search range
FBS = [0.1, 0.3, 0.8, 2.0, 5.0]
MIN_DURATION_OFFS = [0.0, 0.1, 0.2, 0.4]


def load_pipeline(segmentation, device):
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token())
    if segmentation:
        pipe._segmentation.model = Model.from_pretrained(segmentation)
    pipe.to(torch.device(device))
    return pipe


def extract(pipe, wav_path, uri):
    """Run the pipeline once and keep what clustering needs."""
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float32) / 32768
    got = {}

    def hook(step, artefact, file=None, completed=None, total=None):
        if completed is None and step in ("segmentation", "speaker_counting", "embeddings"):
            got[step] = copy.deepcopy(artefact)  # apply() later caps `count` in place

    pipe({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr, "uri": uri}, hook=hook)
    return got


def diarize_from_cache(pipe, c, num_speakers=None):
    """The post-embedding half of SpeakerDiarization.apply (pyannote.audio 4.0)."""
    segmentations, embeddings = c["segmentation"], c["embeddings"]
    count = c["speaker_counting"]
    count = type(count)(count.data.copy(), count.sliding_window)
    min_speakers = num_speakers or 1
    max_speakers = num_speakers or np.inf
    hard_clusters, _, _ = pipe.clustering(
        embeddings=embeddings, segmentations=segmentations, num_clusters=num_speakers,
        min_clusters=min_speakers, max_clusters=max_speakers)
    count.data = np.minimum(count.data, max_speakers).astype(np.int8)
    hard_clusters[np.sum(segmentations.data, axis=1) == 0] = -2  # powerset: no binarize step
    discrete = pipe.reconstruct(segmentations, hard_clusters, count)
    return pipe.to_annotation(discrete, min_duration_on=0.0,
                              min_duration_off=pipe.segmentation.min_duration_off)


def score(pipe, cache, refs, params, oracle=False):
    pipe.instantiate(params)
    der = DiarizationErrorRate(collar=0.0)
    n_spk_err = n_oracle_fallback = 0
    for name, c in cache.items():
        k = len(refs[name].labels()) if oracle else None
        try:
            hyp = diarize_from_cache(pipe, c, num_speakers=k)
        except ValueError:  # fewer embeddings than reference speakers: KMeans cannot force k
            hyp = diarize_from_cache(pipe, c)
            n_oracle_fallback += 1
        hyp.uri = name
        der(refs[name], hyp)
        n_spk_err += len(hyp.labels()) != len(refs[name].labels())
    comp = der.accumulated_
    total = comp.get("total", 0.0) or 1.0
    return {"der": round(100 * abs(der), 2),
            "missed": round(100 * comp.get("missed detection", 0) / total, 2),
            "false_alarm": round(100 * comp.get("false alarm", 0) / total, 2),
            "confusion": round(100 * comp.get("confusion", 0) / total, 2),
            "speaker_count_wrong": n_spk_err, "oracle_speakers": oracle, "oracle_fallback": n_oracle_fallback, "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--segmentation", default="runs/diar/seg-ft/epoch=14.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="data/diar/pipeline_best.json")
    args = ap.parse_args()

    d = ROOT / "data" / "diar" / args.split
    refs = load_rttm(d / f"{args.split}.rttm")
    names = (d / f"{args.split}.lst").read_text().split()[: args.limit]
    pipe = load_pipeline(ROOT / args.segmentation, args.device)
    default = pipe.parameters(instantiated=True)

    tag = Path(args.segmentation).stem.replace("=", "")
    cache_path = ROOT / "data" / "diar" / "cache" / f"{args.split}__{tag}.pkl"
    if cache_path.exists():
        cache = pickle.loads(cache_path.read_bytes())
    else:
        t0 = time.time()
        cache = {}
        for k, name in enumerate(names):
            cache[name] = extract(pipe, d / "wav" / f"{name}.wav", name)
            if k % 50 == 0:
                print(f"extract {k}/{len(names)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
        print(f"extracted {len(cache)} files in {time.time() - t0:.0f}s", flush=True)
    cache = {n: cache[n] for n in names}

    log = (ROOT / "results" / f"diar_tune_{args.split}.jsonl").open("w")

    def run(params, oracle=False):
        r = score(pipe, cache, refs, params, oracle)
        log.write(json.dumps(r) + "\n")
        log.flush()
        return r

    base = run(default)
    print("default", json.dumps(base), flush=True)

    rows = []
    for th, fa, fb in itertools.product(THRESHOLDS, FAS, FBS):
        params = {"segmentation": dict(default["segmentation"]),
                  "clustering": {"threshold": th, "Fa": fa, "Fb": fb}}
        rows.append(run(params))
    best = min(rows, key=lambda r: r["der"])
    print("best clustering", json.dumps(best), flush=True)

    for mdo in MIN_DURATION_OFFS:
        params = {"segmentation": {"min_duration_off": mdo}, "clustering": best["params"]["clustering"]}
        r = run(params)
        if r["der"] < best["der"]:
            best = r
    print("best", json.dumps(best), flush=True)

    oracle = run(best["params"], oracle=True)
    oracle_default = run(default, oracle=True)
    print("oracle speakers (best)", json.dumps(oracle), flush=True)
    print("oracle speakers (default)", json.dumps(oracle_default), flush=True)
    log.close()

    out = ROOT / args.out
    out.write_text(json.dumps({
        "pipeline": "pyannote/speaker-diarization-community-1",
        "segmentation": args.segmentation, "params": best["params"],
        "tuned_on": args.split, "files": len(names),
        "dev": {k: best[k] for k in ("der", "missed", "false_alarm", "confusion")},
        "dev_default": {k: base[k] for k in ("der", "missed", "false_alarm", "confusion")},
        "dev_oracle_speakers": {k: oracle[k] for k in ("der", "missed", "false_alarm", "confusion")},
    }, indent=1))
    print(f"wrote {out}\nTUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
