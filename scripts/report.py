"""Build the results table from results/<set>__<model>.jsonl.

Every model is scored on the same normalised text. Two views per test set:
  - full set   : every clip, local models only (the API run is a subsample)
  - common     : the clips every model was run on, so the API is comparable
SASRBench clips whose text overlaps our training slice are excluded from every
model's score, so no model gets credit for audio we trained on.

Also reports Singlish particle recall: of the discourse particles in the
reference (lah, leh, lor ...), what share appear in the hypothesis.
"""
import collections
import glob
import json
import os
import sys

from common import ROOT, Normalizer, wer_report

PARTICLES = {"lah", "leh", "lor", "meh", "sia", "hor", "mah", "liao", "hah", "huh", "wah", "alamak"}
SKIP = {"smoke"}


def particle_recall(pairs):
    hit = tot = 0
    for r, h in pairs:
        rc = collections.Counter(w for w in r.split() if w in PARTICLES)
        hc = collections.Counter(w for w in h.split() if w in PARTICLES)
        tot += sum(rc.values())
        hit += sum(min(c, hc[w]) for w, c in rc.items())
    return round(100 * hit / tot, 1) if tot else None, tot


def main():
    norm = Normalizer()
    seen = set(json.load(open(f"{ROOT}/results/sasrbench_overlap_with_train.json"))) \
        if os.path.exists(f"{ROOT}/results/sasrbench_overlap_with_train.json") else set()
    table = []
    for s in ("mnsc_part3", "sasrbench"):
        runs = {}
        for p in sorted(glob.glob(f"{ROOT}/results/{s}__*.jsonl")):
            name = os.path.basename(p)[len(s) + 2:-6]
            if name in SKIP:
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            ok = {r["idx"]: r for r in rows if "error" not in r}
            if ok:
                runs[name] = ok
        if not runs:
            continue
        excl = seen if s == "sasrbench" else set()
        common = set.intersection(*(set(v) for v in runs.values())) - excl
        uneven = any(set(rows) - excl != common for rows in runs.values())
        for name, rows in runs.items():
            full_ids = set(rows) - excl
            for view, ids in (("full", full_ids), ("common", common)):
                if view == "common" and not uneven:
                    continue
                pairs = [(norm(rows[i]["ref"]), norm(rows[i]["hyp"])) for i in sorted(ids)]
                pairs = [(r, h) for r, h in pairs if r]
                rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
                pr, ptot = particle_recall(pairs)
                table.append({"set": s, "view": view, "model": name, "clips": rep["n"], "wer": rep["wer"],
                              "sub": rep["sub"], "del": rep["del"], "ins": rep["ins"],
                              "particle_recall": pr, "particles_in_ref": ptot})
    json.dump(table, open(f"{ROOT}/results/table.json", "w"), indent=1)
    print(f"{'set':11} {'view':7} {'model':42} {'clips':>5} {'WER%':>6} {'sub':>5} {'del':>5} {'ins':>5} {'particle%':>9}")
    for t in table:
        print(f"{t['set']:11} {t['view']:7} {t['model']:42} {t['clips']:>5} {t['wer']:>6} {t['sub']:>5} {t['del']:>5} {t['ins']:>5} {str(t['particle_recall']):>9}")
    if seen:
        print(f"\nSASRBench: {len(seen)} clips overlapping our training slice excluded from every model.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
