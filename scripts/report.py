"""Build the results table from results/<set>__<model>.jsonl.

Every model is scored on the same normalised text and the same clips. A clip
whose reference shares >=30% of its 8-grams with any training transcript
(results/<set>_overlap_with_train.json, from prep_v2.py --check) is excluded
from every model's score, so no model gets credit for text we trained on.
If models were run on different clip sets, a "common" view on the shared clips
is added.

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
SKIP = {"smoke", "speedtest"}
SETS = ("sasrbench", "mnsc_part3", "mnsc_part1", "mnsc_part2", "mnsc_part4", "mnsc_part5", "mnsc_part6")


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
    table = []
    for s in SETS:
        ov = f"{ROOT}/results/{s}_overlap_with_train.json"
        excl = set(json.load(open(ov))) if os.path.exists(ov) else set()
        runs = {}
        for p in sorted(glob.glob(f"{ROOT}/results/{s}__*.jsonl")):
            name = os.path.basename(p)[len(s) + 2:-6]
            if name in SKIP:
                continue
            rows = [json.loads(line) for line in open(p, encoding="utf-8")]
            ok = {r["idx"]: r for r in rows if "error" not in r}
            if ok:
                runs[name] = ok
        if not runs:
            continue
        common = set.intersection(*(set(v) for v in runs.values())) - excl
        uneven = any(set(rows) - excl != common for rows in runs.values())
        for name, rows in runs.items():
            for view, ids in (("full", set(rows) - excl), ("common", common)):
                if view == "common" and not uneven:
                    continue
                pairs = [(norm(rows[i]["ref"]), norm(rows[i]["hyp"])) for i in sorted(ids)]
                pairs = [(r, h) for r, h in pairs if r]
                rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
                pr, ptot = particle_recall(pairs)
                table.append({"set": s, "view": view, "model": name, "clips": rep["n"], "excluded": len(excl),
                              "wer": rep["wer"], "sub": rep["sub"], "del": rep["del"], "ins": rep["ins"],
                              "particle_recall": pr, "particles_in_ref": ptot})
    json.dump(table, open(f"{ROOT}/results/table.json", "w"), indent=1)
    print(f"{'set':11} {'view':7} {'model':30} {'clips':>5} {'WER%':>6} {'sub':>5} {'del':>5} {'ins':>5} {'particle%':>9}")
    for t in table:
        print(f"{t['set']:11} {t['view']:7} {t['model']:30} {t['clips']:>5} {t['wer']:>6} {t['sub']:>5} "
              f"{t['del']:>5} {t['ins']:>5} {str(t['particle_recall']):>9}")
    print("\nClips whose reference text overlaps any training transcript are excluded from every model's score.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
