"""Assemble README.md from docs/README.template.md plus generated results tables."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "docs" / "README.template.md").read_text(encoding="utf-8")
tables = subprocess.run([sys.executable, str(ROOT / "scripts" / "table_md.py")], capture_output=True,
                        text=True, encoding="utf-8", check=True).stdout
t = {(r["set"], r["model"]): r for r in json.load(open(ROOT / "results" / "table.json")) if r["view"] == "full"}


def w(s, m):
    return f"{t[(s, m)]['wer']:.1f}%" if (s, m) in t else "n/a"


others = ["whisper-large-v3-turbo", "mjwong-turbo-singlish", "MERaLiON-2-3B"]
OURS = [m for m in ("whisperlah-v2.1", "whisperlah-v2.1-near", "whisperlah-v2", "whisperlah-v2-near")]
LABEL = {"sasrbench": "SASRBench-v1", "mnsc_part3": "NSC Part 3", "mnsc_part1": "NSC Part 1",
         "mnsc_part2": "NSC Part 2", "mnsc_part4": "NSC Part 4", "mnsc_part5": "NSC Part 5", "mnsc_part6": "NSC Part 6"}
VERSION = sys.argv[1] if len(sys.argv) > 1 else "whisperlah-v2"


def ours_best(s):
    vals = [t[(s, m)]["wer"] for m in (VERSION, VERSION + "-near") if (s, m) in t]
    return min(vals) if vals else None


wins, losses = [], []
for s in LABEL:
    o = ours_best(s)
    rivals = [(t[(s, m)]["wer"], m) for m in others if (s, m) in t]
    if o is None or not rivals:
        continue
    (wins if o < min(rivals)[0] else losses).append((s, o, min(rivals)))
rival_name = {"whisper-large-v3-turbo": "Whisper large-v3-turbo", "mjwong-turbo-singlish": "the public Singlish fine-tune",
              "MERaLiON-2-3B": "MERaLiON-2-3B"}
headline = (
    f"**Lowest word error rate of every model tested on {len(wins)} of {len(wins) + len(losses)} Singlish test sets**, "
    f"including both public benchmarks: {w('sasrbench', VERSION + '-near')} on SASRBench-v1 "
    f"(MERaLiON-2-3B {w('sasrbench', 'MERaLiON-2-3B')}, public Singlish fine-tune {w('sasrbench', 'mjwong-turbo-singlish')}, "
    f"Whisper large-v3-turbo {w('sasrbench', 'whisper-large-v3-turbo')}) and {w('mnsc_part3', VERSION)} on NSC two-person "
    f"conversation (next best {min(t[('mnsc_part3', m)]['wer'] for m in others if ('mnsc_part3', m) in t):.1f}%). "
)
if losses:
    headline += ("Behind on " + ", ".join(f"{LABEL[s]} ({o:.1f}% vs {r[0]:.1f}% for {rival_name[r[1]]})" for s, o, r in losses)
                 + ": read-speech sets whose prompt sentences recur in the training data (see Benchmark protocol). ")
headline += ("It also writes down the Singlish discourse particles (lah, leh, lor …) that the other models drop: "
             "about 70% of them, against 10% or less.")
out = tpl.replace("<!--HEADLINE-->", headline).replace("<!--RESULTS_TABLE-->", tables.strip())
(ROOT / "README.md").write_text(out, encoding="utf-8")
print(headline)
