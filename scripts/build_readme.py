"""Assemble README.md from docs/README.template.md plus generated results tables."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from table_md import MODELS  # noqa: E402
tpl = (ROOT / "docs" / "README.template.md").read_text(encoding="utf-8")
tables = subprocess.run([sys.executable, str(ROOT / "scripts" / "table_md.py")], capture_output=True,
                        text=True, encoding="utf-8", check=True).stdout
t = {(r["set"], r["model"]): r for r in json.load(open(ROOT / "results" / "table.json")) if r["view"] == "full"}


def w(s, m):
    return f"{t[(s, m)]['wer']:.1f}%" if (s, m) in t else "n/a"


# every outside model in the results table (table_md.py), each at its headline setting
others = [m for m, _ in MODELS if not m.startswith("whisperlah") and any(k[1] == m for k in t)]
OURS = [m for m in ("whisperlah-v3", "whisperlah-v3-near", "whisperlah-v2.1", "whisperlah-v2.1-near", "whisperlah-v2", "whisperlah-v2-near")]
LABEL = {"sasrbench": "SASRBench-v1", "mnsc_part3": "NSC Part 3", "mnsc_part1": "NSC Part 1",
         "mnsc_part2": "NSC Part 2", "mnsc_part4": "NSC Part 4", "mnsc_part5": "NSC Part 5", "mnsc_part6": "NSC Part 6"}
VERSION = sys.argv[1] if len(sys.argv) > 1 else "whisperlah-v3"


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
# short names for the README prose
rival_name = {m: lab.split(" (")[0] for m, lab in MODELS}
rival_name.update({"mjwong-turbo-singlish": "the public Singlish fine-tune", "gpt-transcribe": "OpenAI's GPT Transcribe"})
def best_other(s):
    return min((t[(s, m)]["wer"], m) for m in others if (s, m) in t)


headline = (
    f"**Lowest word error rate of every model tested on {len(wins)} of {len(wins) + len(losses)} Singlish test sets**, "
    f"against {len(others)} other models and API settings, including both public benchmarks: "
    f"{w('sasrbench', VERSION + '-near')} on SASRBench-v1 (next best {best_other('sasrbench')[0]:.1f}%, "
    f"{rival_name[best_other('sasrbench')[1]]}) and {w('mnsc_part3', VERSION)} on NSC two-person conversation "
    f"(next best {best_other('mnsc_part3')[0]:.1f}%, {rival_name[best_other('mnsc_part3')[1]]}). "
)
if losses:
    parts = [f"{LABEL[s]} ({o:.1f}% vs {r[0]:.1f}% for {rival_name[r[1]]})" for s, o, r in losses]
    headline += ("Behind on " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0])
                 + ": read-speech sets whose prompt sentences recur in the training data (see Benchmark protocol). ")
pr = {m: t[("mnsc_part3", m)]["particle_recall"] for m in [VERSION] + others if ("mnsc_part3", m) in t}
top = max(others, key=lambda m: pr.get(m) or 0)
headline += (f"It also writes down the Singlish discourse particles (lah, leh, lor …) that other models drop: "
             f"{pr[VERSION]:.0f}% of them on NSC Part 3, against {pr[top]:.0f}% for the next best "
             f"({rival_name[top]}) and 10% or less for {sum(v <= 10 for m, v in pr.items() if m != VERSION)} of the "
             f"{len(pr) - 1} others.")
out = tpl.replace("<!--HEADLINE-->", headline).replace("<!--RESULTS_TABLE-->", tables.strip())
(ROOT / "README.md").write_text(out, encoding="utf-8")
print(headline)
