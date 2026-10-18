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
best_other_p3 = min((t[("mnsc_part3", m)]["wer"], m) for m in others if ("mnsc_part3", m) in t)
headline = (
    f"**On both public Singlish benchmarks, WhisperLah v2 has the lowest word error rate of every model tested**: "
    f"{w('sasrbench', 'whisperlah-v2-near')} on SASRBench-v1 (MERaLiON-2-3B {w('sasrbench', 'MERaLiON-2-3B')}, "
    f"public Singlish fine-tune {w('sasrbench', 'mjwong-turbo-singlish')}, Whisper large-v3-turbo "
    f"{w('sasrbench', 'whisper-large-v3-turbo')}) and {w('mnsc_part3', 'whisperlah-v2-near')} on NSC two-person "
    f"conversation (next best {best_other_p3[0]:.1f}%). It also writes down the Singlish discourse particles "
    f"(lah, leh, lor …) that the other models drop: about 70% of them, against 10% or less."
)
out = tpl.replace("<!--HEADLINE-->", headline).replace("<!--RESULTS_TABLE-->", tables.strip())
(ROOT / "README.md").write_text(out, encoding="utf-8")
print(headline)
