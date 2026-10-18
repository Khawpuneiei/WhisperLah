# One appender at a time: NSC streaming -> AMI (10 shards per mic) -> LibriSpeech slice.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
until grep -q "^done:" ../results/log_prep_stream.txt; do sleep 30; done
$PY prep_ami.py --mics ihm sdm --shards 10 > ../results/log_prep_ami.txt 2>&1 && echo AMI_PREP_OK || { echo AMI_PREP_FAILED; tail -3 ../results/log_prep_ami.txt; }
$PY prep_libri.py --shards 5 > ../results/log_prep_libri.txt 2>&1 && echo LIBRI_PREP_OK || { echo LIBRI_PREP_FAILED; tail -3 ../results/log_prep_libri.txt; }
$PY - <<'PY'
import json, collections
i = json.load(open("../data/v2_index.json", encoding="utf-8"))
print("store:", len(i["clips"]), "clips;", dict(collections.Counter(c["src"] for c in i["clips"])))
PY
echo DATA_V3_DONE
