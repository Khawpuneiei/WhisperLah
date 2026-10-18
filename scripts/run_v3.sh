# v3: waits for the English gate and the streamed data, trains ~1 epoch over
# the enlarged store from v2, scores every checkpoint on a fixed 600-clip
# probe (300 NSC Part 3 + 300 SASRBench), and names the best one.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
BASE=openai/whisper-large-v3-turbo
STEPS=${STEPS:-7000}
# English sources (AMI meetings, LibriSpeech) are in the store to stop the
# English regression seen at the gate; drawn at 0.6x their share so the mix
# stays mostly Singlish. NSC Part 2 (read sentences, 37k clips because its
# shards are ten times denser than Part 3's) is held to 0.1x: it is there for
# Singapore names only, and its prompt sentences recur in the test split.
EXTRA=${EXTRA:---weights AMI-IHM=0.6,AMI-SDM=0.6,LIBRI=0.6,ASR-PART2-Train=0.1}

until grep -q ENGLISH_GATE_DONE ../results/log_english_gate.txt; do sleep 30; done
until grep -q "^done:" ../results/log_prep_stream.txt; do sleep 30; done
until grep -q PRE_V3_DONE ../results/log_pre_v3.txt 2>/dev/null; do sleep 30; done
until grep -q DATA_V3_DONE ../results/log_data_v3.txt 2>/dev/null; do sleep 30; done
if [ -f ../results/log_prep_noisy.txt ]; then
  until grep -q -E "added|SystemExit|Traceback" ../results/log_prep_noisy.txt; do sleep 30; done
fi

$PY train_lora.py --steps $STEPS --bs 4 --accum 4 --lr 1.5e-4 --data v1,v2 --init ../runs/v2/final \
    --specaug --save_every 1000 --name v3 $EXTRA > ../results/log_train_v3.txt 2>&1 \
  && echo TRAIN_V3_OK || { echo TRAIN_V3_FAILED; tail -5 ../results/log_train_v3.txt; exit 1; }

# probe every checkpoint; same clips every time (eval_hf samples with seed 0)
probe() { ck=$1; for s in mnsc_part3 sasrbench; do
    $PY eval_hf.py --set $s --limit 300 --name probe-$ck --model $BASE --adapter ../runs/v3/$ck \
      > ../results/log_probe_${s}_$ck.txt 2>&1 && echo "PROBE $ck $s $(grep '"wer"' ../results/log_probe_${s}_$ck.txt)"
  done; }
for ck in $(ls -d ../runs/v3/step* | xargs -n1 basename | sort -t p -k2 -n); do probe $ck; done
$PY - <<'EOF'
import json, re, glob
best = None
for f in glob.glob("../results/log_probe_mnsc_part3_step*.txt"):
    ck = re.search(r"(step\d+)", f).group(1)
    w3 = float(re.search(r'"wer": ([\d.]+)', open(f).read()).group(1))
    fs = f.replace("mnsc_part3", "sasrbench")
    ws = float(re.search(r'"wer": ([\d.]+)', open(fs).read()).group(1))
    score = (w3 + ws) / 2
    print(f"{ck:10} part3 {w3:5.2f}  sasr {ws:5.2f}  mean {score:5.2f}")
    if best is None or score < best[0]:
        best = (score, ck)
print("BEST_CHECKPOINT", best[1])
EOF
echo V3_DONE
