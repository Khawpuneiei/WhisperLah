# Full scoring of v3 (runs/v3/final = step 7000) on every Singlish test set,
# then the English gate (LibriSpeech, AMI headset and table mic), with the same
# clips, limits and batch size as the v2 table and run_english_gate.sh.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
BASE=openai/whisper-large-v3-turbo
lim() { case $1 in sasrbench) echo 1200;; mnsc_part3) echo 1000;; mnsc_part1|mnsc_part2) echo 500;; librispeech_*|ami|ami_sdm) echo 0;; *) echo 300;; esac; }
ev() { name=$1; set_=$2; shift 2
       log=../results/log_ev_${set_}__${name}.txt
       l=$(lim $set_); [ "$l" = 0 ] && L="" || L="--limit $l"
       if $PY eval_hf.py --set $set_ $L --name $name "$@" > $log 2>&1; then
         echo "OK $name $set_ $(grep '"wer"' $log)"; else echo "FAILED $name $set_"; tail -3 $log; fi; }
V3="--model $BASE --adapter ../runs/v3/final"
NEAR="Near speaker only."
for s in mnsc_part3 sasrbench mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do ev whisperlah-v3 $s $V3; done
ev whisperlah-v3-near sasrbench $V3 --prompt "$NEAR"
ev whisperlah-v3-near mnsc_part3 $V3 --prompt "$NEAR"
for s in librispeech_clean librispeech_other ami ami_sdm; do ev whisperlah-v3 $s $V3 --bs 16; done
$PY report.py 2>&1 | grep -v -i warn
echo ALL_V3_EVALS_DONE
