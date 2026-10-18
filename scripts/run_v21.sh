# v2.1: continue v2 on the clips its 1,400 steps never reached, then re-score.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
BASE=openai/whisper-large-v3-turbo
until grep -q ALL_V2_EVALS_DONE ../results/log_v2_evals_b.txt; do sleep 15; done
$PY train_lora.py --steps 300 --bs 4 --accum 4 --lr 1e-4 --data v1,v2 --init ../runs/v2/final \
    --specaug --start_pos 22400 --save_every 100 --name v2.1 > ../results/log_train_v21.txt 2>&1 \
  && echo TRAIN_V21_OK || { echo TRAIN_V21_FAILED; exit 1; }
lim() { case $1 in sasrbench) echo 1200;; mnsc_part3) echo 1000;; mnsc_part1|mnsc_part2) echo 500;; *) echo 300;; esac; }
ev() { name=$1; set_=$2; shift 2
       log=../results/log_ev_${set_}__${name}.txt
       if $PY eval_hf.py --set $set_ --limit $(lim $set_) --name $name "$@" > $log 2>&1; then
         echo "OK $name $set_ $(grep '"wer"' $log)"; else echo "FAILED $name $set_"; tail -3 $log; fi; }
V="--model $BASE --adapter ../runs/v2.1/final"
ev whisperlah-v2.1-near sasrbench $V --prompt "Near speaker only."
ev whisperlah-v2.1 mnsc_part3 $V
ev whisperlah-v2.1-near mnsc_part3 $V --prompt "Near speaker only."
ev whisperlah-v2.1 sasrbench $V
for s in mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do ev whisperlah-v2.1 $s $V; done
echo V21_DONE
