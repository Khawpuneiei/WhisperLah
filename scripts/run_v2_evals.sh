# Waits for v2 training, then evaluates one model at a time on the GPU,
# most important first. Every Whisper model uses the same decoding.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
MPY=../.venv-meralion/Scripts/python.exe
BASE=openai/whisper-large-v3-turbo
until [ -f ../runs/v2/final/adapter_model.safetensors ] || grep -q -E "Traceback|TRAIN_V2_FAILED" ../results/log_train_v2.txt; do sleep 20; done
sleep 15
lim() { case $1 in sasrbench) echo 1200;; mnsc_part3) echo 1000;; mnsc_part1|mnsc_part2) echo 500;; *) echo 300;; esac; }
ev() { name=$1; set_=$2; shift 2
       log=../results/log_ev_${set_}__${name}.txt
       if $PY eval_hf.py --set $set_ --limit $(lim $set_) --name $name "$@" > $log 2>&1; then
         echo "OK $name $set_ $(grep '"wer"' $log)"; else echo "FAILED $name $set_"; tail -3 $log; fi; }
mer() { set_=$1; log=../results/log_ev_${set_}__MERaLiON-2-3B.txt
        if $MPY eval_meralion.py --set $set_ --limit $(lim $set_) --bs 4 > $log 2>&1; then
          echo "OK MERaLiON-2-3B $set_"; else echo "FAILED MERaLiON-2-3B $set_"; tail -3 $log; fi; }
V2="--model $BASE --adapter ../runs/v2/final"
NEAR="Near speaker only."
ev whisperlah-v2 sasrbench $V2
ev whisperlah-v2 mnsc_part3 $V2
ev whisperlah-v2-near sasrbench $V2 --prompt "$NEAR"
mer sasrbench
mer mnsc_part3
for s in mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do ev whisperlah-v2 $s $V2; done
for s in mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do ev whisper-large-v3-turbo $s --model $BASE; done
for s in mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do ev mjwong-turbo-singlish $s --model mjwong/whisper-large-v3-turbo-singlish; done
for s in mnsc_part1 mnsc_part2 mnsc_part4 mnsc_part5 mnsc_part6; do mer $s; done
ev whisperlah-v2-near mnsc_part3 $V2 --prompt "$NEAR"
$PY report.py 2>&1 | grep -v -i warn
echo ALL_V2_EVALS_DONE
