# Between the 10B eval and v3 training: score the AMI table-mic test for the
# base model and v2 while the GPU is free, and re-run any gate eval that failed.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
BASE=openai/whisper-large-v3-turbo
until grep -q ENGLISH_GATE_DONE ../results/log_english_gate.txt; do sleep 30; done
until grep -q TENB_EVAL_DONE ../results/log_10b.txt 2>/dev/null; do sleep 30; done
ev() { name=$1; set_=$2; shift 2
       log=../results/log_ev_${set_}__${name}.txt
       if $PY eval_hf.py --set $set_ --name $name --bs 16 "$@" > $log 2>&1; then
         echo "OK $name $set_ $(grep '"wer"' $log)"; else echo "FAILED $name $set_"; tail -3 $log; fi; }
# gate evals that did not finish (a FAILED line, or no result file)
for s in librispeech_clean librispeech_other ami; do
  [ -f ../results/${s}__whisper-large-v3-turbo.jsonl ] || ev whisper-large-v3-turbo $s --model $BASE
  [ -f ../results/${s}__whisperlah-v2.jsonl ] || ev whisperlah-v2 $s --model $BASE --adapter ../runs/v2/final
done
ev whisper-large-v3-turbo ami_sdm --model $BASE
ev whisperlah-v2 ami_sdm --model $BASE --adapter ../runs/v2/final
$PY report.py 2>&1 | grep -E "set |ami|librispeech"
echo PRE_V3_DONE
