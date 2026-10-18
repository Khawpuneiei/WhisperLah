# Step 0 gate: English regression. Waits for the ESB download, then scores
# base Whisper turbo and WhisperLah v2 on LibriSpeech clean/other and AMI.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=../.venv/Scripts/python.exe
until grep -q ESB_DONE ../results/download_esb.log; do sleep 15; done
ev() { name=$1; set_=$2; shift 2
       log=../results/log_ev_${set_}__${name}.txt
       if $PY eval_hf.py --set $set_ --name $name --bs 16 "$@" > $log 2>&1; then
         echo "OK $name $set_ $(grep '"wer"' $log)"; else echo "FAILED $name $set_"; tail -3 $log; fi; }
for s in librispeech_clean librispeech_other ami; do
  ev whisper-large-v3-turbo $s --model openai/whisper-large-v3-turbo
  ev whisperlah-v2 $s --model openai/whisper-large-v3-turbo --adapter ../runs/v2/final
done
$PY report.py 2>&1 | grep -E "set |ami|librispeech"
echo ENGLISH_GATE_DONE
