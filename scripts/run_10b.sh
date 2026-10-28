# MERaLiON-2-10B-ASR in 4-bit on the two headline Singlish sets. Runs after the
# English gate (GPU is free) and before v3 training starts. Always writes a
# marker so the next queued job never waits forever.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
MPY=../.venv-meralion/Scripts/python.exe
until grep -q ENGLISH_GATE_DONE ../results/log_english_gate.txt; do sleep 30; done
until grep -q -E "MERALION_10B_READY|Error|Traceback" ../results/log_meralion10b_dl.txt; do sleep 30; done
if grep -q MERALION_10B_READY ../results/log_meralion10b_dl.txt; then
  for s in sasrbench mnsc_part3; do
    lim=1200; [ $s = mnsc_part3 ] && lim=1000
    log=../results/log_ev_${s}__MERaLiON-2-10B-ASR.txt
    if $MPY eval_meralion.py --model MERaLiON/MERaLiON-2-10B-ASR --load-in-4bit --set $s --limit $lim --bs 2 > $log 2>&1; then
      echo "OK MERaLiON-2-10B-ASR $s"; else echo "FAILED MERaLiON-2-10B-ASR $s"; tail -4 $log; fi
  done
else
  echo "10B download failed; skipping"
fi
echo TENB_EVAL_DONE
