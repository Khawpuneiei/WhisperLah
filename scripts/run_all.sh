# One process at a time on the GPU: training, then each eval in turn.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
$PY train_lora.py --steps 500 --bs 4 --accum 4 --lr 3e-4 --rank 32 --save_every 125 --name lora-r32 > ../results/log_train.txt 2>&1 \
  && echo "TRAIN_OK" || { echo "TRAIN_FAILED"; tail -5 ../results/log_train.txt; exit 1; }
ev() { log=../results/log_final_$(echo "$*" | md5sum | cut -c1-8).txt
       $PY eval_hf.py "$@" > $log 2>&1 && echo "OK $* $(grep '"wer"' $log)" || { echo "FAILED $*"; tail -3 $log; }; }
ev --model openai/whisper-large-v3-turbo --adapter ../runs/lora-r32/final --set mnsc_part3 --name lora-r32
ev --model openai/whisper-large-v3-turbo --set mnsc_part3 --name whisper-large-v3-turbo
ev --model mjwong/whisper-large-v3-turbo-singlish --set mnsc_part3 --name mjwong-turbo-singlish
ev --model openai/whisper-large-v3-turbo --adapter ../runs/lora-r32/final --set sasrbench --limit 1200 --greedy --name lora-r32
$PY report.py 2>&1 | grep -v -i warn
python -I chart.py
echo ALL_DONE
