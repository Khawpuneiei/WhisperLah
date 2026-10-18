# AMI into the store once the NSC streaming has finished (one appender at a time).
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
until grep -q "^done:" ../results/log_prep_stream.txt; do sleep 30; done
until grep -q AMI_PROBE_DONE ../results/download_ami_probe.log; do sleep 30; done
../.venv/Scripts/python.exe prep_ami.py --mics ihm sdm --shards 20 && echo AMI_PREP_OK || echo AMI_PREP_FAILED
