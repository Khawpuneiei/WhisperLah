"""Copy the pyannote diarization pipeline into a plain folder that loads offline.

The server and the Modal worker load the pipeline from a local folder
(WHISPERLAH_DIARIZE=<dir>, HF_HUB_OFFLINE=1), so the cloud never needs a
Hugging Face token. This copies the Hub snapshot (config.yaml plus the
segmentation, WeSpeaker ResNet34 embedding and PLDA files), then checks that
it loads with the Hub switched off.

    python scripts/serve/export_pyannote.py --out models/pyannote-community-1
    python scripts/serve/export_pyannote.py --out models/pyannote-community-1-segft --segmentation runs/diar/seg-ft/best.ckpt
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--segmentation", help="fine-tuned segmentation checkpoint to put in place of the stock one")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    snapshot_download(args.repo, local_dir=out, token=os.environ.get("HF_TOKEN"))
    shutil.rmtree(out / ".cache", ignore_errors=True)
    if args.segmentation:
        target = next((out / "segmentation").glob("*.bin"), None) or next((out / "segmentation").iterdir())
        shutil.copyfile(args.segmentation, target)
        print(f"segmentation replaced: {target.name} <- {args.segmentation}")

    # Load it in a fresh process with the Hub off, the way the worker will.
    check = ("import sys; from pyannote.audio import Pipeline; "
             "p = Pipeline.from_pretrained(sys.argv[1]); print('loads offline:', type(p).__name__)")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env.pop("HF_TOKEN", None)
    subprocess.run([sys.executable, "-c", check, str(out)], env=env, check=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {size / 2**20:.1f} MB")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  {f.stat().st_size}")


if __name__ == "__main__":
    main()
