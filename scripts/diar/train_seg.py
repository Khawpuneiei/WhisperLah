"""Fine-tune community-1's segmentation model on our Singlish RTTM sets.

Only the segmentation network (~1.5 M params) is trained; the WeSpeaker
ResNet34 embedding and the clustering stay as shipped. Weights land in
runs/diar/<name>/ and, like every WhisperLah weight, stay on this machine.

    .venv-diar/Scripts/python.exe scripts/diar/train_seg.py --name seg-ft --epochs 20

Writes data/diar/database.yml (pyannote.database protocol over the
train/dev/test splits from make_synth.py) on first use.
"""
import argparse
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "diar" / "database.yml"


def hf_token():
    for name in (".env.hf.txt", ".env.hf.txt.txt"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def write_database():
    # paths are relative to this file; {uri} names are unique across splits
    sets = {"train": "train", "development": "dev", "test": "test"}
    lines = ["Databases:", "  SynthSinglish: ./*/wav/{uri}.wav", "", "Protocols:", "  SynthSinglish:",
             "    SpeakerDiarization:", "      mix:"]
    for subset, d in sets.items():
        lines += [f"        {subset}:", f"          uri: {d}/{d}.lst",
                  f"          annotation: {d}/{d}.rttm", f"          annotated: {d}/{d}.uem"]
    DB.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seg-ft")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from pyannote.audio import Model
    from pyannote.audio.tasks import SpeakerDiarization
    from pyannote.database import FileFinder, registry
    try:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

    if not DB.exists():
        write_database()
    registry.load_database(str(DB))
    protocol = registry.get_protocol("SynthSinglish.SpeakerDiarization.mix",
                                     preprocessors={"audio": FileFinder()})

    model = Model.from_pretrained("pyannote/speaker-diarization-community-1", subfolder="segmentation",
                                  token=hf_token())
    # same chunking as the pretrained model; num_workers=0 because Windows spawns slowly
    task = SpeakerDiarization(protocol, duration=model.specifications.duration,
                              max_speakers_per_chunk=len(model.specifications.classes),
                              max_speakers_per_frame=model.specifications.powerset_max_classes,
                              batch_size=args.batch, num_workers=0)
    model.task = task
    model.prepare_data()
    model.setup()

    def configure_optimizers():
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    model.configure_optimizers = configure_optimizers

    out = ROOT / "runs" / "diar" / args.name
    ckpt = ModelCheckpoint(dirpath=out, filename="{epoch:02d}", save_top_k=-1, every_n_epochs=1)
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                      max_epochs=args.epochs, callbacks=[ckpt], default_root_dir=out, log_every_n_steps=10)
    trainer.fit(model)
    print("best/last:", ckpt.last_model_path or sorted(out.glob("*.ckpt"))[-1], "\nTRAIN_SEG_DONE", flush=True)


if __name__ == "__main__":
    main()
