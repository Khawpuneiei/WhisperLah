# Project rules

- NEVER commit, push, upload or otherwise publish model weights or LoRA adapters
  (`*.safetensors`, `*.pt`, `*.bin` checkpoints, anything in `models/` or `runs/`).
  They are the startup's deep-tech advantage. Keep them local only.
- Do not add weights to Hugging Face, releases, artifacts or any other host
  without the owner's explicit, per-action approval.
- Code, benchmark harness and results tables may be published; weights may not.
