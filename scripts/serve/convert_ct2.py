"""Merge a WhisperLah LoRA adapter into whisper-large-v3-turbo and convert the
result to CTranslate2, the runtime faster-whisper uses.

    python scripts/serve/convert_ct2.py --adapter models/whisperlah-v2-lora --out models/whisperlah-v2-ct2
    python scripts/serve/convert_ct2.py --adapter models/whisperlah-v2-lora --out models/whisperlah-v2-ct2-int8 --quant int8_float16

Output goes under models/, which is gitignored: the merged weights are as
proprietary as the adapter and never leave this machine.

Quantization is a load-time choice in CTranslate2 (compute_type), so one fp16
conversion serves both fp16 and int8_float16 inference. Converting int8 here
only shrinks the file on disk (1.6 GB -> 0.8 GB).
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-large-v3-turbo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA adapter dir; omit to convert the plain base model")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="float16",
                    choices=["float16", "int8_float16", "int8", "bfloat16", "float32"])
    ap.add_argument("--keep-merged", help="also save the merged HF model to this dir")
    args = ap.parse_args()

    from ctranslate2.converters import TransformersConverter

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")

    # CTranslate2's converter wants a Transformers model directory, so the
    # merged model goes to disk first (a temp dir unless --keep-merged).
    merged_dir = Path(args.keep_merged) if args.keep_merged else Path(tempfile.mkdtemp(prefix="whisperlah-merged-"))
    try:
        model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float16)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
            print(f"merged {args.adapter} into {args.base}")
        model.save_pretrained(merged_dir, safe_serialization=True)
        WhisperProcessor.from_pretrained(args.base).save_pretrained(merged_dir)
        del model

        conv = TransformersConverter(
            str(merged_dir),
            copy_files=["tokenizer.json", "preprocessor_config.json"],
            load_as_float16=True,
        )
        conv.convert(str(out), quantization=args.quant, force=False)
        size = sum(p.stat().st_size for p in out.rglob("*")) / 1e9
        print(f"wrote {out} ({args.quant}, {size:.2f} GB)")
    finally:
        if not args.keep_merged:
            shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
