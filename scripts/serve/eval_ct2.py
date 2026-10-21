"""WER and speed of a CTranslate2 (faster-whisper) WhisperLah model on a test set.

Same clips, same normaliser and same summary.jsonl row format as eval_hf.py,
so the numbers sit in the same table. Decoding mirrors eval_hf.py: greedy with
Whisper's temperature fallback at compression ratio 1.35.

    python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --set mnsc_part3 --limit 300 --name ct2-fp16
    python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --compute int8_float16 --set sasrbench --name ct2-int8 --prompt "Near speaker only."
    python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --batched --bs 16 --set mnsc_part3 --name ct2-batched
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())  # scripts/
from common import ROOT, Normalizer, decode_audio, load_set, save_jsonl, wer_report  # noqa: E402

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

if os.name == "nt":  # CTranslate2 finds cuBLAS/cuDNN on PATH; torch cu121 ships them in torch/lib
    import importlib.util
    _torch = importlib.util.find_spec("torch")
    if _torch:
        os.environ["PATH"] = os.path.join(os.path.dirname(_torch.origin), "lib") + os.pathsep + os.environ["PATH"]


def transcribe_clip(model, wav, args):
    kw = dict(language="en", task="transcribe", beam_size=1, best_of=1,
              temperature=(0.0,) if args.greedy else TEMPERATURES,
              compression_ratio_threshold=1.35,
              log_prob_threshold=None, no_speech_threshold=None,  # eval_hf has no such gates
              condition_on_previous_text=False, vad_filter=args.vad,
              without_timestamps=True)
    if args.prompt:
        kw["initial_prompt"] = args.prompt
    if args.batched:
        segs, _ = model.transcribe(wav, batch_size=args.bs, **kw)
    else:
        segs, _ = model.transcribe(wav, **kw)
    return " ".join(s.text.strip() for s in segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CTranslate2 model dir from convert_ct2.py")
    ap.add_argument("--set", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compute", default="float16", help="float16 | int8_float16 | int8 | float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batched", action="store_true", help="BatchedInferencePipeline (VAD-chunked batching)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--vad", action="store_true", help="Silero VAD filter before decoding")
    ap.add_argument("--greedy", action="store_true", help="no temperature fallback")
    ap.add_argument("--prompt", help='e.g. "Near speaker only." for near-mic mode')
    args = ap.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    rows = load_set(args.set, args.limit)
    audio = [(i, decode_audio(b), ref) for i, b, ref in rows]
    hours = sum(len(w) for _, w, _ in audio) / 16000 / 3600
    print(f"{args.set}: {len(audio)} clips, {hours:.2f} h", flush=True)

    t_load = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)
    if args.batched:
        model = BatchedInferencePipeline(model=model)
    print(f"loaded {args.model} ({args.compute}) in {time.time() - t_load:.1f}s", flush=True)

    out, t0 = [], time.time()
    for i, wav, ref in tqdm(audio, desc=args.name):
        out.append({"idx": i, "ref": ref, "hyp": transcribe_clip(model, wav, args).strip()})
    secs = time.time() - t0

    norm = Normalizer()
    pairs = [(norm(r["ref"]), norm(r["hyp"])) for r in out]
    pairs = [(r, h) for r, h in pairs if r]
    rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
    rep.update(model=args.model, adapter=None, set=args.set, name=args.name,
               decoding=("greedy" if args.greedy else "greedy+temperature_fallback")
                        + ("+batched" if args.batched else "") + ("+vad" if args.vad else ""),
               runtime="ctranslate2", compute_type=args.compute, prompt=args.prompt,
               audio_hours=round(hours, 2), runtime_s=round(secs), rtf=round(secs / (hours * 3600), 4))
    print(json.dumps(rep, indent=1), flush=True)

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    save_jsonl(f"{ROOT}/results/{args.set}__{args.name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    with open(f"{ROOT}/results/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()
