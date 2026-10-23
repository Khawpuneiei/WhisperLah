"""Time one recording through the WhisperLah server and through the OpenAI API,
the way Rapport does it: split into parts (AUDIO_CHUNK_SECONDS, default 600),
up to AUDIO_CHUNK_CONCURRENCY (default 3) parts in flight.

    python scripts/serve/bench.py meeting.mp3 --server http://localhost:8765
    python scripts/serve/bench.py meeting.mp3 --server http://localhost:8765 --openai --env-file ../rapport/.env.local
    python scripts/serve/bench.py meeting.mp3 --server http://localhost:8765 --gpu-usd-per-hour 0.80

Prints wall-clock, real-time factor and cost per audio hour for each backend
and appends a row per backend to results/bench.jsonl. The OpenAI price is
US$0.006 per audio minute (gpt-4o-transcribe-diarize); the local price is the
GPU rental rate times wall-clock, zero for a laptop you already own.
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())  # scripts/
from common import ROOT  # noqa: E402
from eval_openai import read_env  # noqa: E402

OPENAI_USD_PER_MIN = 0.006


def split(wav, sr, part_s):
    n = int(part_s * sr)
    return [(i / sr, wav[i:i + n]) for i in range(0, len(wav), n)]


def to_wav_bytes(wav, sr):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_parts(parts, fn, concurrency):
    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(lambda p: fn(*p), parts))
    return time.time() - t0, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--mode", default="room")
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--openai", action="store_true", help="also time the OpenAI API")
    ap.add_argument("--env-file", help="dotenv holding OPENAI_API_KEY (never printed)")
    ap.add_argument("--part-seconds", type=int, default=int(os.environ.get("AUDIO_CHUNK_SECONDS", 600)))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("AUDIO_CHUNK_CONCURRENCY", 3)))
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.0)
    ap.add_argument("--label", default=None, help="name for the local row, e.g. 4060-int8")
    args = ap.parse_args()

    from faster_whisper.audio import decode_audio

    wav = decode_audio(args.audio, sampling_rate=16000)
    sr = 16000
    hours = len(wav) / sr / 3600
    parts = split(wav, sr, args.part_seconds)
    print(f"{args.audio}: {hours * 60:.1f} min, {len(parts)} parts of {args.part_seconds}s, "
          f"{args.concurrency} in flight", flush=True)
    rows = []

    # -- local server (parts run concurrently, but the server serialises them on one GPU)
    client = httpx.Client(timeout=3600)
    headers = {}
    if os.environ.get("WHISPERLAH_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['WHISPERLAH_API_KEY']}"

    def local(offset, part):
        r = client.post(f"{args.server}/transcribe", headers=headers,
                        files={"file": ("part.wav", to_wav_bytes(part, sr), "audio/wav")},
                        data={"mode": args.mode, "diarize": str(not args.no_diarize).lower()})
        r.raise_for_status()
        return r.json()

    health = client.get(f"{args.server}/health").json()
    wall, res = run_parts(parts, local, args.concurrency)
    n_seg = sum(len(r["segments"]) for r in res)
    asr_s = sum(r["timing"].get("asr_s", 0) for r in res)
    dia_s = sum(r["timing"].get("diarize_s", 0) for r in res)
    rows.append({
        "backend": args.label or "whisperlah-local", "diarized": health.get("diarize") and not args.no_diarize,
        "wall_s": round(wall, 1), "rtf": round(wall / (hours * 3600), 4),
        "asr_s": round(asr_s, 1), "diarize_s": round(dia_s, 1), "segments": n_seg,
        "parts": len(res), "speakers": sum(len(r.get("speakers", {})) for r in res),
        "redecoded": sum(r.get("redecoded", 0) for r in res),
        "usd_per_audio_hour": round(args.gpu_usd_per_hour * wall / 3600 / hours, 4),
    })

    # -- OpenAI
    if args.openai:
        from openai import OpenAI

        env = read_env(args.env_file)
        model = env.get("OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-transcribe-diarize"
        oai = OpenAI(api_key=env["OPENAI_API_KEY"])

        def remote(offset, part):
            r = oai.audio.transcriptions.create(
                model=model, file=("part.wav", to_wav_bytes(part, sr)),
                response_format="diarized_json", chunking_strategy="auto")
            return r.model_dump()

        wall, res = run_parts(parts, remote, args.concurrency)
        rows.append({
            "backend": model, "diarized": True,
            "wall_s": round(wall, 1), "rtf": round(wall / (hours * 3600), 4),
            "segments": sum(len(r.get("segments") or []) for r in res),
            "usd_per_audio_hour": round(OPENAI_USD_PER_MIN * 60, 4),
        })

    for r in rows:
        r.update(audio=os.path.basename(args.audio), audio_min=round(hours * 60, 1),
                 part_seconds=args.part_seconds, concurrency=args.concurrency)
        print(json.dumps(r), flush=True)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/bench.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
