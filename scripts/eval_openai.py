"""WER of the OpenAI transcription model Rapport uses in production, on the same clips.

Reads OPENAI_API_KEY from the environment, or from --env-file, and never prints it.

    python scripts/eval_openai.py --set mnsc_part3 --limit 300
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from openai import OpenAI
from tqdm import tqdm

from common import ROOT, Normalizer, decode_audio, load_set, save_jsonl, wer_report


def read_env(path=None):
    env = dict(os.environ)
    if not path:
        return env
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", help="dotenv file holding OPENAI_API_KEY")
    ap.add_argument("--language", help='e.g. "en"; off by default, as in the headline runs')
    ap.add_argument("--chunking", action="store_true", help='send chunking_strategy="auto"')
    ap.add_argument("--name", help="results name; defaults to the model id")
    args = ap.parse_args()

    env = read_env(args.env_file)
    model = args.model or env.get("OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-transcribe-diarize"
    client = OpenAI(api_key=env["OPENAI_API_KEY"])
    diarize = "diarize" in model
    name = args.name or model

    rows = load_set(args.set, args.limit)
    hours = 0.0

    def run(row):
        i, b, ref = row
        wav = decode_audio(b)
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        kw = dict(model=model, file=(f"clip{i}.wav", buf.read()))
        if diarize:
            kw.update(response_format="diarized_json", chunking_strategy="auto")
        if args.language:
            kw["language"] = args.language
        if args.chunking:
            kw["chunking_strategy"] = "auto"
        for attempt in range(5):
            try:
                r = client.audio.transcriptions.create(**kw)
                text = getattr(r, "text", None)
                if not text and getattr(r, "segments", None):
                    text = " ".join(s.text for s in r.segments)
                return {"idx": i, "ref": ref, "hyp": (text or "").strip(), "sec": len(wav) / 16000}
            except Exception as e:  # rate limits and transient 5xx
                if attempt == 4:
                    return {"idx": i, "ref": ref, "hyp": "", "sec": len(wav) / 16000, "error": str(e)[:200]}
                time.sleep(2 ** attempt * 2)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        out = list(tqdm(ex.map(run, rows), total=len(rows), desc=model))
    secs = time.time() - t0
    hours = sum(r["sec"] for r in out) / 3600
    errors = sum("error" in r for r in out)

    norm = Normalizer()
    pairs = [(norm(r["ref"]), norm(r["hyp"])) for r in out if "error" not in r]
    pairs = [(r, h) for r, h in pairs if r]
    rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
    rep.update(model=model, adapter=None, set=args.set, name=name, language=args.language, chunking=args.chunking, audio_hours=round(hours, 2),
               runtime_s=round(secs), api_errors=errors, est_cost_usd=round(hours * 60 * 0.006, 2))
    print(json.dumps(rep, indent=1), flush=True)

    save_jsonl(f"{ROOT}/results/{args.set}__{name}.jsonl", sorted(out, key=lambda r: r["idx"]))
    with open(f"{ROOT}/results/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()
