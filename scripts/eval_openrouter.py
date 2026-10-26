"""OpenAI transcription models through OpenRouter's /audio/transcriptions, on the same clips.

    python scripts/eval_openrouter.py --model openai/gpt-transcribe --name gpt-transcribe --set meetings --limit 0
    python scripts/eval_openrouter.py --model openai/whisper-1 --name whisper-1 --language en --set sasrbench --sample 20

Reads OPENROUTER_API_KEY from --env-file (default .env.openrouter.txt, or the
.txt.txt Notepad sometimes makes) and never prints it. Mirrors eval_openai.py,
except that OpenRouter takes no chunking_strategy, so long clips go up whole.

Every response's usage.cost goes into its row and into a ledger shared by all
runs (data/openrouter_ledger.json). A run stops sending once the ledger total
reaches --budget. Finished rows are appended to <out>.partial as they arrive,
so a rerun resends only what is missing or failed.
"""
import argparse
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from tqdm import tqdm

from common import ROOT, Normalizer, decode_audio, load_set, save_jsonl, wer_report

LEDGER = Path(ROOT) / "data" / "openrouter_ledger.json"


def read_key(path):
    for p in (Path(path), Path(str(path) + ".txt")):
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no OPENROUTER_API_KEY in {path}")


class Ledger:
    def __init__(self, budget):
        self.budget, self.lock = budget, threading.Lock()
        self.total = json.loads(LEDGER.read_text())["total_usd"] if LEDGER.exists() else 0.0

    def can_spend(self):
        with self.lock:
            return self.total < self.budget

    def add(self, cost):
        with self.lock:
            self.total += cost
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            LEDGER.write_text(json.dumps({"total_usd": round(self.total, 6)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int, default=300, help="0 = the whole set")
    ap.add_argument("--sample", type=int, help="then keep this many of those clips (seed 0)")
    ap.add_argument("--model", required=True, help="OpenRouter id, e.g. openai/gpt-4o-transcribe")
    ap.add_argument("--name", required=True, help="results name, as in the main table")
    ap.add_argument("--language", help='e.g. "en" (the Whisper-1 table row uses it)')
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=2.80, help="stop when the shared ledger reaches this (USD)")
    ap.add_argument("--env-file", default=f"{ROOT}/.env.openrouter.txt")
    ap.add_argument("--out-dir", default=f"{ROOT}/results")
    args = ap.parse_args()

    client = OpenAI(api_key=read_key(args.env_file), base_url="https://openrouter.ai/api/v1",
                    max_retries=0, timeout=300)
    ledger = Ledger(args.budget)

    rows = load_set(args.set, args.limit or None)
    if args.sample:
        rows = sorted(random.Random(0).sample(rows, args.sample))
    out_path = Path(args.out_dir) / f"{args.set}__{args.name}.jsonl"
    if args.set == "meetings":  # meeting hypotheses never sit next to published results
        out_path = Path(ROOT) / "results" / "meetings" / out_path.name
    partial = out_path.with_suffix(".jsonl.partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" not in r:
                done[r["idx"]] = r
    todo = [r for r in rows if r[0] not in done]
    print(f"{args.set}: {len(rows)} clips, {len(done)} already done, {len(todo)} to send; "
          f"ledger ${ledger.total:.4f} of ${args.budget:.2f}", flush=True)
    write_lock = threading.Lock()

    def run(row):
        i, b, ref = row
        wav = decode_audio(b)
        sec = len(wav) / 16000
        if not ledger.can_spend():
            return {"idx": i, "ref": ref, "hyp": "", "sec": sec, "error": "budget reached, not sent"}
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV", subtype="PCM_16")
        kw = dict(model=args.model, file=(f"clip{i}.wav", buf.getvalue()), response_format="json")
        if args.language:
            kw["language"] = args.language
        for attempt in range(4):  # first try + 3 retries on 429/5xx/network
            try:
                raw = client.audio.transcriptions.with_raw_response.create(**kw)
                body = raw.http_response.json()
                cost = float((body.get("usage") or {}).get("cost") or 0.0)
                ledger.add(cost)
                r = {"idx": i, "ref": ref, "hyp": (body.get("text") or "").strip(), "sec": sec,
                     "cost": cost, "provider": "openrouter", "model": args.model}
                break
            except (APIStatusError, APIConnectionError, APITimeoutError) as e:
                code = getattr(e, "status_code", None)
                if attempt == 3 or (code is not None and code != 429 and code < 500):
                    r = {"idx": i, "ref": ref, "hyp": "", "sec": sec, "error": f"{code} {str(e)[:200]}"}
                    break
                time.sleep(2 ** attempt * 2)
        with write_lock, partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        new = list(tqdm(ex.map(run, todo), total=len(todo), desc=args.name))
    out = sorted(list(done.values()) + new, key=lambda r: r["idx"])
    save_jsonl(out_path, out)

    errors = [r for r in out if "error" in r]
    norm = Normalizer()
    pairs = [(norm(r["ref"]), norm(r["hyp"])) for r in out if "error" not in r]
    pairs = [(r, h) for r, h in pairs if r]
    rep = wer_report([r for r, _ in pairs], [h for _, h in pairs])
    rep.update(model=args.model, provider="openrouter", set=args.set, name=args.name, language=args.language,
               clips=len(out), errors=len(errors), empty=sum(1 for r in out if "error" not in r and not r["hyp"]),
               audio_hours=round(sum(r["sec"] for r in out) / 3600, 3),
               cost_usd=round(sum(r.get("cost", 0.0) for r in out), 4), ledger_usd=round(ledger.total, 4),
               runtime_s=round(time.time() - t0))
    print(json.dumps(rep), flush=True)
    if any("budget" in r["error"] for r in errors):
        raise SystemExit("BUDGET_REACHED")


if __name__ == "__main__":
    main()
