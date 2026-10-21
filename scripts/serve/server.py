"""WhisperLah transcription server: faster-whisper (CTranslate2) + speaker diarization.

One process, one GPU, requests served one at a time. Returns the segment shape
Rapport's `lib/ai/transcribe.ts` expects: speaker label, start, end, text.

    WHISPERLAH_MODEL=models/whisperlah-v2-ct2 python scripts/serve/server.py --port 8765

    curl -F file=@meeting.mp3 -F mode=room http://localhost:8765/transcribe

Environment:
    WHISPERLAH_MODEL     CTranslate2 model dir (from convert_ct2.py)       required
    WHISPERLAH_COMPUTE   float16 | int8_float16 | int8                     default float16
    WHISPERLAH_DEVICE    cuda | cpu                                        default cuda
    WHISPERLAH_BATCH     batch size for the VAD-chunked batched pipeline   default 16
    WHISPERLAH_DIARIZE   pyannote | none                                   default pyannote if HF_TOKEN is set
    HF_TOKEN             Hugging Face token with pyannote/speaker-diarization-3.1 accepted
    WHISPERLAH_API_KEY   if set, requests must send `Authorization: Bearer <key>`
"""
import argparse
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger("whisperlah")

NEAR_PROMPT = "Near speaker only."
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------- speaker attribution

@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def assign_speakers(words: list[Word], turns: list[Turn], default="A") -> list[dict]:
    """Give each word the speaker whose diarization turn overlaps it most, then
    group consecutive same-speaker words into segments.

    A word that overlaps no turn (diarizer missed it, or a tiny gap between
    turns) takes the nearest turn by midpoint, so no word is dropped: dropped
    words are what Rapport would show as a hole in the transcript.
    """
    if not words:
        return []
    if not turns:
        return [{"speaker": default, "start": words[0].start, "end": words[-1].end,
                 "text": " ".join(w.text for w in words).strip()}]

    turns = sorted(turns, key=lambda t: t.start)
    labelled = []
    j = 0
    for w in words:
        # turns are sorted; skip those that end before this word starts
        while j < len(turns) and turns[j].end < w.start:
            j += 1
        best, best_ov = None, 0.0
        k = j
        while k < len(turns) and turns[k].start <= w.end:
            ov = min(w.end, turns[k].end) - max(w.start, turns[k].start)
            if ov > best_ov:
                best, best_ov = turns[k].speaker, ov
            k += 1
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid))).speaker
        labelled.append((best, w))

    segments = []
    for spk, w in labelled:
        if segments and segments[-1]["speaker"] == spk:
            segments[-1]["end"] = w.end
            segments[-1]["text"] += " " + w.text
        else:
            segments.append({"speaker": spk, "start": w.start, "end": w.end, "text": w.text})
    for s in segments:
        s["text"] = s["text"].strip()
    return [s for s in segments if s["text"]]


def relabel(segments: list[dict]) -> list[dict]:
    """pyannote names speakers SPEAKER_00, SPEAKER_01 …; Rapport stores A, B, C …"""
    names = {}
    for s in segments:
        if s["speaker"] not in names:
            names[s["speaker"]] = chr(ord("A") + len(names))
        s["speaker"] = names[s["speaker"]]
    return segments


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self, model_dir, device, compute, batch_size, diarize, hf_token):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        t0 = time.time()
        self.model = WhisperModel(model_dir, device=device, compute_type=compute)
        self.batched = BatchedInferencePipeline(model=self.model)
        self.batch_size = batch_size
        self.lock = threading.Lock()
        log.info("loaded %s (%s, %s) in %.1fs", model_dir, device, compute, time.time() - t0)

        self.diarizer = None
        if diarize == "pyannote":
            import torch
            from pyannote.audio import Pipeline

            t0 = time.time()
            self.diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            self.diarizer.to(torch.device(device))
            log.info("loaded pyannote diarization in %.1fs", time.time() - t0)

    def transcribe(self, path: str, mode: str, diarize: bool, language: Optional[str]) -> dict:
        from faster_whisper.audio import decode_audio

        timing = {}
        t0 = time.time()
        wav = decode_audio(path, sampling_rate=16000)
        duration = len(wav) / 16000
        timing["decode_s"] = round(time.time() - t0, 3)

        kw = dict(language=language, task="transcribe", beam_size=1, best_of=1,
                  temperature=TEMPERATURES, compression_ratio_threshold=1.35,
                  condition_on_previous_text=False, word_timestamps=True,
                  batch_size=self.batch_size)
        if mode == "near":
            kw["initial_prompt"] = NEAR_PROMPT

        with self.lock:
            t0 = time.time()
            segs, info = self.batched.transcribe(wav, **kw)
            words = [Word(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or [])]
            timing["asr_s"] = round(time.time() - t0, 3)

            turns = []
            if diarize and self.diarizer is not None:
                import torch

                t0 = time.time()
                dia = self.diarizer({"waveform": torch.from_numpy(wav)[None], "sample_rate": 16000})
                turns = [Turn(seg.start, seg.end, spk) for seg, _, spk in dia.itertracks(yield_label=True)]
                timing["diarize_s"] = round(time.time() - t0, 3)

        segments = relabel(assign_speakers(words, turns))
        timing["total_s"] = round(sum(timing.values()), 3)
        return {
            "duration": round(duration, 3),
            "language": info.language,
            "mode": mode,
            "diarized": bool(turns),
            "segments": segments,
            "timing": timing,
            "rtf": round(timing["total_s"] / duration, 4) if duration else None,
        }


# ---------------------------------------------------------------- app

app = FastAPI(title="WhisperLah")
engine: Optional[Engine] = None


def auth(authorization: Optional[str] = Header(default=None)):
    key = os.environ.get("WHISPERLAH_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(401, "bad or missing bearer token")


@app.get("/health")
def health():
    return {"ok": engine is not None, "diarize": bool(engine and engine.diarizer)}


@app.post("/transcribe", dependencies=[Depends(auth)])
async def transcribe(
    file: UploadFile = File(...),
    mode: str = Form("room"),
    diarize: bool = Form(True),
    language: Optional[str] = Form(None),
):
    if mode not in ("room", "near"):
        raise HTTPException(400, "mode must be room or near")
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        t0 = time.time()
        result = engine.transcribe(path, mode, diarize, language or None)
        result["timing"]["wall_s"] = round(time.time() - t0, 3)
        return JSONResponse(result)
    finally:
        os.unlink(path)


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_dir = os.environ.get("WHISPERLAH_MODEL")
    if not model_dir:
        raise SystemExit("set WHISPERLAH_MODEL to a CTranslate2 model dir (see convert_ct2.py)")
    hf_token = os.environ.get("HF_TOKEN")
    diarize = os.environ.get("WHISPERLAH_DIARIZE", "pyannote" if hf_token else "none")
    engine = Engine(
        model_dir,
        device=os.environ.get("WHISPERLAH_DEVICE", "cuda"),
        compute=os.environ.get("WHISPERLAH_COMPUTE", "float16"),
        batch_size=int(os.environ.get("WHISPERLAH_BATCH", "16")),
        diarize=diarize,
        hf_token=hf_token,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
