"""WhisperLah transcription server: faster-whisper (CTranslate2) + speaker diarization.

One process, one GPU. Within a request, diarization runs in a second thread
while ASR runs; across requests, ASR and diarization hold separate locks, so
one part's ASR overlaps another part's diarization. Returns the segment shape
Rapport's `lib/ai/transcribe.ts` expects (speaker label, start, end, text)
plus one embedding per speaker.

    WHISPERLAH_MODEL=models/whisperlah-v2-ct2 python scripts/serve/server.py --port 8765

    curl -F file=@meeting.mp3 -F mode=room http://localhost:8765/transcribe

Environment:
    WHISPERLAH_MODEL     CTranslate2 model dir (from convert_ct2.py)       required
    WHISPERLAH_COMPUTE   float16 | int8_float16 | int8                     default float16
    WHISPERLAH_DEVICE    cuda | cpu                                        default cuda
    WHISPERLAH_BATCH     batch size for the VAD-chunked batched pipeline   default 16
    WHISPERLAH_DIARIZE   community-1 | 3.1 | none                          default community-1 if HF_TOKEN is set
    HF_TOKEN             Hugging Face token with the pyannote model accepted on the Hub
    WHISPERLAH_DIA_BATCH segmentation/embedding batch size for pyannote     default 32
    WHISPERLAH_API_KEY   if set, requests must send `Authorization: Bearer <key>`
"""
import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import fallback  # noqa: E402

if os.name == "nt":  # CTranslate2 finds cuBLAS/cuDNN on PATH; torch cu121 ships them in torch/lib
    import importlib.util
    _torch = importlib.util.find_spec("torch")
    if _torch:
        os.environ["PATH"] = os.path.join(os.path.dirname(_torch.origin), "lib") + os.pathsep + os.environ["PATH"]

log = logging.getLogger("whisperlah")

NEAR_PROMPT = "Near speaker only."

# Chosen stack (21 Oct 2026): pyannote community-1 for diarization, its built-in
# WeSpeaker ResNet34 centroids as the speaker embedding, no fine-tuning.
DIARIZERS = {
    "community-1": ("pyannote/speaker-diarization-community-1", "pyannote/wespeaker-voxceleb-resnet34-LM"),
    "3.1": ("pyannote/speaker-diarization-3.1", "pyannote/wespeaker-voxceleb-resnet34-LM"),
}


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


def parse_bool_form(value: Optional[str], default: bool = True) -> bool:
    """Form fields arrive as strings. An omitted or empty field means `default`;
    newer FastAPI reads an omitted `bool` form field as False (checkbox
    semantics), which once silently turned diarization off."""
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("false", "0", "no", "off")


def embeddings_by_label(labels, vectors) -> dict:
    """Pair pyannote's per-speaker centroids with their labels.

    The centroids are ordered like the labels of the *full* diarization. The
    exclusive diarization can drop a speaker who only ever talks over someone
    else, so zipping with its labels would shift every vector after that one.
    """
    if vectors is None:
        return {}
    return {label: [round(float(x), 5) for x in vec] for label, vec in zip(labels, vectors)}


def relabel(segments: list[dict], names: Optional[dict] = None) -> list[dict]:
    """pyannote names speakers SPEAKER_00, SPEAKER_01 …; Rapport stores A, B, C …
    in order of first appearance. `names` (if given) collects the mapping."""
    names = {} if names is None else names
    for s in segments:
        if s["speaker"] not in names:
            names[s["speaker"]] = chr(ord("A") + len(names))
        s["speaker"] = names[s["speaker"]]
    return segments


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self, model_dir, device, compute, batch_size, diarize, hf_token, dia_batch=32):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        t0 = time.time()
        self.model = WhisperModel(model_dir, device=device, compute_type=compute)
        fallback.install(self.model)  # same compression-ratio rule as eval_hf.py
        self.batched = BatchedInferencePipeline(model=self.model)
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="diarize")
        # Separate locks: pyannote's clustering phase is CPU-bound, so the GPU
        # can run the next part's ASR while this part is still being diarized.
        self.asr_lock = threading.Lock()
        self.dia_lock = threading.Lock()
        log.info("loaded %s (%s, %s) in %.1fs", model_dir, device, compute, time.time() - t0)

        self.diarizer = None
        self.embedding_model = None
        if diarize in DIARIZERS:
            import pyannote.audio
            import torch
            from pyannote.audio import Pipeline

            self.pyannote_v3 = pyannote.audio.__version__.startswith("3.")
            name, self.embedding_model = DIARIZERS[diarize]
            t0 = time.time()
            try:
                self.diarizer = Pipeline.from_pretrained(name, token=hf_token)      # pyannote.audio 4.x
            except TypeError:
                self.diarizer = Pipeline.from_pretrained(name, use_auth_token=hf_token)  # 3.x
            if self.diarizer is None:
                raise SystemExit(f"could not load {name}: accept its terms on the Hub and set HF_TOKEN")
            self.diarizer.to(torch.device(device))
            for attr in ("segmentation_batch_size", "embedding_batch_size"):
                if hasattr(self.diarizer, attr):
                    setattr(self.diarizer, attr, dia_batch)
            log.info("loaded %s in %.1fs", name, time.time() - t0)

    def diarize(self, wav) -> tuple[list[Turn], dict]:
        """Run pyannote; return turns and {pyannote label: centroid embedding}.

        pyannote.audio 4.x returns an object with `speaker_diarization`,
        `exclusive_speaker_diarization` (one speaker at a time, cleaner for
        word assignment) and `speaker_embeddings`; 3.x returns a tuple
        (Annotation, embeddings). Both are handled.
        """
        import torch

        audio = {"waveform": torch.from_numpy(wav)[None], "sample_rate": 16000}
        if self.pyannote_v3:  # 4.x always returns embeddings and warns on the kwarg
            out = self.diarizer(audio, return_embeddings=True)
        else:
            out = self.diarizer(audio)
        if isinstance(out, tuple):
            full, emb = out
            ann = full
        else:
            full = out.speaker_diarization
            ann = getattr(out, "exclusive_speaker_diarization", None) or full
            emb = getattr(out, "speaker_embeddings", None)
        turns = [Turn(seg.start, seg.end, spk) for seg, _, spk in ann.itertracks(yield_label=True)]
        return turns, embeddings_by_label(full.labels(), emb)

    def _diarize_locked(self, wav):
        with self.dia_lock:
            t0 = time.time()
            turns, embeddings = self.diarize(wav)
            return turns, embeddings, round(time.time() - t0, 3)

    def transcribe(self, path: str, mode: str, diarize: bool, language: Optional[str]) -> dict:
        from faster_whisper.audio import decode_audio

        timing = {}
        t_start = t0 = time.time()
        wav = decode_audio(path, sampling_rate=16000)
        duration = len(wav) / 16000
        timing["decode_s"] = round(time.time() - t0, 3)

        # Diarization starts first, in its own thread, and runs while ASR does.
        dia_job = None
        if diarize and self.diarizer is not None:
            dia_job = self.pool.submit(self._diarize_locked, wav)

        common = dict(language=language, task="transcribe", beam_size=1, best_of=1,
                      word_timestamps=True)
        if mode == "near":
            common["initial_prompt"] = NEAR_PROMPT

        with self.asr_lock:
            t0 = time.time()
            # The batched pipeline decodes greedily (it keeps only the first
            # temperature); the few segments that come out as repetition
            # loops are re-decoded with temperature fallback below.
            segs, info = self.batched.transcribe(
                wav, temperature=0.0, condition_on_previous_text=False,
                batch_size=self.batch_size, **common)
            pieces = fallback.pieces_from_segments(list(segs))
            pieces, redecoded = fallback.redecode_loops(self.model, wav, pieces, **common)
            words = [Word(*w) for p in pieces for w in p.words]
            timing["asr_s"] = round(time.time() - t0, 3)

        turns, embeddings = [], {}
        if dia_job is not None:
            turns, embeddings, timing["diarize_s"] = dia_job.result()

        speakers = {}
        if turns:
            names = {}
            segments = relabel(assign_speakers(words, turns), names)
            speakers = {names[label]: {"embedding": vec, "model": self.embedding_model}
                        for label, vec in embeddings.items() if label in names}
        else:
            # No diarizer: keep Whisper's own segments (roughly one per
            # sentence) rather than one block per request, so the transcript
            # still reads and scrolls in Rapport. Everything is speaker A.
            segments = [{"speaker": "A", "start": round(p.start, 3), "end": round(p.end, 3),
                         "text": p.text} for p in pieces if p.text]
        # ASR and diarization overlap, so the stage times no longer add up.
        timing["total_s"] = round(time.time() - t_start, 3)
        return {
            "duration": round(duration, 3),
            "language": info.language,
            "mode": mode,
            "diarized": bool(turns),
            "segments": segments,
            "speakers": speakers,
            "redecoded": redecoded,
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
    diarize: Optional[str] = Form(None),  # str on purpose, see parse_bool_form
    language: Optional[str] = Form(None),
):
    if mode not in ("room", "near"):
        raise HTTPException(400, "mode must be room or near")
    diarize = parse_bool_form(diarize)
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        t0 = time.time()
        # Off the event loop: a blocking call here would serialise every
        # request and make the separate ASR/diarizer locks pointless.
        result = await run_in_threadpool(engine.transcribe, path, mode, diarize, language or None)
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
    diarize = os.environ.get("WHISPERLAH_DIARIZE", "community-1" if hf_token else "none")
    if diarize not in DIARIZERS and diarize != "none":
        raise SystemExit(f"WHISPERLAH_DIARIZE must be one of {', '.join(DIARIZERS)} or none")
    engine = Engine(
        model_dir,
        device=os.environ.get("WHISPERLAH_DEVICE", "cuda"),
        compute=os.environ.get("WHISPERLAH_COMPUTE", "float16"),
        batch_size=int(os.environ.get("WHISPERLAH_BATCH", "16")),
        diarize=diarize,
        hf_token=hf_token,
        dia_batch=int(os.environ.get("WHISPERLAH_DIA_BATCH", "32")),
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
