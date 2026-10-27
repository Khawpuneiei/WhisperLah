"""Whole-file transcription jobs: HTTP API, worker step and signed callback.

No platform imports here, so the same logic runs on any host; deploy/modal/app.py wires
it to Modal and deploy/container/ to Cloud Run or SageMaker.

Contract with the Lepaq webapp (agreed 27 Oct 2026):

    POST /jobs            bearer ASR_API_TOKEN
        {job_id, audio_url, callback_url, mode: room|near, num_speakers?,
         min_speakers?, max_speakers?, language?}
        -> 202 {job_id, status: "queued"}; the same job_id again returns its
           current status and is never queued twice.
    GET  /jobs/{job_id}   bearer ASR_API_TOKEN
        -> {job_id, status: queued|running|done|failed, attempts,
            model_version, error?, result?}
    callback              POST callback_url
        headers X-Lepaq-Timestamp: <unix s>,
                X-Lepaq-Signature: v1=<hex HMAC-SHA256(secret, "<ts>.<raw body>")>
        body    {job_id, status: done|failed, model_version, error?, result?}
        2xx = delivered; 4xx = receiver refused, stop; 5xx/network = retry.
"""
import hashlib
import hmac
import json
import os
import tempfile
import time
from typing import Callable, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MAX_AUDIO_BYTES = 1024 * 1024 * 1024  # 1 GB: a few hours of 48 kHz audio
CALLBACK_BACKOFF = (1, 4, 16)          # seconds before retries 1, 2 and 3


class PermanentError(Exception):
    """A job that will fail the same way on every retry (bad URL, bad audio)."""


class JobRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    audio_url: str = Field(pattern=r"^https://")
    callback_url: str = Field(pattern=r"^https://")
    mode: Literal["room", "near"] = "room"
    num_speakers: Optional[int] = Field(None, ge=1, le=20)
    min_speakers: Optional[int] = Field(None, ge=1, le=20)
    max_speakers: Optional[int] = Field(None, ge=1, le=20)
    language: Optional[str] = Field(None, max_length=8)


# ---------------------------------------------------------------- signing

def sign(secret: str, timestamp: int, body: bytes) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return "v1=" + mac.hexdigest()


def verify(secret: str, timestamp: str, signature: str, body: bytes,
           now: Optional[float] = None, max_skew: int = 300) -> bool:
    """What the receiver does; here for tests and as a reference."""
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs((time.time() if now is None else now) - ts) > max_skew:
        return False
    return hmac.compare_digest(sign(secret, ts, body), signature or "")


def send_callback(url: str, payload: dict, secret: str, post: Callable,
                  sleep: Callable = time.sleep, backoff=CALLBACK_BACKOFF) -> bool:
    """POST the signed payload. True once delivered (2xx). A 4xx means the
    receiver refused it for good (unknown job, bad signature), so stop."""
    body = json.dumps(payload, separators=(",", ":")).encode()
    for attempt in range(len(backoff) + 1):
        if attempt:
            sleep(backoff[attempt - 1])
        ts = int(time.time())  # re-signed per attempt so retries never go stale
        headers = {"Content-Type": "application/json", "X-Lepaq-Timestamp": str(ts),
                   "X-Lepaq-Signature": sign(secret, ts, body)}
        try:
            status = post(url, body, headers)
        except Exception:  # network error: retry
            continue
        if 200 <= status < 300:
            return True
        if 400 <= status < 500:
            return False
    return False


# ---------------------------------------------------------------- state

def public(job: dict) -> dict:
    keep = ("job_id", "status", "attempts", "model_version", "error", "result")
    return {k: job[k] for k in keep if job.get(k) is not None}


def bearer_auth(token: str) -> Callable:
    """FastAPI dependency; with no token configured every request is refused."""
    def auth(authorization: Optional[str] = Header(default=None)):
        if not token or not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "bad or missing bearer token")
    return auth


def make_api(store, enqueue: Callable[[dict], None], token: str, model_version: str) -> FastAPI:
    """`store` is dict-like with get(key) and put(key, value, skip_if_exists=)
    (a modal.Dict, or a FileStore); `enqueue` hands a job to the worker queue."""
    app = FastAPI(title="Lepaq ASR jobs")
    auth = bearer_auth(token)

    @app.get("/health")
    def health():
        return {"ok": True, "model_version": model_version}

    @app.post("/jobs", dependencies=[Depends(auth)], status_code=202)
    def create(req: JobRequest):
        job = {**req.model_dump(), "status": "queued", "attempts": 0,
               "model_version": model_version, "created_at": time.time()}
        if not store.put(req.job_id, job, skip_if_exists=True):
            return JSONResponse(public(store.get(req.job_id)), status_code=200)
        try:
            enqueue(job)
        except Exception:
            store.put(req.job_id, {**job, "status": "failed", "error": "could not queue"})
            raise HTTPException(503, "could not queue the job; retry with the same job_id later")
        return {"job_id": req.job_id, "status": "queued"}

    @app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def status(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job_id")
        return public(job)

    return app


# ---------------------------------------------------------------- worker step

def download(url: str, dest: str, get_stream: Callable, max_bytes: int = MAX_AUDIO_BYTES):
    """Stream `url` to `dest`. A 4xx (expired or wrong signed URL) is
    permanent; a 5xx or a dropped connection is left to the job retries."""
    with get_stream(url) as r:
        if 400 <= r.status_code < 500:
            raise PermanentError(f"audio_url returned {r.status_code}")
        r.raise_for_status()
        n = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                n += len(chunk)
                if n > max_bytes:
                    raise PermanentError(f"audio is larger than {max_bytes // 2**20} MB")
                f.write(chunk)


def run_job(job: dict, *, store, transcribe: Callable, get_stream: Callable,
            post: Callable, secret: str, max_attempts: int, sleep=time.sleep) -> Optional[dict]:
    """One attempt at a job. Raises on transient errors so the queue retries
    it; sends the callback once the job is done or has failed for good."""
    job_id = job["job_id"]
    state = store.get(job_id) or job
    if state.get("status") in ("done", "failed"):
        return state  # a duplicate delivery; already settled
    attempts = state.get("attempts", 0) + 1
    store.put(job_id, {**state, "status": "running", "attempts": attempts, "error": None})

    def finish(status, result=None, error=None):
        final = {**state, "status": status, "attempts": attempts,
                 "result": result, "error": error, "finished_at": time.time()}
        store.put(job_id, final)
        payload = {"job_id": job_id, "status": status,
                   "model_version": state.get("model_version")}
        if error:
            payload["error"] = error
        if result is not None:
            payload["result"] = result
        send_callback(job["callback_url"], payload, secret, post, sleep=sleep)
        return final

    suffix = os.path.splitext(job["audio_url"].split("?")[0])[1][:8] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        download(job["audio_url"], path, get_stream)
        result = transcribe(path, job.get("mode", "room"), True, job.get("language"),
                            job.get("num_speakers"), job.get("min_speakers"),
                            job.get("max_speakers"))
        # Plain JSON types only: numpy floats from the models would make the
        # stored job unreadable wherever numpy is not installed.
        result = json.loads(json.dumps(result, default=float))
        return finish("done", result=result)
    except PermanentError as e:
        return finish("failed", error=str(e))
    except Exception as e:
        if attempts >= max_attempts:
            finish("failed", error=f"{type(e).__name__}: {e}"[:500])
        else:
            store.put(job_id, {**state, "status": "queued", "attempts": attempts,
                               "error": f"{type(e).__name__}: {e}"[:500]})
        raise
    finally:
        os.unlink(path)
