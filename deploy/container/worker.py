"""Platform-neutral job server: the same container on Cloud Run, SageMaker or a laptop.

Routes
    POST /jobs, GET /jobs/{id}   the webapp contract (see scripts/serve/jobs.py)
    POST /run                    run one job now; a queue (Cloud Tasks) calls this.
                                 2xx = settled, 503 = transient, retry me
    GET /ping, POST /invocations SageMaker's container contract (async inference
                                 puts the job JSON in S3 and calls /invocations)
    GET /health

Environment
    MODELS_DIR / MODEL_VERSION   weights at $MODELS_DIR/$MODEL_VERSION/{whisper-ct2,pyannote}
    ASR_API_TOKEN, ASR_CALLBACK_SECRET
    ASR_QUEUE        inline (default): jobs run in this container, 2 at a time.
                     external: POST /jobs only records the job; the host's queue
                     must call /run (wired once the host is chosen).
    ASR_STATE_DIR    where job state is kept (default /tmp/asr-jobs). Use a
                     shared mount, or swap FileStore for the host's database,
                     when more than one container serves status lookups.
    ASR_MAX_ATTEMPTS 3 (first try + 2 retries)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.environ.get("WHISPERLAH_CODE", "/app"))
import jobs as J  # noqa: E402

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3")
MAX_ATTEMPTS = int(os.environ.get("ASR_MAX_ATTEMPTS", "3"))


class FileStore:
    """One JSON file per job; put(skip_if_exists=True) is atomic (O_EXCL)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.root / f"{key}.json"  # job_id is validated to [A-Za-z0-9_-]

    def get(self, key):
        try:
            return json.loads(self._path(key).read_text())
        except FileNotFoundError:
            return None

    def put(self, key, value, skip_if_exists=False):
        data = json.dumps(value).encode()
        if skip_if_exists:
            try:
                fd = os.open(self._path(key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return False
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return True
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self._path(key))
        return True


def load_engine():
    import server

    root = Path(os.environ.get("MODELS_DIR", "/models")) / MODEL_VERSION
    if not (root / "whisper-ct2" / "model.bin").exists():
        raise SystemExit(f"no weights at {root}/whisper-ct2 (MODEL_VERSION={MODEL_VERSION})")
    return server.Engine(str(root / "whisper-ct2"), device=os.environ.get("WHISPERLAH_DEVICE", "cuda"),
                         compute=os.environ.get("WHISPERLAH_COMPUTE", "int8_float16"),
                         batch_size=16, diarize=str(root / "pyannote"), hf_token=None,
                         dia_batch=int(os.environ.get("WHISPERLAH_DIA_BATCH", "32")))


def create_app(engine=None, store=None):
    store = store or FileStore(os.environ.get("ASR_STATE_DIR", "/tmp/asr-jobs"))
    token = os.environ.get("ASR_API_TOKEN", "")
    secret = os.environ.get("ASR_CALLBACK_SECRET", "")
    engine = engine or load_engine()

    def get_stream(url):
        return httpx.stream("GET", url, timeout=httpx.Timeout(30, read=300), follow_redirects=True)

    def post(url, body, headers):
        return httpx.post(url, content=body, headers=headers, timeout=30).status_code

    def run(job):
        return J.run_job(job, store=store, transcribe=engine.transcribe, get_stream=get_stream,
                         post=post, secret=secret, max_attempts=MAX_ATTEMPTS)

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")  # 2 jobs per GPU

    def run_with_retries(job):
        for attempt in range(MAX_ATTEMPTS):
            try:
                return run(job)
            except Exception:
                time.sleep(10 * 2 ** attempt)

    if os.environ.get("ASR_QUEUE", "inline") == "inline":
        enqueue = lambda job: pool.submit(run_with_retries, job)  # noqa: E731
    else:
        enqueue = lambda job: None  # noqa: E731  the host's queue calls /run

    app = J.make_api(store, enqueue, token, MODEL_VERSION)
    auth = J.bearer_auth(token)

    @app.post("/run", dependencies=[Depends(auth)])
    def run_now(req: J.JobRequest):
        job = store.get(req.job_id) or {**req.model_dump(), "status": "queued", "attempts": 0,
                                        "model_version": MODEL_VERSION}
        try:
            return J.public(run(job))
        except Exception as e:
            raise HTTPException(503, f"transient failure, retry: {type(e).__name__}")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.post("/invocations")
    async def invocations(request: Request):
        # SageMaker has already authenticated the caller; the body is the job.
        req = J.JobRequest(**json.loads(await request.body()))
        job = {**req.model_dump(), "status": "queued", "attempts": 0, "model_version": MODEL_VERSION}
        store.put(req.job_id, job, skip_if_exists=True)
        from starlette.concurrency import run_in_threadpool
        return JSONResponse(J.public(await run_in_threadpool(run, store.get(req.job_id))))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), workers=1)
