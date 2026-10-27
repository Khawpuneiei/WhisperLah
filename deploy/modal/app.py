"""Lepaq ASR on Modal: whole-file jobs on an L4, weights from a private Volume.

    POST /jobs -> modal.Dict (job state) -> Worker.process.spawn (queue, 2 retries)
      -> L4 worker (0-3 containers, 2 jobs each) -> signed callback to Lepaq

Deploy (needs P's Modal account; nothing here uploads weights):

    modal secret create lepaq-asr ASR_API_TOKEN=... ASR_CALLBACK_SECRET=...
    MODEL_VERSION=v3 modal deploy deploy/modal/app.py

Weights live in the private Volume `lepaq-asr-models`, one folder per version
(see README.md for the layout). Rolling back is redeploying with the previous
MODEL_VERSION. The image holds code and pinned libraries only.
"""
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SERVE = HERE.parents[1] / "scripts" / "serve"

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3")
MODELS = "/models"
MAX_RETRIES = 2
JOBS_PER_CONTAINER = 2

app = modal.App("lepaq-asr")
volume = modal.Volume.from_name("lepaq-asr-models")  # created and filled by hand, never by deploy
jobs = modal.Dict.from_name("lepaq-asr-jobs", create_if_missing=True)
secret = modal.Secret.from_name("lepaq-asr")  # ASR_API_TOKEN, ASR_CALLBACK_SECRET

# Same stack as measured on the 4060 (results 23 Oct 2026).
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0",
                    index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("pyannote.audio==4.0.7", "faster-whisper==1.2.1", "ctranslate2==4.8.3",
                    "av<16", "fastapi", "httpx", "python-multipart")
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MODEL_VERSION": MODEL_VERSION})
    .add_local_file(SERVE / "server.py", "/root/server.py")
    .add_local_file(SERVE / "fallback.py", "/root/fallback.py")
    .add_local_file(HERE / "jobs.py", "/root/jobs.py")
)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "pydantic>=2")
    .env({"MODEL_VERSION": MODEL_VERSION})
    .add_local_file(HERE / "jobs.py", "/root/jobs.py")
)


@app.cls(image=gpu_image, gpu="L4", volumes={MODELS: volume}, secrets=[secret],
         min_containers=0, max_containers=3, scaledown_window=300, timeout=60 * 60,
         retries=modal.Retries(max_retries=MAX_RETRIES, initial_delay=10.0))
@modal.concurrent(max_inputs=JOBS_PER_CONTAINER)
class Worker:
    @modal.enter()
    def load(self):
        import server

        root = Path(MODELS) / MODEL_VERSION
        if not (root / "whisper-ct2" / "model.bin").exists():
            raise RuntimeError(f"no weights for MODEL_VERSION={MODEL_VERSION} in the volume")
        self.engine = server.Engine(
            str(root / "whisper-ct2"), device="cuda", compute="int8_float16",
            batch_size=16, diarize=str(root / "pyannote"), hf_token=None, dia_batch=32)

    @modal.method()
    def process(self, job: dict):
        import httpx

        import jobs as J

        def get_stream(url):
            return httpx.stream("GET", url, timeout=httpx.Timeout(30, read=300),
                                follow_redirects=True)

        def post(url, body, headers):
            return httpx.post(url, content=body, headers=headers, timeout=30).status_code

        J.run_job(job, store=jobs, transcribe=self.engine.transcribe,
                  get_stream=get_stream, post=post,
                  secret=os.environ["ASR_CALLBACK_SECRET"], max_attempts=MAX_RETRIES + 1)


@app.function(image=web_image, secrets=[secret], min_containers=0)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    import jobs as J

    return J.make_api(jobs, enqueue=lambda job: Worker().process.spawn(job),
                      token=os.environ["ASR_API_TOKEN"], model_version=MODEL_VERSION)
