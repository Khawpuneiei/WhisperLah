# Code only: weights are mounted at $MODELS_DIR/$MODEL_VERSION at run time
# (Cloud Run volume, SageMaker model data, or a local folder). Never COPY models/.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv curl ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/0.11.32/install.sh | sh
ENV PATH=/root/.local/bin:/opt/venv/bin:$PATH
RUN uv venv --python python3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
         torch==2.11.0 torchaudio==2.11.0 \
    && uv pip install --python /opt/venv/bin/python \
         pyannote.audio==4.0.7 faster-whisper==1.2.1 ctranslate2==4.8.3 "av<16" \
         fastapi uvicorn httpx python-multipart

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    MODELS_DIR=/models MODEL_VERSION=v3 WHISPERLAH_CODE=/app PORT=8080
WORKDIR /app
COPY scripts/serve/server.py scripts/serve/fallback.py scripts/serve/jobs.py /app/
COPY deploy/container/worker.py /app/
EXPOSE 8080
# SageMaker runs the image with the argument "serve"; it is ignored here.
ENTRYPOINT ["python", "/app/worker.py"]
