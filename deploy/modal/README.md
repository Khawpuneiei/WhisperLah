# Lepaq ASR on Modal

Whole-file transcription jobs for the Lepaq webapp: WhisperLah (CTranslate2
int8) + pyannote community-1 + WeSpeaker ResNet34 embeddings, on an L4,
scaled to zero when idle.

```
Lepaq /process ── POST /jobs (bearer) ──> api (CPU) ──> modal.Dict job state
                                            │
                                            └─ Worker.process.spawn ──> L4 worker
                                               (2 retries)               0-1 containers (testing), 2 jobs each
                                                                         │
Lepaq /api/transcription/callback <── HMAC-signed POST ─────────────────┘
Lepaq cron ── GET /jobs/{id} (bearer) ──> job state, for jobs with no callback after 15 min
```

The contract (request, status and callback bodies, signature) is at the top
of `jobs.py`; the webapp side builds to it on lepaq#2.

## Files

| File | What |
|---|---|
| `app.py` | Modal wiring: images, Volume, Dict, worker class, web endpoint |
| `../../scripts/serve/jobs.py` | API, worker step, download, signed callback (shared with deploy/container) |
| `../../scripts/serve/test_jobs.py` | unit tests, CPU only |

The worker runs `scripts/serve/server.py`'s `Engine`, the same code measured
on the 4060, with the token-ratio fallback from `fallback.py`.

## Weights: private Volume, one folder per version

Nothing in this repo or the image contains weights. They go in the private
Volume `lepaq-asr-models`:

```
/v3/
  whisper-ct2/        model.bin, config.json, tokenizer.json, preprocessor_config.json, vocabulary.json
  pyannote/           a local copy of pyannote/speaker-diarization-community-1:
                      config.yaml + segmentation/, embedding/ (WeSpeaker ResNet34), plda/
```

`MODEL_VERSION` picks the folder at deploy time; rollback is redeploying with
the previous value. `HF_HUB_OFFLINE=1`, so a missing file fails at start
instead of reaching for the Hub. The fine-tuned segmentation replaces
`pyannote/segmentation/` only after it beats stock community-1 on a gold
meeting; until then the folder holds the stock checkpoint.

Uploading is a manual step and each upload needs P's explicit OK (project
rule: weights never leave the laptop without it):

```
modal volume create lepaq-asr-models
modal volume put lepaq-asr-models models/whisperlah-v3-final-ct2 /v3/whisper-ct2
modal volume put lepaq-asr-models <local community-1 copy> /v3/pyannote
```

## Deploy

```
modal secret create lepaq-asr ASR_API_TOKEN=<random> ASR_CALLBACK_SECRET=<random>
MODEL_VERSION=v3 modal deploy deploy/modal/app.py
```

Give the webapp `ASR_URL` (the `api` endpoint URL Modal prints),
`ASR_API_TOKEN` and `ASR_CALLBACK_SECRET`.

## Testing budget (to 15 Nov 2026)

Default region, `min_containers=0`, `scaledown_window=60`, at most one L4
(`ASR_MAX_GPUS=1`). One hour of audio is about 3 min of L4 time plus a cold
start. Set a spending limit in the Modal dashboard to stay inside the
$30/month free credit.

## Behaviour

- Re-POSTing a `job_id` returns its status; it is never queued twice.
- A 4xx on `audio_url`, or audio over 1 GB, fails the job at once. Other
  errors (GPU, network) are retried twice; the callback goes out when the job
  is done or has failed for good.
- The callback is retried after 1, 4 and 16 s on a 5xx or network error and
  stops on a 4xx (unknown job, bad signature). After that, the webapp's cron
  reads `GET /jobs/{id}`.
- `num_speakers` (from the meeting's participant list) is passed to pyannote;
  on the synthetic test set, knowing the head count removed about half the
  speaker mix-ups.
- Measured on the 4060: ~3 min per hour of audio once warm. A cold start from
  zero containers adds the image pull and model load (not yet measured on
  Modal).
