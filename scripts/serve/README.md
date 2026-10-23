# Serving WhisperLah

Self-hosted transcription for Rapport/Lepaq: the WhisperLah v2 adapter merged
into `whisper-large-v3-turbo`, run with CTranslate2 through faster-whisper,
plus pyannote speaker diarization, behind a small HTTP server that returns the
speaker-labelled segments Rapport's `lib/ai/transcribe.ts` expects.

The merged model and the adapter are proprietary and stay on the serving
machine (`models/` is gitignored).

## Files

| File | What |
|---|---|
| `convert_ct2.py` | merge adapter into the base model, convert to CTranslate2 |
| `eval_ct2.py` | WER + speed of the converted model on the benchmark sets (same table as `eval_hf.py`) |
| `server.py` | FastAPI server: `POST /transcribe`, `GET /health` |
| `fallback.py` | Whisper's temperature fallback with the same compression-ratio rule as `eval_hf.py` |
| `bench.py` | time a recording through the server and the OpenAI API, Rapport-style (600 s parts, 3 in flight) |
| `test_assign.py`, `test_fallback.py` | unit tests (CPU, no model) |

## Setup

ASR only (fits in the training `.venv`):

```bash
uv pip install --python .venv/Scripts/python.exe -r scripts/serve/requirements.txt
```

With diarization (pyannote community-1 needs pyannote.audio 4.x, which may
move torch, so use a separate venv):

```bash
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve/Scripts/python.exe -r scripts/serve/requirements.txt "pyannote.audio>=4.0,<5"
```

pyannote needs a Hugging Face token with
`pyannote/speaker-diarization-community-1` accepted on the Hub (the
`WHISPERLAH_DIARIZE=3.1` fallback needs `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`). Put it in `HF_TOKEN`. Without it the server runs
ASR only, returns Whisper's own sentence segments, and labels everything `A`.

Chosen stack (21 Oct): WhisperLah v2 for ASR, pyannote community-1 for
diarization (segmentation fine-tune later), pyannote's built-in WeSpeaker
ResNet34 centroids as the speaker embedding, no fine-tune. The embedding
comes free with diarization (`return_embeddings=True`); the response names
the model with each vector so a later embedder swap stays traceable.

## 1. Convert

```bash
python scripts/serve/convert_ct2.py --adapter models/whisperlah-v2-lora --out models/whisperlah-v2-ct2
```

One fp16 conversion serves fp16 and int8 inference (`compute_type` is chosen
at load time).

## 2. Check the conversion did not move WER

```bash
python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --set mnsc_part3 --limit 300 --name ct2-fp16
python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --set mnsc_part3 --limit 300 --name ct2-int8 --compute int8_float16
python scripts/serve/eval_ct2.py --model models/whisperlah-v2-ct2 --set sasrbench --limit 300 --name ct2-fp16-near --prompt "Near speaker only."
```

Gate: within 0.3 WER points of the HF numbers on the same 300 clips
(`eval_hf.py --limit 300` draws the same clips, seed 0). Add `--batched` to
score exactly what the server does (batched greedy, then loops re-decoded).

Fallback: transformers measures the compression ratio on token ids, where
1.35 flags only real repetition loops; faster-whisper measures it on text,
where 1.35 flags almost every clip, and its batched pipeline never falls back.
`fallback.py` puts the token measure into faster-whisper and re-decodes only
the looping segments, so both scripts and the server use eval_hf's rule. The
row's `clips_redecoded` says how many clips needed it.

## 3. Serve

```bash
WHISPERLAH_MODEL=models/whisperlah-v2-ct2 WHISPERLAH_COMPUTE=int8_float16 HF_TOKEN=hf_... \
  python scripts/serve/server.py --port 8765
curl -F file=@meeting.mp3 -F mode=room http://127.0.0.1:8765/transcribe
```

Response:

```json
{"duration": 612.4, "language": "en", "mode": "room", "diarized": true,
 "segments": [{"speaker": "A", "start": 0.52, "end": 4.10, "text": "So how was the trip lah?"}, ...],
 "speakers": {"A": {"embedding": [0.0123, ...], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"}, "B": {...}},
 "redecoded": 0,
 "timing": {"decode_s": 0.4, "asr_s": 9.8, "diarize_s": 6.1, "total_s": 10.3, "wall_s": 10.5}, "rtf": 0.0168}
```

Speakers are `A`, `B`, … in order of first appearance within the request;
`speakers[X].embedding` is that speaker's 256-dim centroid, for matching the
same person across parts and across meetings. `redecoded` counts segments
that came out of the batched decode as a repetition loop and were decoded
again with temperature fallback.

Requests run off the event loop, and a request's diarization starts in its
own thread while its ASR runs, so `total_s` is close to the slower of the two,
not their sum. ASR and diarization hold separate locks, so with several parts
in flight one part's ASR runs while another's diarization is clustering.

`mode=near` sends the `Near speaker only.` prompt (lapel or headset mic).
Set `WHISPERLAH_API_KEY` to require a bearer token. `WHISPERLAH_DIA_BATCH`
(default 32) sets pyannote's segmentation and embedding batch sizes; 64 fits
next to the int8 Whisper on 8 GB.

## 4. Benchmark against the API

```bash
python scripts/serve/bench.py path/to/meeting.m4a --openai --env-file ../rapport/.env.local --label 4060-int8
```

Appends one row per backend to `results/bench.jsonl`: wall-clock, RTF,
segments, US$ per audio hour (OpenAI: $0.36; local: `--gpu-usd-per-hour` x
wall-clock, 0 on your own laptop).

## 5. Point Rapport at it

In `rapport/.env.local`:

```
TRANSCRIBE_BACKEND=whisperlah
WHISPERLAH_URL=http://127.0.0.1:8765
WHISPERLAH_MODE=room
```

Everything downstream (speaker mapping, analysis, review) is unchanged.
