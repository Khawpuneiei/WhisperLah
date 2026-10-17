# WhisperLah: Singlish speech recognition

A LoRA adapter for Whisper large-v3-turbo, trained on open Singlish speech, and
a contamination-checked benchmark harness to measure it. Built for Rapport's
transcription step, measured on Singlish conversation and adapted to it
with a LoRA on Whisper large-v3-turbo. Everything here runs on one 8 GB laptop
GPU (RTX 4060) in under a day: training took 97 minutes of GPU time.

## Results

**MNSC Part 3 test**: 1,000 held-out two-speaker conversation clips, 7.7 h (primary)

| Model | WER ↓ | Singlish particles caught ↑ |
|---|---|---|
| Whisper large-v3-turbo (no adaptation) | 27.7% | 10% |
| Public Singlish fine-tune (`mjwong/whisper-large-v3-turbo-singlish`, full fine-tune) | 26.2% | 6% |
| **Rapport LoRA (ours)** | **19.5%** | **74%** |

8.2 points (30%) fewer word errors than the base model and 6.7 points (25%) fewer than the public
Singlish model, on the kind of audio Rapport records: two people talking.

**SASRBench-v1**: 1,101 short single-speaker clips, ≈4.5 h (secondary)

| Model | WER ↓ | Singlish particles caught ↑ |
|---|---|---|
| Whisper large-v3-turbo | 24.1% | 6% |
| Public Singlish fine-tune | **17.1%** | 9% |
| Rapport LoRA (ours) | 22.4% | **69%** |

Ours beats the base model here but not the public fine-tune, which was
trained on exactly this format (≤20 s single-speaker clips). Our adapter only
ever saw 30 s two-speaker windows and sometimes keeps talking past the end of a
short clip (insertions: 4,976 vs 2,039 for the base). Mixing short crops into
training is the fix; see *Limits*.

Results: `results/table.json`, chart: `results/singlish_asr_chart.png`,
per-clip transcripts: `results/<set>__<model>.jsonl`.

## Use it

```python
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo", torch_dtype=torch.float16).to("cuda")
model = PeftModel.from_pretrained(model, "models/whisperlah-v1-lora").merge_and_unload()

feats = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
ids = model.generate(feats.to("cuda", torch.float16), language="en", task="transcribe",
                     temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

The adapter (`models/whisperlah-v1-lora`, 56 MB, fp16) needs the base model
`openai/whisper-large-v3-turbo` (MIT). Clips up to 30 s.

## Test sets

| Set | What it is | Size used | Licence |
|---|---|---|---|
| **MNSC Part 3 test** (primary) | Held-out, human-verified split of IMDA's National Speech Corpus Part 3: two Singaporeans in free conversation, ≤30 s clips, often both speakers in one clip. Released by A*STAR's MERaLiON team with standardised splits. | all 1,000 clips, 7.7 h | as published on Hugging Face (`MERaLiON/Multitask-National-Speech-Corpus-v1`), derived from IMDA NSC |
| **SASRBench-v1** (secondary) | Public Singlish ASR benchmark, NSC Part 3 close-talk, 4–20 s single-speaker clips. | fixed random 1,200 of 3,747 clips, minus overlap (below) | MIT (`mjwong/SASRBench-v1`) |

**Contamination check.** Both sets are cut from NSC Part 3, so clips in
SASRBench may also be in our training slice. `prep_train.py` measures 8-gram
overlap between every SASRBench reference and the training transcripts; 274 of
3,747 clips overlap ≥30% and are excluded from *every* model's SASRBench score.
MNSC train/test are separate published splits.

## Scoring

WER with `jiwer` after one normaliser applied identically to reference and
hypothesis for every model (`scripts/common.py`):

1. NSC markup removed: `<Speaker1>:` tags, `(uh) (ppl)` filler/paralinguistic
   tags, `**` unintelligible, `#name#` markers, `~` truncation marks.
2. Singlish particles kept: `[lah]` → `lah`. They are real words and the thing
   generic models drop; normalising them away would hide the gap.
3. Whisper's `EnglishTextNormalizer` (numbers, spelling, contractions, fillers),
   spelled acronyms joined (`n_t_u`, `u s` → `ntu`, `us`), `yeah/yah` → `ya`, `ok` → `okay`.

Numbers are therefore comparable across the models in this table, not with
leaderboards that use a different normaliser.

**Decoding.** Every model, both test sets: greedy, with Whisper's standard
temperature fallback when an output's compression ratio exceeds 1.35 (the sign
of a repetition loop, e.g. "ya, ya, ya, …"). The guard helps every model, not
just ours: versus plain greedy, MNSC moves Whisper turbo 29.9% → 27.7% and the
public fine-tune 29.3% → 26.2%; SASRBench moves Whisper turbo 26.0% → 24.1%
and ours 28.9% → 22.4%. Plain-greedy runs are kept in `results/greedy/`.

**Particle recall**: of the discourse particles in the reference (lah, leh,
lor, meh, sia, hor, mah, liao, …), the share that also appear in the
hypothesis.

## Training

| | |
|---|---|
| Base | `openai/whisper-large-v3-turbo` (809 M params), frozen, bf16 |
| Adapter | LoRA r=32, α=64, dropout 0.05 on q/k/v/out and fc1/fc2 in **encoder and decoder**: 27.9 M trainable (3.3%) |
| Data | 12 of 171 MNSC Part 3 train shards, chosen evenly across the corpus for speaker spread: 6,754 clips, 51.9 h |
| Targets | NSC markup cleaned as above, casing and punctuation kept, particles kept |
| Optimiser | AdamW, lr 3e-4, cosine, warm-up 25 steps, batch 16 (4 × 4 accumulation), 500 steps ≈ 1.2 epochs |
| Loss | 1.11 → 0.61 |
| Cost | 97 min on one RTX 4060 laptop GPU, 3.1 GB peak with gradient checkpointing |

The encoder is adapted, not just the decoder: the gap on Singlish is mainly
acoustic (accent, particles, fast overlapping turns), not vocabulary.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/Scripts/python.exe transformers==4.46.3 peft==0.13.2 datasets==3.1.0 accelerate==1.1.1 jiwer soundfile librosa huggingface_hub==0.26.2 openai pyarrow pandas tqdm
python scripts/download.py
cd scripts
python prep_train.py
python train_lora.py --steps 500 --save_every 125 --name lora-r32
python eval_hf.py --model openai/whisper-large-v3-turbo --adapter ../runs/lora-r32/final --set mnsc_part3 --name lora-r32
python eval_hf.py --model openai/whisper-large-v3-turbo --adapter ../runs/lora-r32/final --set sasrbench --limit 1200 --name lora-r32
python report.py && python chart.py
```

## Limits, stated plainly

- One training run, one seed, ~1.2 epochs over 12 of 171 Part 3 training
  shards (7%). More data and steps are the obvious next lever.
- Short clips: trained only on full 30 s windows, the adapter over-generates
  on 4–20 s audio (see SASRBench). Next run: add random short crops and NSC
  Parts 1–2 read speech to the mix.
- Not yet compared with the production API: the OpenAI key was revoked during
  this run. `scripts/eval_openai.py` is ready for a new key.
- Whisper does not label speakers. In the product, the adapted model would
  replace only the transcription step; speaker attribution still needs a
  diarizer (pyannote or the current API).
- MNSC Part 3 is close-talk, studio-quiet conversation. Noisy cafés are not
  covered by this benchmark; that needs our own recordings.

## Data and attribution

Training and test audio come from IMDA's National Speech Corpus, through
A*STAR MERaLiON's Multitask National Speech Corpus release on Hugging Face.
Contains information from the National Speech Corpus accessed from IMDA, made
available under the terms of the Singapore Open Data Licence version 1.0.
SASRBench-v1 (`mjwong/SASRBench-v1`) is MIT-licensed. The per-clip result files
include these reference transcripts so every number here can be re-scored.
No audio is redistributed in this repository.
