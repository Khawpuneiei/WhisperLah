# WhisperLah: Singlish speech recognition

WhisperLah is a Singlish speech recognition model built for Lepaq. This repo
is its benchmark showcase: the contamination-checked evaluation harness and the
per-clip results used to measure WhisperLah against every Singlish model we
could run, including A*STAR's MERaLiON-2. **The model weights are proprietary
and are not published here.** How WhisperLah is built and trained will be
described in a technical report.

<!--HEADLINE-->

![Singlish accuracy index: WhisperLah v3 ranks first of 16 models on average word accuracy across all 7 test sets](results/whisperlah_ranked_chart.png)

Every model we have scored on both public Singlish benchmarks, with each
WhisperLah release (`scripts/chart_sota.py`):

![WhisperLah against other Singlish models on SASRBench and NSC Part 3](results/whisperlah_sota_chart.png)

## Results

Set by set, WhisperLah v3 against its five closest rivals and its base model
(`scripts/chart_grouped.py`):

![WhisperLah v3 and its closest rivals on each Singlish test set](results/whisperlah_grouped_chart.png)

Word error rate, %, lower is better. Every model is scored on the same clips
with the same text normaliser. Bold is the best score on each test set.

<!--RESULTS_TABLE-->

TBD means not tested yet; those cells are left for a later round. So far
the English sets (AMI, LibriSpeech) are regression checks, run only for base
Whisper and WhisperLah. Near-mic mode and WhisperLah v1 are scored on the two
public benchmarks only. Whisper-1 and MERaLiON-2-10B-ASR were also run on
those two only, to save API cost and because the 10B runs at up to 4x real
time on the 8 GB GPU. Canary 1B v2's Parts 4 to 6 were stopped (see Other
notes).

**OpenAI API settings.** Each OpenAI model is shown at the setting that
scored better for it. GPT Transcribe and GPT-4o Transcribe use
`chunking_strategy="auto"`: without it GPT-4o Transcribe stops early and drops
most of each clip (58.4% on SASRBench), and even with it, deletions make up
most of its errors. GPT-4o mini Transcribe is shown without chunking, which is
better for it on every set (with chunking: 26.8% on SASRBench). Whisper-1 gets
`language="en"`, as every local Whisper run does; left to auto-detect, it
translated the Singlish into Malay (97% on SASRBench). Whisper-1 was scored on
the two public benchmarks only. The other runs are kept in `results/` as
`-nochunk` and `-autolang`. GPT-4o Transcribe diarize, like WhisperLah v1,
writes down the background speaker on SASRBench, which that benchmark counts
as insertions. OpenAI API cost for all of this: about US$48.

**Other notes.** Canary 1B v2 and Granite Speech 3.3 2B fall into repetition
loops on long conversation clips (Canary writes up to 1,000 words for one
clip). That is most of their NSC Part 3 error (58.5% and 74.5%) and Granite's
129.8% on Part 4; Canary's Parts 4 to 6 were stopped for that reason.
Qwen2.5-Omni 3B is a chat model and often answers the speaker instead of
transcribing them ("I have never received or shared a lobang so I can't
really answer that question", on NSC Part 3). It does this on about one in
seven of NSC Part 2's short read questions (95.8%), and it loops on the
Mandarin code-switched clips of Part 4 (132.4%). Its outputs are scored as
they are, like every other model's.
SenseVoice Small's 10.4% on NSC Part 2 is far below every other model except
MERaLiON-2-3B, which suggests it, too, was trained on NSC read speech (see
Benchmark protocol). † MERaLiON-2-10B-ASR is scored on a fixed 300-clip
sample of NSC Part 3 (299 after the contamination filter): in 4-bit on the
8 GB GPU it runs at up to 4x real time. On those same clips WhisperLah v3
scores 18.8%, against its 22.4%.

Full numbers, including substitutions/deletions/insertions per model:
`results/table.json`. Every hypothesis for every clip: `results/<set>__<model>.jsonl`.

## Where WhisperLah v3 is best, and where it is not

Lowest word error rate of every model we ran on the same clips (release adapter v3, best of room and near-mic mode; v2 shown for reference):

| Test set | WhisperLah v3 | v2 | Best other model | Result |
|---|---|---|---|---|
| SASRBench-v1, close-talk | **14.8** | 15.1 | 15.8 (GPT Transcribe) | win |
| NSC Part 3, two-person conversation | **17.9** | 18.9 | 25.2 (Distil-Whisper large-v3.5)† | win |
| NSC Part 4, conversation | **27.8** | 29.5 | 34.3 (GPT Transcribe) | win |
| NSC Part 5, conversation | 14.3 | 14.3 | 16.3 (Parakeet TDT 0.6B v3) | win (v2.1's 13.7 is lower still) |
| NSC Part 6, conversation | **10.7** | 11.1 | 13.5 (Parakeet TDT 0.6B v3) | win |
| AMI meetings, headset mic (English) | **9.2** | 11.3 | 14.9 (Whisper large-v3-turbo) | win, vs Whisper only |
| AMI meetings, table mic (English) | **24.9** | 33.0 | 34.1 (Whisper large-v3-turbo) | win, vs Whisper only |
| NSC Part 1, read speech | 6.0 | 6.2 | 4.8 (MERaLiON-2-3B) | behind |
| NSC Part 2, read speech, local names | 21.7 | 23.6 | 6.6 (MERaLiON-2-3B) | behind |
| LibriSpeech clean (English) | 1.9 | 2.2 | 1.9 (Whisper large-v3-turbo) | level with base |
| LibriSpeech other (English) | 4.5 | 4.7 | 3.9 (Whisper large-v3-turbo) | 0.6 pt behind base |

† MERaLiON-2-10B-ASR was run on a 300-clip sample of Part 3 only: 22.4 there,
against 18.8 for v3 on the same clips.

Five of the seven Singlish test sets, including both public benchmarks, are
won outright by a WhisperLah model, against every open model and OpenAI API
we could run, and v3 improves on v2 on six of the seven (Part 5 is flat). The
two losses are read-speech sets whose prompt sentences recur in the training
data (see Benchmark protocol), so we are not chasing them. No single outside
model comes closest everywhere: OpenAI's GPT Transcribe is nearest on
SASRBench (1.0 pt behind v3's near-mic mode) and Part 4 (6.5 pt), NVIDIA's
Parakeet on Parts 5 and 6 (2.0 and 2.7 pt), and MERaLiON-2-10B-ASR on Part 3
(3.6 pt, on its 300-clip sample). The largest
v3 gain is on far-field English meetings: AMI table mic falls from 33.0 to
24.9, from v3's AMI training data. On English meetings, only base Whisper has
been measured so far; other English models have not been run, so those rows
do not claim a leaderboard position.

**Release gate.** Set before scoring: v3 must improve NSC Part 3, be no more
than 0.5 pt worse than v2 on any set, and stay within 0.5 pt of base Whisper
on LibriSpeech. v3 passes every Singlish and AMI condition and LibriSpeech
clean, and misses LibriSpeech other by 0.1 pt (4.53 against a 4.41 limit),
while still improving on v2's 4.68 there. We release v3 as the default and
report the miss rather than move the gate.

## Next benchmarks to beat

Chosen by what a meeting or coffee-chat product needs, and by what one 8 GB
laptop GPU can reach (full reasoning in the repo's planning notes).

| Target | Today | Aim | Status |
|---|---|---|---|
| **KopiChat**, our own café-chat set (phone on a table, two speakers; recording protocol in `scripts/kopichat/README.md`) | not recorded yet | lowest of every model and the OpenAI API | recordings and consent forms needed |
| **NSC Parts 3 to 6**, further | 17.9 / 27.8 / 14.3 / 10.7 (v3) | lower again, especially Part 5 | v3 done: 3 of 4 improved, Part 5 flat |
| **AMI table-mic meetings** (far-field) | 24.9 (v3) | well below Whisper's 34.1 | done with v3: 9.2 pt below Whisper |
| **SASRBench and NSC Part 3 vs MERaLiON-2-10B** | SASRBench 14.8 (v3 near-mic) vs 16.7; Part 3 18.8 vs 22.4 on 300 clips | confirm the win against the 10B | done in 4-bit (about 23 GPU-hours on 8 GB); Part 3 on a 300-clip sample |
| **English regression gate** (LibriSpeech) | clean level with base, other 0.6 pt behind (v3) | within 0.5 pt of base | clean met; other missed by 0.1 pt |
| OpenAI and other closed APIs | all four OpenAI transcription models on all 7 Singlish sets (Whisper-1 on 2) | KopiChat; other vendors | OpenAI done: best is GPT Transcribe, behind v3 on 6 of 7 sets |
| Other open ASR models | Parakeet, Canary, Voxtral, Granite, SenseVoice, Qwen2.5-Omni, Distil-Whisper | ahead of every open model on conversation | done: v3 ahead on all five conversational sets; best new model is SenseVoice Small |

Not targeted: NSC Parts 1 and 2 (read speech), and large English leaderboards
(Earnings-22, GigaSpeech, and similar). Those are decided by 10B-parameter
models trained on 100k hours, and they do not match the product.

## Two modes: room and near-mic

The two public Singlish benchmarks disagree about *whose* words count.

- **NSC conversation tests** label everyone in the room.
- **SASRBench** labels only the person wearing the close-talk mic, even
  though the other speaker is audible in the recording.

A model that writes down the background speaker is right for meeting notes and
"wrong" for SASRBench. WhisperLah v1 did exactly that: its SASRBench errors
were mostly the other person's real words, scored as insertions. The public
Singlish fine-tune learned the opposite convention and drops the second speaker
on NSC conversation.

WhisperLah v2 and v3 learn both, switched by a Whisper text prompt:

| Mode | How | Use it for |
|---|---|---|
| **Room** (default) | no prompt | meetings: transcribe everyone |
| **Near-mic** | prompt `Near speaker only.` | a lapel or headset mic: transcribe the wearer |

Both modes are reported on every test, so the choice is
visible rather than tuned per benchmark.

## Benchmark protocol

| Test set | What it is | Clips used |
|---|---|---|
| SASRBench-v1 (`mjwong/SASRBench-v1`, MIT) | NSC Part 3 close-talk, single speaker, 4–20 s | fixed random 1,200 of 3,747 |
| NSC Part 3 test (MNSC) | two-person conversation, ≤30 s | all 1,000 |
| NSC Parts 1, 2 test (MNSC) | read speech | fixed random 500 each |
| NSC Parts 4, 5, 6 test (MNSC) | conversation | fixed random 300 each |

The MNSC test splits are the standardised held-out splits released by A*STAR's
MERaLiON team, the same ones AudioBench reports as "IMDA Part 1–6 ASR".

**Contamination.** Any test clip whose reference shares ≥30% of its 8-grams
with *any* WhisperLah training transcript is excluded from *every* model's
score (lists in `results/*_overlap_with_train.json`).
Removed: SASRBench 145 of the 1,200 sampled clips, NSC P1 / P2 / P3 / P6 a
handful each. One caveat we cannot remove: SASRBench is cut from NSC Part 3
recordings, and our check found 436 of its 3,747 clips inside just 20 of the
171 MNSC Part 3 training shards. A model trained on the full corpus has
therefore likely seen most of SASRBench's audio. That applies to MERaLiON-2,
which was trained on NSC-derived data at scale, and not to WhisperLah.

The read-speech tests (Parts 1 and 2) have a second, larger version of the
same issue. Each is about 1,000 prompt sentences, each read by three speakers,
and the same prompts recur across the training split: a single one of our
training shards already contains the text of 2.1% (Part 2) to 3.7% (Part 1) of
the test clips. Across the 727–771 training shards of each part, essentially
every test sentence appears in training text. Clips overlapping *our* training
text are excluded, but a model trained on the full split, as MERaLiON-2 likely
was, has read every test sentence. Its lead on Parts 1 and 2 should be read
with that in mind; the conversational tests (Parts 3–6, SASRBench) have no
such recurring prompts.

**Scoring.** `jiwer` WER after one normaliser applied identically to every
reference and hypothesis (`scripts/common.py`): NSC markup removed
(`<Speaker1>:`, `(uh)`, `(ppl)`, `**`, `#…#`, `~`), Singlish particles kept
(`[lah]` → `lah`), then Whisper's `EnglishTextNormalizer`, spelled acronyms
joined, `yeah/yah` → `ya`, `ok` → `okay`. Numbers are comparable within this
table, not with leaderboards that normalise differently.

**Decoding.** Every Whisper-based model: greedy with Whisper's standard
temperature fallback when an output's compression ratio exceeds 1.35, which
catches repetition loops ("ya, ya, ya, …"). MERaLiON-2-3B and -10B-ASR: the
authors' transcription prompt and settings, greedy, up to 320 new tokens; the
10B in 4-bit NF4 (`eval_meralion.py --load-in-4bit`) to fit 8 GB. Parakeet
TDT 0.6B v3 and Canary 1B v2: the ONNX exports through `onnx-asr`
(`eval_onnxasr.py`). Voxtral Mini 3B: 4-bit NF4 through the Transformers
transcription request (`eval_voxtral.py`). Granite Speech 3.3 2B: the model
card's transcription prompt (`eval_granite.py`). SenseVoice Small: `funasr`,
English, inverse text normalisation on, its language, emotion and event tags
removed (`eval_sensevoice.py`). Qwen2.5-Omni 3B: the text-only "thinker" with
its language model in 4-bit NF4 (about 9 GB in bf16), the model card's system
prompt and English transcription instruction, stopping at the end-of-turn
token (`eval_qwen_omni.py`). The chat-style models (Voxtral, Granite,
Qwen2.5-Omni) get up to 500 new tokens. Every model is run in English with its
own default greedy decoding and no prompt tricks; OpenAI settings are above.

## Limits

- v3 is 0.6 pt behind base Whisper on LibriSpeech other, 0.1 pt outside our
  own release gate.
- Whisper does not label speakers. In Lepaq, WhisperLah replaces only the
  transcription step; speaker attribution still needs a diarizer.
- NSC is studio-quiet close-mic audio. Café noise is not covered by any of
  these benchmarks.

## Data and attribution

Training and test audio come from IMDA's National Speech Corpus, through
A*STAR MERaLiON's Multitask National Speech Corpus release on Hugging Face.
Contains information from the National Speech Corpus accessed from IMDA, made
available under the terms of the Singapore Open Data Licence version 1.0.
SASRBench-v1 (`mjwong/SASRBench-v1`) is MIT-licensed. MERaLiON-2-3B and
MERaLiON-2-10B-ASR are used for evaluation only, under the MERaLiON Public
Licence; every other outside model is used for evaluation only, under its own
licence. The per-clip result
files include reference transcripts so every number can be re-scored; no
audio is redistributed here. Company logos in the charts are trademarks of
their owners, used only to show who made each model; their sources are in
`docs/logos/SOURCES.md`.
