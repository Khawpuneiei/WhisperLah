# KopiChat: a coffee-chat benchmark for Singlish speech recognition

No public test set covers the situation Rapport is built for: two people
talking in a café with a phone on the table. KopiChat is that test set.
It is small, real, and scored with the same harness as every other benchmark
in this repo, so every model and API gets the same clips and the same rules.

## What to record

| | |
|---|---|
| Chats | 20–30, two speakers each, both Singaporean English speakers |
| Length | 2–4 minutes of natural conversation per chat (aim for ~1.5 h total) |
| Where | real cafés, kopitiams, food courts, office pantries: background chatter, music, cutlery, espresso machine |
| Device | a phone lying flat on the table, 30–60 cm from both speakers, normal voice memo app, no earbuds, no external mic |
| Format | whatever the phone records (m4a is fine); do not convert or denoise |
| Topic | anything ordinary: weekend plans, work, food, football. Code-switching and particles are welcome, they are the point |
| Avoid | health, religion, politics, money problems, family problems, anyone's full name or address, third parties who did not consent |

Spread the chats across different cafés, times of day and speaker pairs.
Ten chats in one café with the same two people would measure that café, not
the scenario.

Also record **café ambience without speech**: 15–20 minutes total, several
cafés, phone on the table, nobody at the table talking. This is used to make
noisy training data; it is never part of the test set.

## Consent

Both speakers sign before recording. Keep the signed form with the recording's
filename written on it. Suggested wording:

> I agree that this conversation may be recorded and transcribed, and that the
> recording and transcript may be published as part of an open speech
> recognition benchmark under the Singapore Open Data Licence or a similar
> open licence. I understand I can ask for it to be withdrawn at any time before
> publication, and that I should not say anything I would not want public.

Anyone at a neighbouring table whose words come out clearly in the transcript
is not a consenting participant: cut that stretch of audio.

## Transcription

1. Run WhisperLah on the recording (room mode) for a first draft.
2. A team member corrects every line while listening. Rules follow the
   National Speech Corpus so the existing normaliser applies unchanged:
   - keep every particle: `lah`, `leh`, `lor`, `meh`, `sia`, `hor`, `mah`, `liao`, `ah`, `eh`
   - write fillers as `(uh)`, `(um)`; unintelligible stretches as `**`
   - mark speaker turns as `<Speaker1>:` and `<Speaker2>:`
   - numbers as words as spoken ("thirty five dollar")
   - no punctuation needed; casing does not matter
3. **Spot check**: two different people independently correct the same three
   chats. The harness reports the word error rate between their versions; if
   it is above 5%, the rules are being applied differently and the whole set
   is re-checked before use.

## Packaging

```
data/kopichat/
  audio/chat01.m4a ...          original recordings
  transcripts/chat01.txt ...    corrected transcripts, one turn per line
  consent/                      scanned forms (not committed)
python scripts/kopichat/make_kopichat.py
```

`make_kopichat.py` cuts each recording at turn boundaries into clips of at
most 30 s and writes `data/kopichat/test.parquet` in the same
`(audio bytes, text)` schema as the other test sets. From there,
`eval_hf.py --set kopichat`, `eval_openai.py --set kopichat` and `report.py`
work without changes.

Audio and transcripts are committed to the repo only after every consent form
is in hand.
