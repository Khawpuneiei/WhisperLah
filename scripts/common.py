"""Shared loading and text normalisation for the Singlish ASR benchmark.

Both test sets come from IMDA's National Speech Corpus (NSC) and use its
annotation conventions, which no ASR model will ever output:

    [lah] [leh] [ah]   discourse particles   -> keep the word, drop brackets
    (uh) (um) (ppl)    fillers / paralinguistic tags -> drop
    !huh! !wah!        interjections          -> keep the word
    n_t_u  b_a         spelled-out acronyms   -> ntu  ba
    ** <unk>           unintelligible         -> drop

Particles are kept on purpose. "lah", "leh", "lor" are real Singlish words and
transcribing them is exactly what a generic model gets wrong; normalising them
away would hide the thing we are trying to measure.

After that, Whisper's own EnglishTextNormalizer runs on reference and
hypothesis alike (numbers to digits, British/American spelling, contractions,
filler removal), so the comparison is like-for-like across every model,
including the OpenAI API.
"""
import glob
import io
import json
import re

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].as_posix()

TEST_SETS = {
    # standardised, human-verified held-out split from MERaLiON (A*STAR)
    "mnsc_part3": (f"{ROOT}/data/Multitask-National-Speech-Corpus-v1/ASR-PART3-Test/*.parquet", "context", "answer"),
    # MIT-licensed Singlish benchmark, NSC Part 3 close-talk, 4-20 s clips
    "sasrbench": (f"{ROOT}/data/SASRBench-v1/data/test-*.parquet", "audio", "transcript"),
}
# the rest of IMDA NSC via MNSC's held-out splits (AudioBench's "IMDA Part 1-6 ASR")
for _p in (1, 2, 4, 5, 6):
    TEST_SETS[f"mnsc_part{_p}"] = (
        f"{ROOT}/data/Multitask-National-Speech-Corpus-v1/ASR-PART{_p}-Test/*.parquet", "context", "answer")
TRAIN_GLOB = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1/ASR-PART3-Train/*.parquet"


def iter_parquet(pattern, audio_col, text_col):
    """Yield (index, audio_bytes, text) without materialising the whole set."""
    i = 0
    for path in sorted(glob.glob(pattern)):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=[audio_col, text_col]):
            audio = batch.column(audio_col).to_pylist()
            text = batch.column(text_col).to_pylist()
            for a, t in zip(audio, text):
                yield i, a["bytes"], t
                i += 1


def load_set(name, limit=None, seed=0):
    """List of (idx, audio_bytes, ref). With limit, a fixed random subsample."""
    pattern, ac, tc = TEST_SETS[name]
    rows = list(iter_parquet(pattern, ac, tc))
    if limit and limit < len(rows):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(len(rows), size=limit, replace=False).tolist())
        rows = [r for r in rows if r[0] in keep]
    return rows


def decode_audio(b, target_sr=16000):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


_FILLER_TAG = re.compile(r"\((?:[^)]*)\)")          # (uh) (ppl) ...
_PARTICLE = re.compile(r"\[([^\]]*)\]")             # [lah] -> lah
_INTERJ = re.compile(r"!([a-zA-Z']+)!")             # !wah! -> wah
_ANGLE = re.compile(r"<[^>]*>:?")                   # <Speaker1>: <unk> ...
_ACRONYM = re.compile(r"\b(?:[a-zA-Z]_)+[a-zA-Z]\b")  # n_t_u -> ntu


def clean_nsc(text):
    """Strip NSC annotation markup. Used for references and training targets."""
    t = text or ""
    t = _ANGLE.sub(" ", t)
    t = _FILLER_TAG.sub(" ", t)
    t = _PARTICLE.sub(r"\1", t)
    t = _INTERJ.sub(r"\1", t)
    t = _ACRONYM.sub(lambda m: m.group(0).replace("_", ""), t)
    t = t.replace("**", " ").replace("#", "").replace("~", "")
    t = re.sub(r"\s*\.\.\.\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def join_letters(t):
    """'u s' -> 'us': NSC spells acronyms as letters, models write 'US'."""
    toks = t.split()
    out, run = [], []
    for w in toks + [""]:
        if len(w) == 1 and w.isalpha() and w not in ("a", "i"):
            run.append(w)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if w:
            out.append(w)
    return " ".join(out)


class Normalizer:
    def __init__(self):
        from transformers import WhisperTokenizer
        tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
        self._whisper = tok.normalize  # EnglishTextNormalizer + spelling map

    # spelling variants of the same word, applied to every model alike
    VARIANTS = {"yeah": "ya", "yah": "ya", "ok": "okay"}

    def __call__(self, text):
        t = join_letters(self._whisper(clean_nsc(text)))
        return " ".join(self.VARIANTS.get(w, w) for w in t.split())


def wer_report(refs, hyps):
    import jiwer
    o = jiwer.process_words(refs, hyps)
    return {
        "wer": round(100 * o.wer, 2),
        "sub": o.substitutions, "del": o.deletions, "ins": o.insertions,
        "ref_words": sum(len(r.split()) for r in refs), "n": len(refs),
    }


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
