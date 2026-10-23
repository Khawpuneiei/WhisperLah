"""Temperature fallback for faster-whisper that means what it means in eval_hf.py.

Two differences made the CTranslate2 model look ~1 WER point worse than the HF
harness on NSC Part 3 (21.25 vs 20.18), although decoded greedy it is better
(21.32 vs 22.04):

1. The compression ratio is measured on different things. transformers zlibs
   the *token ids* (`_retrieve_compression_ratio`), where 1.35 flags only real
   repetition loops (5 of 300 clips). faster-whisper zlibs the *text*, where
   1.35 flags almost everything (293 of 300), so it re-decoded nearly every
   clip at higher temperatures, doubled the runtime, and fixed none of the
   loops. `install()` swaps in the token-based measure, so the same threshold
   flags the same clips.
2. BatchedInferencePipeline keeps only the first temperature (no fallback at
   all), and the server uses it. `redecode_loops()` re-decodes just the
   segments whose output is a loop, sequentially, with the remaining
   temperatures. Loops are rare, so the batched speed is kept.

All of HF's 1.86-point gain over greedy came from those few looping clips
(greedy WER on them 150%, after fallback 48%).
"""
import math
import zlib
from dataclasses import dataclass, field

TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CR_THRESHOLD = 1.35  # token-based, as in eval_hf.py and the transformers docs


def token_compression_ratio(hf_tokenizer, text: str) -> float:
    """transformers' measure: zlib ratio of the token ids packed as bytes."""
    ids = hf_tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        return 0.0
    width = int(math.log2(hf_tokenizer.get_vocab_size()) / 8) + 1
    raw = b"".join(i.to_bytes(width, "little") for i in ids)
    return len(raw) / len(zlib.compress(raw))


def install(model):
    """Make faster-whisper's fallback check use the token-based ratio.

    faster-whisper looks up `get_compression_ratio` as a module global when it
    decides whether to fall back, so replacing it there is enough. Every
    WhisperModel in the process then uses it; this process only has one.
    """
    import faster_whisper.transcribe as fwt

    tok = model.hf_tokenizer
    fwt.get_compression_ratio = lambda text: token_compression_ratio(tok, text)


@dataclass
class Piece:
    """One decoded stretch of audio, in seconds from the start of the request."""
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)  # (start, end, word) tuples


def pieces_from_segments(segments, offset=0.0) -> list:
    out = []
    for s in segments:
        words = [(w.start + offset, w.end + offset, w.word.strip()) for w in (s.words or [])]
        out.append(Piece(s.start + offset, s.end + offset, s.text.strip(), words))
    return out


def redecode_loops(model, wav, pieces, *, sr=16000, threshold=CR_THRESHOLD,
                   temperatures=TEMPERATURES[1:], **transcribe_kw):
    """Re-decode every piece whose text is a repetition loop.

    `model` is the plain WhisperModel (sequential path, which honours the
    temperature list); `transcribe_kw` carries language, initial_prompt,
    word_timestamps and so on. Returns (pieces, number re-decoded).
    """
    tok = model.hf_tokenizer
    out, n = [], 0
    for p in pieces:
        if token_compression_ratio(tok, p.text) <= threshold:
            out.append(p)
            continue
        n += 1
        span = wav[int(p.start * sr):int(p.end * sr)]
        segs, _ = model.transcribe(
            span, temperature=temperatures, compression_ratio_threshold=threshold,
            log_prob_threshold=None, no_speech_threshold=None,
            condition_on_previous_text=False, vad_filter=False, **transcribe_kw)
        redone = pieces_from_segments(list(segs), offset=p.start)
        out.extend(redone if redone else [p])
    return out, n
