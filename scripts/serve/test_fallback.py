"""Unit checks for the token-ratio fallback; no model needed.

    python scripts/serve/test_fallback.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from fallback import CR_THRESHOLD, Piece, redecode_loops, token_compression_ratio  # noqa: E402


class FakeTokenizer:
    """Word-level stand-in for the Whisper fast tokenizer (same vocab size)."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 100) for w in text.split()]
        return SimpleNamespace(ids=ids)

    def get_vocab_size(self):
        return 51866


class FakeModel:
    def __init__(self, result):
        self.hf_tokenizer = FakeTokenizer()
        self.result = result
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((len(audio), kw))
        return iter(self.result), None


def seg(start, end, text, words=()):
    return SimpleNamespace(start=start, end=end, text=" " + text,
                           words=[SimpleNamespace(start=s, end=e, word=" " + w) for s, e, w in words])


NORMAL = "so how was the trip lah we went to the hawker centre after work"
LOOP = " ".join(["then we go lah"] * 30)


def test_ratio_separates_loops():
    tok = FakeTokenizer()
    assert token_compression_ratio(tok, NORMAL) < CR_THRESHOLD
    assert token_compression_ratio(tok, LOOP) > CR_THRESHOLD
    assert token_compression_ratio(tok, "") == 0.0


def test_only_loops_are_redecoded():
    model = FakeModel([seg(0.1, 1.5, "then we go home", [(0.1, 0.4, "then"), (1.2, 1.5, "home")])])
    wav = np.zeros(16000 * 6, dtype=np.float32)
    pieces = [Piece(0.0, 2.0, NORMAL, [(0.0, 0.3, "so")]), Piece(2.0, 5.0, LOOP)]
    out, n = redecode_loops(model, wav, pieces, language="en", word_timestamps=True)
    assert n == 1 and len(model.calls) == 1
    samples, kw = model.calls[0]
    assert samples == 3 * 16000                       # just the looping span
    assert kw["temperature"][0] > 0                   # greedy already failed
    assert kw["compression_ratio_threshold"] == CR_THRESHOLD
    assert kw["language"] == "en" and kw["word_timestamps"] is True
    assert out[0] is pieces[0]
    assert (out[1].start, out[1].end, out[1].text) == (2.1, 3.5, "then we go home")
    assert out[1].words == [(2.1, 2.4, "then"), (3.2, 3.5, "home")]  # back on the request clock


def test_empty_redecode_keeps_original():
    model = FakeModel([])
    pieces = [Piece(0.0, 2.0, LOOP)]
    out, n = redecode_loops(model, np.zeros(32000, dtype=np.float32), pieces)
    assert n == 1 and out == pieces


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
