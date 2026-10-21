"""Unit checks for speaker attribution; no model needed.  python scripts/serve/test_assign.py"""
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from server import Turn, Word, assign_speakers, relabel  # noqa: E402


def words(spec):
    return [Word(s, e, t) for s, e, t in spec]


def test_two_speakers():
    w = words([(0.0, 0.4, "hello"), (0.5, 0.9, "ah"), (1.2, 1.6, "ya"), (1.7, 2.1, "lah")])
    t = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.1, 2.2, "SPEAKER_01")]
    out = relabel(assign_speakers(w, t))
    assert out == [{"speaker": "A", "start": 0.0, "end": 0.9, "text": "hello ah"},
                   {"speaker": "B", "start": 1.2, "end": 2.1, "text": "ya lah"}], out


def test_gap_word_goes_to_nearest_turn():
    w = words([(0.0, 0.4, "one"), (1.04, 1.09, "two"), (1.5, 1.9, "three")])
    t = [Turn(0.0, 1.0, "S0"), Turn(1.1, 2.0, "S1")]
    out = relabel(assign_speakers(w, t))
    assert [s["text"] for s in out] == ["one", "two three"], out


def test_overlap_picks_larger_share():
    w = words([(0.8, 1.4, "mid")])
    t = [Turn(0.0, 1.0, "S0"), Turn(1.0, 2.0, "S1")]
    out = assign_speakers(w, t)
    assert out[0]["speaker"] == "S1", out


def test_no_turns_single_speaker():
    w = words([(0.0, 0.4, "a"), (0.5, 0.9, "b")])
    assert assign_speakers(w, []) == [{"speaker": "A", "start": 0.0, "end": 0.9, "text": "a b"}]


def test_empty():
    assert assign_speakers([], [Turn(0, 1, "S0")]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
