"""Unit checks for speaker attribution and the server helpers; no model needed.

    python scripts/serve/test_assign.py
"""
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from server import Turn, Word, assign_speakers, embeddings_by_label, parse_bool_form, relabel  # noqa: E402


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


def test_embeddings_follow_full_labels():
    # S1 only ever talks over someone, so the exclusive diarization drops it;
    # S2 must still get the third centroid, not the second.
    full_labels = ["S0", "S1", "S2"]
    vectors = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    emb = embeddings_by_label(full_labels, vectors)
    assert emb["S2"] == [3.0, 0.0], emb
    exclusive_turns = [Turn(0, 1, "S0"), Turn(1, 2, "S2")]
    names = {}
    relabel(assign_speakers(words([(0.1, 0.5, "a"), (1.1, 1.5, "b")]), exclusive_turns), names)
    speakers = {names[k]: v for k, v in emb.items() if k in names}
    assert speakers == {"A": [1.0, 0.0], "B": [3.0, 0.0]}, speakers


def test_embeddings_missing():
    assert embeddings_by_label(["S0"], None) == {}


def test_parse_bool_form():
    assert parse_bool_form(None) is True          # omitted: diarize by default
    assert parse_bool_form("") is True
    assert parse_bool_form("false") is False
    assert parse_bool_form(" 0 ") is False
    assert parse_bool_form("Off") is False
    assert parse_bool_form("true") is True
    assert parse_bool_form("1") is True
    assert parse_bool_form(None, default=False) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
