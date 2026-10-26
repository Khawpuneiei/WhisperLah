"""Cut full meetings into ASR test chunks with Silero VAD, independent of any labels.

    .venv-serve/Scripts/python.exe scripts/diar/chunk_meetings.py data/meetings/audio/KP-meet1.wav ... \
        --out data/meetings/chunks.parquet

Speech from Silero (faster-whisper's copy, no padding, 100 ms minimum silence)
is padded by 0.3 s and regions closer than 1 s are merged. Regions longer than
25 s are cut at their longest inner pause (recursively; lowest-energy 100 ms if
there is no pause) so every chunk is 3-25 s; a region shorter than 3 s is
widened into the surrounding silence, never past the midpoint to a neighbour.
Chunks never overlap. Output columns: audio {bytes: 16 kHz WAV}, text (empty),
meeting, start, end. The parquet stays under data/ and is never committed.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

SR = 16000


def cut(lo, hi, gaps, wav, min_len, max_len):
    """Split [lo, hi] (seconds) at the longest pause until every piece <= max_len."""
    if hi - lo <= max_len:
        return [(lo, hi)]
    inner = [(g1 - g0, (g0 + g1) / 2) for g0, g1 in gaps
             if lo + min_len <= (g0 + g1) / 2 <= hi - min_len]
    if inner:
        at = max(inner)[1]
    else:  # continuous speech: quietest 100 ms window
        a, b = int((lo + min_len) * SR), int((hi - min_len) * SR)
        win = SR // 10
        frames = wav[a:b][: (b - a) // win * win].reshape(-1, win)
        at = lo + min_len + (int(np.argmin((frames ** 2).mean(1))) + 0.5) * win / SR
    return cut(lo, at, gaps, wav, min_len, max_len) + cut(at, hi, gaps, wav, min_len, max_len)


def chunk(wav, pad, merge_gap, min_len, max_len):
    dur = len(wav) / SR
    raw = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=100, speech_pad_ms=0))
    raw = [(s["start"] / SR, s["end"] / SR) for s in raw]
    gaps = [(a[1], b[0]) for a, b in zip(raw, raw[1:])]
    regions = []
    for s, e in raw:
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if regions and s - regions[-1][1] < merge_gap:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    out = []
    for i, (s, e) in enumerate(regions):
        if e - s < min_len:
            left = 0.0 if i == 0 else (regions[i - 1][1] + s) / 2
            right = dur if i == len(regions) - 1 else (e + regions[i + 1][0]) / 2
            need = (min_len - (e - s)) / 2
            s, e = max(left, s - need), min(right, e + need)
        out += cut(s, e, gaps, wav, min_len, max_len)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", default="data/meetings/chunks.parquet")
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--min-len", type=float, default=3.0)
    ap.add_argument("--max-len", type=float, default=25.0)
    args = ap.parse_args()

    rows = {"audio": [], "text": [], "meeting": [], "start": [], "end": []}
    for path in map(Path, args.audio):
        wav = decode_audio(str(path), sampling_rate=SR)
        spans = chunk(wav, args.pad, args.merge_gap, args.min_len, args.max_len)
        for s, e in spans:
            buf = io.BytesIO()
            sf.write(buf, wav[int(round(s * SR)):int(round(e * SR))], SR, format="WAV", subtype="PCM_16")
            rows["audio"].append({"bytes": buf.getvalue(), "path": f"{path.stem}_{s:09.2f}.wav"})
            rows["text"].append("")
            rows["meeting"].append(path.stem)
            rows["start"].append(round(s, 3))
            rows["end"].append(round(e, 3))
        lens = [e - s for s, e in spans]
        print(f"{path.stem}: {len(wav) / SR:.2f} s audio, {len(spans)} chunks, {sum(lens):.2f} s speech, "
              f"chunk {min(lens):.1f}-{max(lens):.1f} s", flush=True)
    pq.write_table(pa.table(rows), args.out)
    print(f"wrote {args.out}: {len(rows['text'])} chunks, {sum(e - s for s, e in zip(rows['start'], rows['end'])):.2f} s")


if __name__ == "__main__":
    main()
