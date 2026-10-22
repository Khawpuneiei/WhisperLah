"""Synthetic Singlish conversations with exact speaker timings, for pyannote.

None of our Singlish audio comes with speaker start/end times, and MNSC Parts
1 and 2 carry no speaker id either. Each MNSC row is one speaker reading one
sentence, so a conversation is built from one clip per speaker: the clip is
cut at its internal pauses and the pieces of 2 to 4 speakers are interleaved,
with gaps, some overlap and a quick back-channel now and then. A speaker
therefore never needs a second clip, and the labels stay correct.

Speech times come from an energy detector on each clean clip before mixing,
so the RTTM marks only where someone is actually talking.

    python scripts/diar/make_synth.py --split train --shards "ASR-PART1-Train/*.parquet,ASR-PART2-Train/*.parquet" --n 3000
    python scripts/diar/make_synth.py --split test --shards "ASR-PART1-Test/*.parquet,ASR-PART2-Test/*.parquet" --n 300

Writes data/diar/<split>/wav/*.wav, <split>.rttm, <split>.uem, <split>.lst.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, decode_audio, iter_parquet  # noqa: E402

SR = 16000
HOP = 160  # 10 ms frames for the speech detector
MNSC = f"{ROOT}/data/Multitask-National-Speech-Corpus-v1"


def speech_regions(wav, min_pause=0.25, min_speech=0.15):
    """(start, end) seconds of speech: frame energy above a level set per clip."""
    n = len(wav) // HOP
    if n == 0:
        return []
    rms = np.sqrt(np.mean(wav[: n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    # threshold between the clip's noise floor and its speech level
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    on = db > floor + 0.35 * (peak - floor)
    regions, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append([start, i])
            start = None
    merged = []
    for s, e in regions:  # bridge pauses shorter than min_pause
        if merged and (s - merged[-1][1]) * HOP / SR < min_pause:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s * HOP / SR, e * HOP / SR) for s, e in merged if (e - s) * HOP / SR >= min_speech]


def pieces(wav, regions, max_pieces=3):
    """Cut a clip at its longest pauses into up to max_pieces speech pieces."""
    if not regions:
        return []
    gaps = sorted(range(1, len(regions)), key=lambda i: regions[i][0] - regions[i - 1][1], reverse=True)
    cuts = sorted(gaps[: max_pieces - 1])
    groups, prev = [], 0
    for c in cuts + [len(regions)]:
        groups.append(regions[prev:c])
        prev = c
    out = []
    for g in groups:
        s, e = max(0.0, g[0][0] - 0.05), min(len(wav) / SR, g[-1][1] + 0.05)
        seg = wav[int(s * SR): int(e * SR)]
        out.append((seg, [(a - s, b - s) for a, b in g]))  # speech times relative to the piece
    return out


def make_mix(clips, rng):
    """Interleave 2-4 speakers' pieces into one conversation; return audio and turns."""
    spk_pieces = [pieces(w, r) for w, r in clips]
    gains = 10 ** (rng.uniform(-6, 3, len(clips)) / 20)
    total = sum(len(ps) for ps in spk_pieces)
    # take speakers in turn: order by each piece's position in its own clip, shuffled across speakers
    order, cursor = [], [0] * len(spk_pieces)
    last = None
    while len(order) < total:
        choices = [k for k in range(len(spk_pieces)) if cursor[k] < len(spk_pieces[k]) and k != last]
        if not choices:
            choices = [k for k in range(len(spk_pieces)) if cursor[k] < len(spk_pieces[k])]
        k = int(rng.choice(choices))
        order.append((k, spk_pieces[k][cursor[k]]))
        cursor[k] += 1
        last = k
    placed, prev_t, prev_len = [], 0.5, 0.0
    for i, (k, (seg, speech)) in enumerate(order):
        t = prev_t
        if i:
            r = rng.random()
            if r < 0.2:     # overlap: start before the previous piece ends
                gap = -rng.uniform(0.2, max(0.25, min(1.5, 0.5 * prev_len)))
            elif r < 0.3:   # quick hand-over
                gap = rng.uniform(0.0, 0.15)
            else:           # ordinary pause
                gap = rng.exponential(0.5)
            t = max(0.0, prev_t + prev_len + gap)
        placed.append((k, t, seg * gains[k], speech))
        prev_t, prev_len = t, len(seg) / SR
        # a short back-channel from another speaker over a long piece
        if prev_len > 2.5 and len(spk_pieces) > 1 and rng.random() < 0.15:
            other = int(rng.choice([j for j in range(len(spk_pieces)) if j != k]))
            ow, ospeech = spk_pieces[other][-1]
            bc_len = int(min(len(ow), rng.uniform(0.3, 0.7) * SR))
            bc_speech = [(a, min(b, bc_len / SR)) for a, b in ospeech if a < bc_len / SR]
            if bc_speech:
                placed.append((other, t + rng.uniform(0.5, prev_len - 0.8), ow[:bc_len] * gains[other], bc_speech))
    end = max(t0 + len(seg) / SR for _, t0, seg, _ in placed) + 0.5
    mix = np.zeros(int(end * SR) + 1, dtype=np.float32)
    turns = []
    for k, t0, seg, speech in placed:
        i0 = int(t0 * SR)
        mix[i0: i0 + len(seg)] += seg
        turns += [(t0 + a, b - a, k) for a, b in speech]
    # low background noise at 20-40 dB below speech
    snr = rng.uniform(20, 40)
    level = np.sqrt(np.mean(mix ** 2) + 1e-12) / 10 ** (snr / 20)
    mix += rng.normal(0, level, len(mix)).astype(np.float32)
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sorted(turns)


def load_pool(patterns, need, rng, max_dur=12.0):
    files = sorted(f for p in patterns for f in glob.glob(f"{MNSC}/{p}"))
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    pool = []
    for f in files:
        for _, b, _ in iter_parquet(f, "context", "answer"):
            wav = decode_audio(b).astype(np.float32)
            if len(wav) / SR > max_dur:
                continue
            reg = speech_regions(wav)
            if sum(e - s for s, e in reg) >= 1.0:
                pool.append((wav, reg))
        print(f"{f.split('/')[-1]}: pool {len(pool)}", flush=True)
        if len(pool) >= need:
            break
    rng.shuffle(pool)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shards", required=True, help="comma-separated globs under the MNSC folder")
    ap.add_argument("--n", type=int, required=True, help="number of conversations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(ROOT) / "data" / "diar" / args.split
    (out / "wav").mkdir(parents=True, exist_ok=True)
    speakers_per = rng.choice([2, 2, 3, 3, 4], size=args.n)
    pool = load_pool(args.shards.split(","), int(speakers_per.sum()), rng)
    if len(pool) < speakers_per.sum():
        raise SystemExit(f"only {len(pool)} usable clips for {speakers_per.sum()} speakers")
    rttm, uem, names, hours, overlap = [], [], [], 0.0, 0.0
    i = 0
    for c, n_spk in enumerate(speakers_per):
        name = f"{args.split}_{c:05d}"
        mix, turns = make_mix(pool[i: i + n_spk], rng)
        i += n_spk
        sf.write(out / "wav" / f"{name}.wav", mix, SR, subtype="PCM_16")
        dur = len(mix) / SR
        rttm += [f"SPEAKER {name} 1 {s:.3f} {d:.3f} <NA> <NA> {name}_spk{k} <NA> <NA>" for s, d, k in turns]
        uem.append(f"{name} 1 0.000 {dur:.3f}")
        names.append(name)
        hours += dur / 3600
        # seconds where two or more labelled speakers talk at once
        act = np.zeros(int(dur * 100) + 1, dtype=np.int16)
        for s, d, _ in turns:
            act[int(s * 100): int((s + d) * 100)] += 1
        overlap += (act > 1).sum() / 100
    (out / f"{args.split}.rttm").write_text("\n".join(rttm) + "\n")
    (out / f"{args.split}.uem").write_text("\n".join(uem) + "\n")
    (out / f"{args.split}.lst").write_text("\n".join(names) + "\n")
    stats = {"split": args.split, "conversations": args.n, "hours": round(hours, 2),
             "overlap_share": round(overlap / (hours * 3600), 3), "speakers": int(speakers_per.sum())}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), "\nSYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
