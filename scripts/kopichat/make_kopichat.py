"""Package KopiChat recordings into the harness's parquet schema.

Input (see README.md):
    data/kopichat/audio/<chat>.m4a|.wav|.mp3
    data/kopichat/transcripts/<chat>.txt      one turn per line, "<Speaker1>: ..." style

Each chat is cut into clips of at most 30 s. Turn timestamps are not
available from hand-corrected text, so clips are made by forced alignment:
WhisperLah (or base Whisper) is run with word timestamps, the corrected
transcript is aligned to those timestamps by words, and cuts are placed at
turn boundaries nearest to each 30 s mark. The alignment only decides where
to cut; the clip text is always the corrected transcript.

Output:
    data/kopichat/test.parquet   columns: audio {bytes, path}, text, chat, start, end

    python scripts/kopichat/make_kopichat.py
    python scripts/kopichat/make_kopichat.py --agreement chat03 chat07 chat12   # inter-annotator WER
"""
import argparse
import glob
import io
import os
import re
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT, Normalizer, wer_report  # noqa: E402

KC = f"{ROOT}/data/kopichat"
MAX_S = 30.0


def load_wav(path):
    """Any format to 16 kHz mono float32 via ffmpeg (phones record m4a)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                          "-f", "wav", "-"], capture_output=True, check=True).stdout
    wav, sr = sf.read(io.BytesIO(out), dtype="float32")
    return wav


def turns(path):
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for l in lines:
        m = re.match(r"(<Speaker\d>):\s*(.*)", l)
        out.append((m.group(1), m.group(2)) if m else ("", l))
    return out


def word_times(wav, model_dir=None):
    """Word timestamps from Whisper, used only to place the cuts."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    pr = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16).to("cuda")
    if model_dir:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, model_dir).merge_and_unload()
    f = pr(wav, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
           return_attention_mask=True)
    with torch.inference_mode():
        out = m.generate(f.input_features.to("cuda", torch.float16), attention_mask=f.attention_mask.to("cuda"),
                         language="en", task="transcribe", return_timestamps=True, return_token_timestamps=True,
                         condition_on_prev_tokens=False)
    # (word, start_s, end_s) from token timestamps
    ids = out["sequences"][0].tolist()
    ts = out["token_timestamps"][0].tolist()
    words, cur, t0 = [], "", None
    for tok, t in zip(ids, ts):
        s = pr.tokenizer.decode([tok])
        if tok >= pr.tokenizer.eos_token_id:
            continue
        if s.startswith(" ") and cur:
            words.append((cur.strip(), t0, t))
            cur, t0 = "", None
        if t0 is None:
            t0 = t
        cur += s
    if cur:
        words.append((cur.strip(), t0, ts[-1]))
    return words


def cut_points(ref_turns, hyp_words, total_s):
    """Align corrected words to Whisper's timed words; return turn-end times."""
    import difflib
    ref = [(ti, w) for ti, (_, t) in enumerate(ref_turns) for w in t.split()]
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
    a = [norm(w) for _, w in ref]
    b = [norm(w) for w, _, _ in hyp_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    time_of = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            time_of[blk.a + k] = hyp_words[blk.b + k][2]
    ends = []
    for ti in range(len(ref_turns)):
        idx = [i for i, (t, _) in enumerate(ref) if t == ti and i in time_of]
        ends.append(time_of[idx[-1]] if idx else None)
    # fill unaligned turns by interpolation
    known = [(i, e) for i, e in enumerate(ends) if e is not None]
    for i in range(len(ends)):
        if ends[i] is None:
            prev = max([e for j, e in known if j < i], default=0.0)
            nxt = min([e for j, e in known if j > i], default=total_s)
            ends[i] = (prev + nxt) / 2
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="LoRA adapter dir for alignment (defaults to base Whisper)")
    ap.add_argument("--agreement", nargs="*", help="chats with a second transcript in transcripts2/, report WER between annotators")
    args = ap.parse_args()

    if args.agreement:
        norm = Normalizer()
        for chat in args.agreement:
            a = " ".join(t for _, t in turns(f"{KC}/transcripts/{chat}.txt"))
            b = " ".join(t for _, t in turns(f"{KC}/transcripts2/{chat}.txt"))
            rep = wer_report([norm(a)], [norm(b)])
            print(f"{chat}: annotator disagreement WER {rep['wer']}%  ({rep['ref_words']} words)")
        return

    rows = []
    for audio in sorted(glob.glob(f"{KC}/audio/*")):
        chat = os.path.splitext(os.path.basename(audio))[0]
        tpath = f"{KC}/transcripts/{chat}.txt"
        if not os.path.exists(tpath):
            print(f"skip {chat}: no transcript", flush=True)
            continue
        wav = load_wav(audio)
        tr = turns(tpath)
        ends = cut_points(tr, word_times(wav, args.model), len(wav) / 16000)
        start_t, start_i = 0.0, 0
        for i, e in enumerate(ends):
            last = i == len(ends) - 1
            if e - start_t > MAX_S or last:
                # cut at the previous turn end if this turn would overflow
                j, cut = (i - 1, ends[i - 1]) if (e - start_t > MAX_S and i > start_i) else (i, e)
                seg = wav[int(start_t * 16000):int(cut * 16000)]
                text = "\n".join(f"{sp}: {t}" if sp else t for sp, t in tr[start_i:j + 1])
                buf = io.BytesIO()
                sf.write(buf, seg, 16000, format="WAV", subtype="PCM_16")
                rows.append({"audio": {"bytes": buf.getvalue(), "path": f"{chat}_{len(rows):03d}.wav"},
                             "text": text, "chat": chat, "start": start_t, "end": cut})
                start_t, start_i = cut, j + 1
                if j == i - 1 and not last:
                    continue
        print(f"{chat}: {len(wav) / 16000:.0f} s, {len(tr)} turns", flush=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, f"{KC}/test.parquet")
    secs = sum(r["end"] - r["start"] for r in rows)
    print(f"wrote {len(rows)} clips, {secs / 60:.1f} min -> {KC}/test.parquet", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
