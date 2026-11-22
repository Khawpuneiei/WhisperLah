"""Unit checks for refusing a file with nothing to transcribe; no model, no GPU.
Needs PyAV and faster-whisper (scripts/serve/requirements.txt).

    python scripts/serve/test_audio_check.py

A video with no sound track, such as a screen recording made with the
microphone off, used to reach decode_audio and fail there with "IndexError:
tuple index out of range", three times over (Modal retries), before the
person saw that IndexError as the reason.
"""
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import server  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def media(name, video=True, audio=True, audio_frames=True):
    """A one-second file: 16x16 video and/or a 440 Hz tone."""
    import av

    path = TMP / name
    vcodec, acodec = ("libvpx", "libopus") if name.endswith(".webm") else ("mpeg4", "aac")
    with av.open(str(path), "w") as out:
        vs = as_ = None
        if video:
            vs = out.add_stream(vcodec, rate=10)
            vs.width = vs.height = 16
            vs.pix_fmt = "yuv420p"
        if audio:
            as_ = out.add_stream(acodec, rate=48000, layout="mono")
        if vs:
            for i in range(10):
                img = np.full((16, 16, 3), i * 20, np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
                frame.pts = i
                out.mux(vs.encode(frame))
            out.mux(vs.encode())
        if as_ and audio_frames:
            size = as_.codec_context.frame_size or 1024
            t = np.arange(48000) / 48000
            tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            for start in range(0, 48000 - size, size):
                frame = av.AudioFrame.from_ndarray(tone[None, start:start + size], format="fltp",
                                                   layout="mono")
                frame.sample_rate = 48000
                frame.pts = start
                frame.time_base = Fraction(1, 48000)
                out.mux(as_.encode(frame))
            out.mux(as_.encode())
    return str(path)


def wav(name):
    path = TMP / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.sin(np.arange(16000) / 5) * 8000).astype("<i2").tobytes())
    return str(path)


def text(name, body):
    path = TMP / name
    path.write_text(body)
    return str(path)


def refused(path):
    try:
        server.check_audio(path)
    except server.BadAudio as e:
        assert e.permanent is True
        return str(e)
    return None


def test_files_with_sound_pass():
    for path in (wav("call.wav"), media("audio.m4a", video=False), media("both.mp4"),
                 media("audio.webm", video=False), media("both.webm")):
        assert refused(path) is None, path


def test_video_without_sound_is_refused_with_a_readable_reason():
    for path in (media("screen.mp4", audio=False), media("screen.webm", audio=False)):
        assert refused(path) == server.NO_SOUND_TRACK, path


def test_not_audio_at_all_is_refused():
    for path in (text("page.mp4", "<html>not a video</html>"), text("empty.m4a", ""),
                 text("err.webm", '{"error":"expired"}')):
        assert refused(path) == server.NOT_AUDIO, path


def test_transcribe_stops_before_the_decoder():
    # The real _transcribe on a video without sound: BadAudio, never the
    # IndexError. It stops before touching anything on the engine.
    for path, want in ((media("screen2.mp4", audio=False), server.NO_SOUND_TRACK),
                       (media("silent-track.webm", audio_frames=False), server.EMPTY_SOUND_TRACK)):
        try:
            server.Engine._transcribe(None, path, "room", True, None, None, None, None)
            raise AssertionError("should raise")
        except server.BadAudio as e:
            assert str(e) == want, (path, e)


def test_http_route_answers_422():
    class Engine:
        def transcribe(self, *a):
            raise server.BadAudio(server.NO_SOUND_TRACK)

    server.engine = Engine()
    try:
        r = TestClient(server.app).post("/transcribe", files={"file": ("screen.mp4", b"x")})
    finally:
        server.engine = None
    assert r.status_code == 422 and r.json()["detail"] == server.NO_SOUND_TRACK, r.text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
