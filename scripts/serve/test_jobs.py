"""Unit checks for the job API, worker step and callback signing; no Modal,
no GPU.  python scripts/serve/test_jobs.py"""
import contextlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import jobs as J  # noqa: E402

TOKEN, SECRET = "tok-123", "cb-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
REQ = {"job_id": "job-0001-abcd", "audio_url": "https://x.supabase.co/a.m4a?token=s",
       "callback_url": "https://lepaq.app/api/transcription/callback", "mode": "room",
       "num_speakers": 3}


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def api(store, queued):
    return TestClient(J.make_api(store, queued.append, TOKEN, "v3"))


def test_create_is_idempotent_and_authed():
    store, queued = Store(), []
    c = api(store, queued)
    assert c.post("/jobs", json=REQ).status_code == 401
    assert c.post("/jobs", json=REQ, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.post("/jobs", json=REQ, headers=AUTH)
    assert r.status_code == 202 and r.json() == {"job_id": REQ["job_id"], "status": "queued"}
    r = c.post("/jobs", json=REQ, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert len(queued) == 1 and queued[0]["model_version"] == "v3"
    assert c.get(f"/jobs/{REQ['job_id']}", headers=AUTH).json()["attempts"] == 0
    assert c.get("/jobs/missing-job-id", headers=AUTH).status_code == 404


def test_create_rejects_bad_input():
    c = api(Store(), [])
    for bad in ({"audio_url": "http://plain"}, {"mode": "loud"}, {"num_speakers": 0},
                {"job_id": "../../etc"}):
        assert c.post("/jobs", json={**REQ, **bad}, headers=AUTH).status_code == 422, bad


def test_signature_roundtrip():
    body, ts = b'{"job_id":"j"}', int(time.time())
    sig = J.sign(SECRET, ts, body)
    assert sig.startswith("v1=") and J.verify(SECRET, str(ts), sig, body)
    assert len(sig) == 67 and all(c in "0123456789abcdef" for c in sig[3:])  # webapp: exactly 64 lowercase hex
    assert not J.verify(SECRET, str(ts), sig, body + b" ")
    assert not J.verify("other", str(ts), sig, body)
    assert not J.verify(SECRET, str(ts - 301), J.sign(SECRET, ts - 301, body), body)


def test_callback_retries_5xx_and_stops_on_4xx():
    calls, sleeps = [], []

    def post(url, body, headers):
        calls.append(headers)
        assert J.verify(SECRET, headers["X-Lepaq-Timestamp"], headers["X-Lepaq-Signature"], body)
        return [503, 502, 200][len(calls) - 1]

    assert J.send_callback("https://cb", {"a": 1}, SECRET, post, sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [1, 4]
    calls.clear()
    assert not J.send_callback("https://cb", {"a": 1}, SECRET, lambda *a: calls.append(1) or 404,
                               sleep=lambda s: None)
    assert len(calls) == 1


def fake_stream(status=200, data=b"RIFF....audio"):
    @contextlib.contextmanager
    def get_stream(url):
        def raise_for_status():
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
        yield SimpleNamespace(status_code=status, raise_for_status=raise_for_status,
                              iter_bytes=lambda: iter([data]))
    return get_stream


def run(store, job, transcribe, status=200):
    posted = []

    def post(url, body, headers):
        posted.append(json.loads(body))
        return 200

    out = J.run_job(job, store=store, transcribe=transcribe, get_stream=fake_stream(status),
                    post=post, secret=SECRET, max_attempts=3, sleep=lambda s: None)
    return out, posted


def queued_job():
    store = Store()
    job = {**REQ, "status": "queued", "attempts": 0, "model_version": "v3"}
    store.put(job["job_id"], job)
    return store, job


def test_job_done_sends_result():
    store, job = queued_job()
    seen = {}

    def transcribe(path, mode, diarize, language, n, lo, hi):
        seen.update(exists=Path(path).exists(), mode=mode, n=n, path=path)
        return {"segments": [{"speaker": "A", "start": 0, "end": 1, "text": "can lah"}]}

    out, posted = run(store, job, transcribe)
    assert seen["exists"] and seen["mode"] == "room" and seen["n"] == 3
    assert not Path(seen["path"]).exists()  # temp audio removed
    assert out["status"] == "done" and store[job["job_id"]]["attempts"] == 1
    assert posted == [{"job_id": job["job_id"], "status": "done", "model_version": "v3",
                       "result": {"segments": [{"speaker": "A", "start": 0, "end": 1,
                                                "text": "can lah"}]}}]
    out2, posted2 = run(store, job, transcribe)  # duplicate delivery: no second callback
    assert out2["status"] == "done" and posted2 == []


def test_result_is_plain_json():
    import numpy as np

    store, job = queued_job()
    out, _ = run(store, job, lambda *a: {"segments": [{"start": np.float32(0.5), "end": np.float64(1.0)}]})
    seg = out["result"]["segments"][0]
    assert type(seg["start"]) is float and type(seg["end"]) is float, seg


def test_expired_audio_url_fails_without_retry():
    store, job = queued_job()
    out, posted = run(store, job, lambda *a: {}, status=403)
    assert out["status"] == "failed" and "403" in out["error"]
    assert posted[0]["status"] == "failed"


def test_transient_error_retries_then_fails_on_last_attempt():
    store, job = queued_job()

    def boom(*a):
        raise RuntimeError("CUDA error: out of memory")

    for attempt in (1, 2, 3):
        try:
            run(store, job, boom)
            raise AssertionError("should raise")
        except RuntimeError as e:
            assert "out of memory" in str(e)
        state = store[job["job_id"]]
        assert state["attempts"] == attempt
        assert state["status"] == ("failed" if attempt == 3 else "queued")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
