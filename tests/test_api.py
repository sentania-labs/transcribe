"""API behaviour with stubbed models: no torch, no audio decoding."""
import os

os.environ.setdefault("TRANSCRIBE_API_KEY", "test-key")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as m  # noqa: E402

H = {"Authorization": "Bearer test-key"}
WAV = {"file": ("x.wav", b"RIFF0000", "audio/wav")}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(m, "load_models", lambda: None)  # startup must not load real models
    monkeypatch.setattr(m, "model_error", "")
    monkeypatch.setattr(m, "voice_error", "")
    monkeypatch.setattr(m, "decode", lambda fd, limit, name: m.np.zeros(16000 * 30, dtype="float32"))
    m.async_jobs.clear()
    m.idempotency.clear()
    m.current.clear()
    with TestClient(m.app, raise_server_exceptions=False) as c:
        yield c


def fake_result(*a, **k):
    return {"segments": [{"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 1.0, "text": "hi"}],
            "duration": 30.0, "language": "en", "speakers": None}


def test_errors_have_reason_codes(client, monkeypatch):
    r = client.post("/v1/audio/transcriptions", files=WAV)
    assert r.status_code == 401 and r.json()["error"]["reason"] == "unauthorized"
    r = client.post("/v1/audio/transcriptions", headers=H, files=WAV, data={"num_speakers": "two"})
    assert r.status_code == 400 and r.json() == {"error": {"reason": "invalid_request",
                                                           "message": "invalid or missing fields: num_speakers"}}
    assert "two" not in r.text  # values are never echoed
    r = client.get("/v1/audio/transcriptions/jobs/nope", headers=H)
    assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"


def test_unhandled_error_is_internal_without_detail(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret transcript text")
    monkeypatch.setattr(m, "run_job", boom)
    r = client.post("/v1/audio/transcriptions", headers=H, files=WAV)
    assert r.status_code == 500 and r.json() == {"error": {"reason": "internal", "message": "internal error"}}


def test_busy_has_retry_after(client):
    m.jobs._value = 0  # slot taken
    try:
        m.current.update(started=m.time.monotonic(), audio_seconds=3600)
        r = client.post("/v1/audio/transcriptions", headers=H, files=WAV)
        assert r.status_code == 429 and r.json()["error"]["reason"] == "busy"
        assert 30 <= int(r.headers["retry-after"]) <= 3600
    finally:
        m.jobs._value = 1
        m.current.clear()


def test_job_idempotency_and_same_result(client, monkeypatch):
    monkeypatch.setattr(m, "run_job", fake_result)
    direct = client.post("/v1/audio/transcriptions", headers=H, files=WAV, data={"response_format": "diarized_json"})
    r = client.post("/v1/audio/transcriptions/jobs", headers={**H, "Idempotency-Key": "rec-1"}, files=WAV,
                    data={"response_format": "diarized_json"})
    assert r.status_code == 202 and r.json()["status"] in ("queued", "running", "done")
    job_id = r.json()["id"]
    for _ in range(100):
        j = client.get(f"/v1/audio/transcriptions/jobs/{job_id}", headers=H).json()
        if j["status"] == "done":
            break
        m.time.sleep(0.02)
    assert j["status"] == "done" and j["result"] == direct.json()
    again = client.post("/v1/audio/transcriptions/jobs", headers={**H, "Idempotency-Key": "rec-1"}, files=WAV)
    assert again.status_code == 200 and again.json()["id"] == job_id
    form = client.post("/v1/audio/transcriptions/jobs", headers=H, files=WAV, data={"idempotency_key": "rec-1"})
    assert form.status_code == 200 and form.json()["id"] == job_id
    m.purge(m.time.time() + m.JOB_TTL + 1)
    assert client.get(f"/v1/audio/transcriptions/jobs/{job_id}", headers=H).status_code == 404
    assert "rec-1" not in m.idempotency


def test_internal_failure_frees_the_key(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(m, "run_job", boom)
    r = client.post("/v1/audio/transcriptions/jobs", headers={**H, "Idempotency-Key": "rec-2"}, files=WAV)
    job_id = r.json()["id"]
    for _ in range(100):
        j = client.get(f"/v1/audio/transcriptions/jobs/{job_id}", headers=H).json()
        if j["status"] == "failed":
            break
        m.time.sleep(0.02)
    assert j["error"] == {"reason": "internal", "message": "internal error"}
    assert "rec-2" not in m.idempotency


def test_signatures_follow_segment_labels(monkeypatch):
    class VM:
        entry = {"model": "m", "version": "v", "method": "pipeline-centroid-v1"}
    monkeypatch.setattr(m, "models", {"voices": [VM()]})
    monkeypatch.setattr(m, "VOICE_MIN_SECONDS", 20)
    full = [(0, 60, "SPK_A"), (60, 70, "SPK_B"), (70, 100, "SPK_C")]
    names = {"SPK_C": "Speaker 1", "SPK_A": "Speaker 2", "SPK_B": "Speaker 3", "SPK_D": "Speaker 4"}
    segments = [{"speaker": "Speaker 1"}, {"speaker": "Speaker 2"}, {"speaker": "Speaker 2"}, {"speaker": "Speaker 3"}]
    cents = {"SPK_A": [3.0, 4.0], "SPK_B": [1.0, 0.0], "SPK_C": [0.0, 0.0]}
    out = m.signatures(None, full, cents, names, segments, m.threading.Event())
    assert [s["label"] for s in out] == ["Speaker 1", "Speaker 2", "Speaker 3"]  # Speaker 4 has no segments
    by = {s["label"]: s for s in out}
    assert by["Speaker 2"]["embeddings"][0]["vector"] == [0.6, 0.8] and by["Speaker 2"]["segment_count"] == 2
    assert by["Speaker 3"]["embeddings"] == [] and by["Speaker 3"]["speech_seconds"] == 10.0  # under the floor
    assert by["Speaker 1"]["embeddings"] == []  # zero (padded) centroid
