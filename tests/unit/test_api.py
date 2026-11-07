import time

import pytest

from app.schemas.ocr import ExtractionOptions
from tests.conftest import FakeEngine, build_client


def upload(samples, name):
    return {"file": (name, (samples / name).read_bytes())}


def wait_for_job(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/ocr/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_reports_dependencies_without_loading_models(client, fake_engine):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["dependencies"]["paddleocr"] is not None
    assert body["ready_ocr_engines"] == ["paddleocr"]
    assert body["limits"]["max_upload_mb"] == 25
    assert fake_engine.calls == []


def test_engines_lists_availability_and_hides_unusable_languages(client):
    body = client.get("/api/v1/ocr/engines").json()

    engines = {engine["name"]: engine for engine in body["engines"]}
    assert engines["paddleocr"]["available"] is True
    assert engines["paddleocr"]["supported_languages"] == ["ar", "en", "fr"]
    assert engines["tesseract"]["available"] is False
    assert engines["tesseract"]["supported_languages"] == []
    assert "not found" in engines["tesseract"]["warnings"][0]
    assert body["language_detection"] is True


def test_extract_json_matches_the_documented_schema(client, samples):
    response = client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"), data={"languages": "en,fr"})

    body = response.json()
    assert response.status_code == 200
    assert body["document"] == {
        "filename": "english.png",
        "content_type": "image/png",
        "size_bytes": (samples / "english.png").stat().st_size,
        "page_count": 1,
    }
    assert body["extraction"]["requested_languages"] == ["en", "fr"]
    assert body["extraction"]["detected_languages"] is None
    block = body["pages"][0]["blocks"][0]
    assert set(block) == {"text", "confidence", "bbox"}
    assert all(0 <= value <= 1 for value in block["bbox"])
    assert body["pages"][0]["tables"] == []
    assert body["warnings"] == []


def test_extract_text_output_is_plain_text(client, samples):
    response = client.post("/api/v1/ocr/extract", files=upload(samples, "scanned.pdf"), data={"output_format": "text"})

    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "--- Page 1 ---\nFAKE OCR TEXT\n\n--- Page 2 ---\nFAKE OCR TEXT"
    assert response.headers["x-ocr-engine"] == "paddleocr"


def test_client_supplied_path_in_filename_is_not_used(client, samples):
    files = {"file": ("../../evil.png", (samples / "english.png").read_bytes())}

    body = client.post("/api/v1/ocr/extract", files=files).json()

    assert body["document"]["filename"] == "evil.png"


def test_empty_file_gets_controlled_error(client):
    response = client.post("/api/v1/ocr/extract", files={"file": ("e.png", b"")})

    assert response.status_code == 400
    assert response.json() == {"error": {"code": "EMPTY_FILE", "message": "The uploaded file is empty."}}


def test_invalid_file_gets_controlled_error(client):
    response = client.post("/api/v1/ocr/extract", files={"file": ("x.png", b"\x00\x01not an image")})

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_file_over_limit_is_rejected(tmp_path, samples):
    with build_client(tmp_path, max_upload_mb=1) as small_client:
        response = small_client.post("/api/v1/ocr/extract", files={"file": ("big.txt", b"a" * (2 * 1024 * 1024))})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_omitted_languages_are_detected_and_returned(client, samples):
    body = client.post("/api/v1/ocr/extract", files=upload(samples, "document.docx")).json()

    assert body["extraction"]["requested_languages"] == []
    assert body["extraction"]["detected_languages"] == ["en"]


def test_sync_extraction_over_the_concurrency_limit_is_429(tmp_path, samples):
    with build_client(tmp_path, sync_max_active=1) as small_client:
        slots = small_client.app.state.sync_slots
        slots.acquire()
        try:
            busy = small_client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"))
            busy_batch = small_client.post("/api/v1/ocr/batch", files=[("files", ("e.png", (samples / "english.png").read_bytes()))])
        finally:
            slots.release()
        free = small_client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"))

    assert busy.status_code == busy_batch.status_code == 429
    assert busy.json()["error"]["code"] == "TOO_MANY_REQUESTS"
    assert free.status_code == 200


def test_oversized_upload_rejection_carries_cors_headers(tmp_path):
    with build_client(tmp_path, max_upload_mb=1, cors_origins=["http://allowed.example"]) as cors_client:
        response = cors_client.post(
            "/api/v1/ocr/extract",
            content=b"x",
            headers={"content-length": str(10 * 1024 * 1024), "origin": "http://allowed.example"},
        )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://allowed.example"


def test_unsupported_language_is_a_422_with_error_schema(client, samples):
    response = client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"), data={"languages": "de"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_invalid_enum_value_uses_error_schema(client, samples):
    response = client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"), data={"mode": "magic"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_missing_engine_gives_actionable_503(client, samples):
    response = client.post("/api/v1/ocr/extract", files=upload(samples, "english.png"), data={"engine": "tesseract"})

    assert response.status_code == 503
    assert "not found" in response.json()["error"]["message"]


def test_unknown_route_uses_error_schema(client):
    response = client.get("/api/v1/nothing")

    assert response.status_code == 404
    assert set(response.json()["error"]) == {"code", "message"}


def test_batch_reports_each_file_separately(client, samples):
    files = [
        ("files", ("english.png", (samples / "english.png").read_bytes())),
        ("files", ("broken.bin", b"\x00\x01garbage")),
        ("files", ("document.docx", (samples / "document.docx").read_bytes())),
    ]

    body = client.post("/api/v1/ocr/batch", files=files).json()

    assert [(item["filename"], item["status"]) for item in body["items"]] == [
        ("english.png", "completed"),
        ("broken.bin", "failed"),
        ("document.docx", "completed"),
    ]
    assert body["items"][1]["error"]["code"] == "UNSUPPORTED_FORMAT"
    assert body["items"][1]["result"] is None


def test_batch_size_is_bounded(tmp_path, samples):
    with build_client(tmp_path, batch_max_files=2) as small_client:
        files = [("files", ("a.png", (samples / "english.png").read_bytes()))] * 3
        response = small_client.post("/api/v1/ocr/batch", files=files)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "BATCH_TOO_LARGE"


def test_job_returns_uuid_and_poll_returns_the_result(client, samples):
    submitted = client.post("/api/v1/ocr/jobs", files=upload(samples, "scanned.pdf"), data={"languages": "en"})

    body = submitted.json()
    assert submitted.status_code == 202
    assert body["status"] == "queued"
    assert len(body["job_id"]) == 36
    assert body["status_url"] == f"/api/v1/ocr/jobs/{body['job_id']}"

    job = wait_for_job(client, body["job_id"])
    assert job["status"] == "completed"
    assert job["pages_done"] == job["total_pages"] == 2
    assert job["result"]["pages"][1]["text"] == "FAKE OCR TEXT"
    assert job["error"] is None

    text = client.get(f"/api/v1/ocr/jobs/{body['job_id']}/text")
    assert text.text.startswith("--- Page 1 ---")


def test_job_rejects_bad_upload_immediately(client):
    response = client.post("/api/v1/ocr/jobs", files={"file": ("x.bin", b"\x00\x01garbage")})

    assert response.status_code == 415


def test_job_processing_error_is_reported_when_polling(client, samples):
    body = client.post(
        "/api/v1/ocr/jobs", files=upload(samples, "english.png"), data={"engine": "tesseract"}
    ).json()

    job = wait_for_job(client, body["job_id"])

    assert job["status"] == "failed"
    assert job["error"]["code"] == "ENGINE_UNAVAILABLE"
    assert job["result"] is None
    assert client.get(f"/api/v1/ocr/jobs/{body['job_id']}/text").status_code == 409


def test_unknown_job_is_404(client):
    response = client.get("/api/v1/ocr/jobs/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_job_status_is_visible_while_processing(tmp_path, samples):
    with build_client(tmp_path, engine=FakeEngine(delay=0.4)) as slow_client:
        job_id = slow_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png")).json()["job_id"]

        in_flight = slow_client.get(f"/api/v1/ocr/jobs/{job_id}").json()
        finished = wait_for_job(slow_client, job_id)

    assert in_flight["status"] in ("queued", "processing")
    assert in_flight["result"] is None
    assert finished["status"] == "completed"


def test_active_job_count_is_bounded(tmp_path, samples):
    with build_client(tmp_path, engine=FakeEngine(delay=0.5), job_max_active=1) as small_client:
        first = small_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png"))
        second = small_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png"))

        assert first.status_code == 202
        assert second.status_code == 429
        wait_for_job(small_client, first.json()["job_id"])


def test_stored_jobs_are_capped_by_dropping_the_oldest_finished(tmp_path, samples):
    with build_client(tmp_path, job_max_stored=2) as small_client:
        job_ids = []
        for _ in range(3):
            job_ids.append(small_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png")).json()["job_id"])
            wait_for_job(small_client, job_ids[-1])

        statuses = [small_client.get(f"/api/v1/ocr/jobs/{job_id}").status_code for job_id in job_ids]

    assert statuses == [404, 200, 200]


def test_job_killed_by_a_base_exception_is_marked_failed(tmp_path, samples, monkeypatch):
    with build_client(tmp_path) as test_client:
        def crash(*_args, **_kwargs):
            raise SystemExit("worker stopped")

        monkeypatch.setattr(test_client.app.state.processor, "process", crash)
        job_id = test_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png")).json()["job_id"]

        job = wait_for_job(test_client, job_id)

    assert job["status"] == "failed"
    assert job["error"]["code"] == "INTERNAL_ERROR"


def test_job_not_accepted_by_the_executor_is_not_stored(tmp_path):
    with build_client(tmp_path) as test_client:
        jobs = test_client.app.state.jobs
        jobs._executor.shutdown()
        workdir = tmp_path / "job"
        workdir.mkdir()

        with pytest.raises(RuntimeError):
            jobs.submit(workdir, "e.png", ExtractionOptions())

        assert jobs._jobs == {}


def test_expired_jobs_are_removed(tmp_path, samples):
    with build_client(tmp_path, job_ttl_seconds=0) as short_client:
        job_id = short_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png")).json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = short_client.get(f"/api/v1/ocr/jobs/{job_id}")
            if response.status_code == 404:
                break
            time.sleep(0.05)

    assert response.status_code == 404


def test_job_upload_is_deleted_after_processing(tmp_path, samples, monkeypatch):
    import tempfile

    created = []
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: created.append(real_mkdtemp(*a, **k)) or created[-1])
    with build_client(tmp_path) as test_client:
        job_id = test_client.post("/api/v1/ocr/jobs", files=upload(samples, "english.png")).json()["job_id"]
        wait_for_job(test_client, job_id)

    import os

    assert created and not any(os.path.exists(path) for path in created)


def test_openapi_documents_the_upload_endpoints(client):
    schema = client.get("/openapi.json").json()

    for path in ("/api/v1/ocr/extract", "/api/v1/ocr/batch", "/api/v1/ocr/jobs"):
        assert "multipart/form-data" in schema["paths"][path]["post"]["requestBody"]["content"]
    assert "/api/v1/ocr/jobs/{job_id}" in schema["paths"]
    assert client.get("/docs").status_code == 200


def test_web_ui_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "OCR Platform" in response.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200
