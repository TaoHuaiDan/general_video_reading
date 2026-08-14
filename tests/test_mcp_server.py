from __future__ import annotations

from temporal_ocr.mcp_server import _JobStore


def test_job_store_tracks_async_completion(tmp_path) -> None:
    store = _JobStore()
    try:
        output_dir = tmp_path / "run"
        job = store.submit("test", output_dir, lambda: {"ok": True})
        future = store.get(job["job_id"]).future
        assert future is not None
        future.result(timeout=2)
        status = store.public(job["job_id"])
        assert status["status"] == "completed"
        assert status["output_dir"] == str(output_dir)
    finally:
        store.shutdown()
