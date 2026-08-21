from __future__ import annotations

import json

from temporal_ocr import mcp_server
from temporal_ocr.mcp_server import _JobStore, get_run_result


def _write_run_artifacts(run_dir, row_count: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    rows = [{"event_id": index, "text_normalized": f"text-{index}"} for index in range(1, row_count + 1)]
    events_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(json.dumps({"profile": {}, "input": {}}), encoding="utf-8")


def _submit_completed_run(store: _JobStore, output_dir) -> str:
    def task():
        return {
            "events": str(output_dir / "events.jsonl"),
            "metadata": str(output_dir / "run.json"),
            "event_count": 5,
        }

    job = store.submit("test", output_dir, task)
    future = store.get(job["job_id"]).future
    assert future is not None
    future.result(timeout=2)
    return job["job_id"]


def test_get_run_result_streams_bounded_event_preview(tmp_path, monkeypatch) -> None:
    store = _JobStore(output_root=tmp_path / "root")
    monkeypatch.setattr(mcp_server, "_JOBS", store)
    try:
        run_dir = tmp_path / "run"
        _write_run_artifacts(run_dir, row_count=5)
        job_id = _submit_completed_run(store, run_dir)

        payload = get_run_result(job_id, include_events=True, max_events=2, event_offset=1)

        assert [event["event_id"] for event in payload["events"]] == [2, 3]
        assert payload["events_offset"] == 1
        assert payload["events_truncated"] is True

        tail = get_run_result(job_id, include_events=True, max_events=4, event_offset=3)

        assert [event["event_id"] for event in tail["events"]] == [4, 5]
        assert tail["events_truncated"] is False

        beyond = get_run_result(job_id, include_events=True, max_events=2, event_offset=10)

        assert beyond["events"] == []
        assert beyond["events_truncated"] is False
    finally:
        store.shutdown()


def test_job_store_keeps_only_recent_finished_jobs(tmp_path) -> None:
    store = _JobStore(output_root=tmp_path / "root", max_finished_jobs=2)
    try:
        job_ids = [
            _submit_completed_run(store, tmp_path / f"run-{index}") for index in range(3)
        ]

        assert len(store._jobs) == 2
        assert job_ids[0] not in store._jobs
        assert job_ids[1] in store._jobs
        assert job_ids[2] in store._jobs
    finally:
        store.shutdown()


def test_cleanup_run_removes_directory_and_registry_entry(tmp_path, monkeypatch) -> None:
    import pytest

    store = _JobStore(output_root=tmp_path / "root")
    monkeypatch.setattr(mcp_server, "_JOBS", store)
    try:
        run_dir = tmp_path / "root" / "run"
        _write_run_artifacts(run_dir, row_count=1)
        job_id = _submit_completed_run(store, run_dir)

        result = mcp_server.cleanup_run(job_id, confirm=True)

        assert result["removed"] == str(run_dir)
        assert not run_dir.exists()
        with pytest.raises(KeyError):
            store.get(job_id)
    finally:
        store.shutdown()
