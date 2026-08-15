# Local OCR MCP

The temporal OCR engine exposes a local, stdio MCP adapter.  The adapter only
accepts video files that already exist on disk; Bilibili acquisition, AI
subtitles, comments, and danmaku remain the responsibility of the separate
Bilibili MCP.

## Install and start

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[video,ocr,mcp]"
.\.venv\Scripts\python.exe -m temporal_ocr.mcp_server
```

The installed console entry point is equivalent:

```powershell
temporal-ocr-mcp
```

Configure the MCP client to launch the command with the project's virtual
environment.  The server uses stdio and does not open a network listener.

## Tools

- `inspect_video`: read local dimensions, FPS, duration, and codec without OCR.
- `ocr_video`: start asynchronous OCR with automatic short-video bypass and
  long-video chunk planning.
- `ocr_video_chunked`: expose the same planner while allowing explicit chunk
  controls; it processes seekable overlapping windows in parallel and merges
  boundary duplicates by time/text/geometry.
- `ocr_segment`: start an asynchronous bounded segment job.
- `benchmark_ocr`: run up to eight configurations and compare throughput.
- `get_run_status`: poll a job without loading its event list.
- `get_run_result`: return the profile and artifact paths; events are opt-in and
  capped with `max_events`.
- `list_runs`: list jobs known by the current server process.
- `cleanup_run`: remove one completed run after `confirm=true`.

All OCR tools that process video accept an optional `exclude_regions` argument.
Use normalized rectangles in the form `[left, top, right, bottom]`, with every
coordinate between 0 and 1. For example, a bottom-right watermark can be
passed as `[[0.78, 0.88, 1.0, 1.0]]`. The engine masks that area from motion and
change analysis and drops detector observations overlapping it. The same
regions are reused by every long-video chunk.

The default OCR request samples at 1 FPS and caps decoded frames at 1280 px.
Pass `config_path`, `sample_fps`, and `max_width` explicitly for a benchmark.
The high-speed project config is `benchmarks/config-fast.json`.

`ocr_video_chunked` automatically targets roughly 180-second chunks and keeps
a 4-second overlap. Inputs at or below 300 seconds bypass chunking and use the
continuous engine; pass `chunk_sec` explicitly when you want to override that
choice. `workers` controls the number of parallel chunks (up to 8), and
`ocr_threads_per_worker` controls ONNX Runtime threads inside each worker.
The engine uses logical PyAV seeks and does not create re-encoded temporary
videos. Each chunked job writes per-chunk artifacts under `parts/` and a merged
`events.jsonl` at the job root.

## Artifacts and environment

Each run writes `run.json` and `events.jsonl`.  By default they are placed under
`artifacts/ocr-mcp/<job-id>`.  Set `TEMPORAL_OCR_MCP_OUTPUT_ROOT` to choose a
different root and `TEMPORAL_OCR_MCP_WORKERS` to permit up to four concurrent
jobs.  One worker is the default because each ONNX Runtime OCR instance already
uses its own bounded CPU thread pool.

The MCP returns compact summaries and absolute artifact paths rather than
embedding the complete event stream in the conversation.  Use
`get_run_result(include_events=true, max_events=N)` only when event details are
needed.
