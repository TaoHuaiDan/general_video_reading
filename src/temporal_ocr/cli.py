"""Command line utilities for configuration, diagnostics and evaluation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from temporal_ocr.config import EngineConfig
from temporal_ocr.metrics import evaluate_events, event_from_dict


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _take_until_end(source: Any, end_sec: float | None) -> Any:
    if end_sec is None:
        yield from source
        return
    for frame in source:
        if frame.timestamp > end_sec:
            break
        yield frame


def cmd_show_config(args: argparse.Namespace) -> int:
    config = EngineConfig()
    if args.output:
        config.save(args.output)
        print(str(Path(args.output).resolve()))
    else:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    packages = ["numpy", "cv2", "av", "rapidocr", "onnxruntime"]
    report: dict[str, Any] = {
        name: bool(importlib.util.find_spec(name)) for name in packages
    }
    if report["onnxruntime"]:
        ort = importlib.import_module("onnxruntime")

        report["onnxruntime_version"] = ort.__version__
        report["onnxruntime_providers"] = ort.get_available_providers()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    reference = [event_from_dict(item) for item in _read_jsonl(args.reference)]
    predicted = [event_from_dict(item) for item in _read_jsonl(args.predicted)]
    report = evaluate_events(
        reference,
        predicted,
        video_sec=args.video_sec,
        wall_sec=args.wall_sec,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from temporal_ocr.engine import TemporalOCREngine
    from temporal_ocr.output import write_events, write_run_metadata
    from temporal_ocr.rapidocr_backend import (
        RapidOCRBatchRecognizer,
        RapidOCRDetector,
        RapidOCRRuntime,
    )
    from temporal_ocr.sources import PyAVFrameSource

    config = EngineConfig.load(args.config) if args.config else EngineConfig()
    source = PyAVFrameSource(
        args.video,
        thread_type=args.thread_type,
        sample_fps=args.sample_fps,
        max_width=args.max_width,
    )
    frames = _take_until_end(source, args.end_sec)
    runtime = RapidOCRRuntime(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": config.ocr.intra_op_num_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": config.ocr.inter_op_num_threads,
        }
    )
    engine = TemporalOCREngine(
        RapidOCRDetector(runtime=runtime),
        RapidOCRBatchRecognizer(runtime=runtime),
        config=config,
    )
    result = engine.run(frames)
    output_dir = Path(args.output).resolve()
    events_path = write_events(output_dir / "events.jsonl", result.events)
    metadata = result.to_dict()
    metadata.pop("events", None)
    metadata["input"] = {
        "video": str(Path(args.video).resolve()),
        "end_sec": args.end_sec,
        "sample_fps": args.sample_fps,
        "max_width": args.max_width,
        "thread_type": args.thread_type,
    }
    metadata_path = write_run_metadata(output_dir / "run.json", metadata)
    print(
        json.dumps(
            {
                "events": str(events_path),
                "metadata": str(metadata_path),
                "event_count": len(result.events),
                "video_realtime": result.profile.video_realtime,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporal-ocr")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config = subcommands.add_parser("show-config", help="show or write default configuration")
    config.add_argument("--output")
    config.set_defaults(handler=cmd_show_config)

    doctor = subcommands.add_parser("doctor", help="inspect optional runtime backends")
    doctor.set_defaults(handler=cmd_doctor)

    evaluate = subcommands.add_parser("evaluate", help="evaluate predicted text events")
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--predicted", required=True)
    evaluate.add_argument("--video-sec", type=float, default=0.0)
    evaluate.add_argument("--wall-sec", type=float, default=0.0)
    evaluate.set_defaults(handler=cmd_evaluate)

    run = subcommands.add_parser("run", help="run the RapidOCR baseline on a local video")
    run.add_argument("video")
    run.add_argument("--output", required=True)
    run.add_argument("--config")
    run.add_argument("--end-sec", type=float)
    run.add_argument("--thread-type", choices=["AUTO", "FRAME", "SLICE"], default="AUTO")
    run.add_argument(
        "--sample-fps",
        type=float,
        help="decode only approximately this many frames per second",
    )
    run.add_argument(
        "--max-width",
        type=int,
        help="resize decoded frames to at most this width before OCR",
    )
    run.set_defaults(handler=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
