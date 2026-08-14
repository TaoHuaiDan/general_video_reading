"""Stable local artifact writers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from temporal_ocr.types import TextEvent


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    instance = cast(Any, value)
    if hasattr(instance, "tolist"):
        return instance.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_events(path: str | Path, events: list[TextEvent]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=_json_default))
            handle.write("\n")
    return target


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return target
