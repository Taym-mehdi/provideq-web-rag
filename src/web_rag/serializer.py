from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def to_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(to_serializable(value), ensure_ascii=False, indent=indent)


def save_json(value: Any, path: str | Path, *, indent: int = 2) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_json(value, indent=indent), encoding="utf-8")
    return output_path


def save_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_evidence_outputs(evidence: Any, output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "json": save_json(evidence, directory / "evidence.json"),
        "context": save_text(evidence.context_text, directory / "context.txt"),
    }
