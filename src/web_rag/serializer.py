"""Serialization helpers for Web RAG evidence outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def evidence_pack_to_dict(evidence_pack: Any) -> dict[str, Any]:
    data = to_serializable(evidence_pack)
    if isinstance(data, dict):
        return data
    return {"value": data}


def evidence_pack_to_json(evidence_pack: Any, *, indent: int = 2) -> str:
    return json.dumps(evidence_pack_to_dict(evidence_pack), indent=indent, ensure_ascii=False)


def get_context_text(evidence_pack: Any) -> str:
    data = evidence_pack_to_dict(evidence_pack)
    return str(data.get("context_text") or data.get("context") or "")


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_serializable(data), indent=indent, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_evidence_pack(evidence_pack: Any, path: str | Path, *, indent: int = 2) -> Path:
    return save_json(evidence_pack_to_dict(evidence_pack), path, indent=indent)


def save_context_text(evidence_pack: Any, path: str | Path) -> Path:
    return save_text(get_context_text(evidence_pack), path)


def save_evidence_outputs(
    evidence_pack: Any,
    output_dir: str | Path,
    *,
    json_filename: str = "evidence.json",
    context_filename: str = "context.txt",
) -> dict[str, Path]:
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = save_evidence_pack(evidence_pack, output_directory / json_filename)
    context_path = save_context_text(evidence_pack, output_directory / context_filename)
    return {"json": json_path, "context": context_path}


# Compatibility aliases used by earlier experiments.
to_dict = evidence_pack_to_dict
save_evidence_json = save_evidence_pack
write_json = save_json


__all__ = [
    "evidence_pack_to_dict",
    "evidence_pack_to_json",
    "get_context_text",
    "save_context_text",
    "save_evidence_json",
    "save_evidence_outputs",
    "save_evidence_pack",
    "save_json",
    "save_text",
    "to_dict",
    "to_serializable",
    "write_json",
]
