from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentix_app.openai_types import ChatMessage


@dataclass
class CallSpec:
    call_id: str
    messages: list[ChatMessage]
    max_tokens: int
    thread_id: str = "main"
    reference: str | None = None


@dataclass
class ProgramSpec:
    program_id: str
    dataset: str
    kind: str
    calls: list[CallSpec]
    raw: dict[str, Any] = field(default_factory=dict)


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as source:
            text = source.read()
        data_suffix = path.with_suffix("").suffix.lower()
    else:
        text = path.read_text(encoding="utf-8")
        data_suffix = path.suffix.lower()
    if data_suffix == ".jsonl":
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "instances", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    raise ValueError(f"unsupported dataset shape in {path}")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)
