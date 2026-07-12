from __future__ import annotations

from typing import Any

from agentix_app.datasets.common import CallSpec, ProgramSpec, clean_text, load_json_or_jsonl
from agentix_app.openai_types import ChatMessage


def load_sharegpt(path: str, limit: int | None, max_tokens: int, use_reference_history: bool = True) -> list[ProgramSpec]:
    rows = load_json_or_jsonl(path)
    programs: list[ProgramSpec] = []
    for idx, row in enumerate(rows[:limit] if limit else rows):
        sample_id = clean_text(row.get("id") or row.get("conversation_id") or idx)
        conversations = _extract_conversations(row)
        calls = _build_calls(sample_id, conversations, max_tokens, use_reference_history)
        if not calls:
            continue
        programs.append(
            ProgramSpec(
                program_id=f"sharegpt:{sample_id}",
                dataset="sharegpt",
                kind="chatbot",
                calls=calls,
                raw=row,
            )
        )
    return programs


def _extract_conversations(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("conversations", "messages", "conversation"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def _build_calls(
    sample_id: str,
    conversations: list[dict[str, Any]],
    max_tokens: int,
    use_reference_history: bool,
) -> list[CallSpec]:
    history: list[ChatMessage] = [
        {"role": "system", "content": "你是一个简洁、可靠的中文助手。"},
    ]
    calls: list[CallSpec] = []
    pending_user = False
    turn = 0
    for msg in conversations:
        role = _normalize_role(msg.get("from") or msg.get("role") or msg.get("speaker"))
        content = clean_text(msg.get("value") or msg.get("content") or msg.get("text"))
        if not content:
            continue
        if role == "user":
            history.append({"role": "user", "content": content})
            pending_user = True
            continue
        if role == "assistant" and pending_user:
            turn += 1
            calls.append(
                CallSpec(
                    call_id=f"sharegpt:{sample_id}:turn-{turn}",
                    messages=list(history),
                    max_tokens=max_tokens,
                    reference=content,
                )
            )
            if use_reference_history:
                history.append({"role": "assistant", "content": content})
            pending_user = False
    if pending_user:
        turn += 1
        calls.append(
            CallSpec(
                call_id=f"sharegpt:{sample_id}:turn-{turn}",
                messages=list(history),
                max_tokens=max_tokens,
            )
        )
    return calls


def _normalize_role(role: Any) -> str:
    role = clean_text(role).lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant", "bot"):
        return "assistant"
    if role == "system":
        return "system"
    return "user"
