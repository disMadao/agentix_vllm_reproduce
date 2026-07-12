from __future__ import annotations

from typing import Any

from agentix_app.datasets.common import CallSpec, ProgramSpec, clean_text, load_json_or_jsonl
from agentix_app.react_agent import REACT_SYSTEM_PROMPT


def load_bfcl(path: str, limit: int | None, max_tokens: int, replay_steps: int = 3) -> list[ProgramSpec]:
    rows = load_json_or_jsonl(path)
    programs: list[ProgramSpec] = []
    for idx, row in enumerate(rows[:limit] if limit else rows):
        sample_id = clean_text(row.get("id") or row.get("question_id") or row.get("name") or idx)
        question = _extract_question(row)
        tools = _extract_tools(row)
        calls = _build_replay_calls(sample_id, question, tools, max_tokens, replay_steps)
        programs.append(
            ProgramSpec(
                program_id=f"bfcl:{sample_id}",
                dataset="bfcl",
                kind="react",
                calls=calls,
                raw=row,
            )
        )
    return programs


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "prompt", "user_prompt", "query", "instruction"):
        value = clean_text(row.get(key))
        if value:
            return value
    turns = row.get("turns") or row.get("messages")
    if isinstance(turns, list):
        chunks = []
        for turn in turns:
            if isinstance(turn, dict):
                content = clean_text(turn.get("content") or turn.get("value"))
            else:
                content = clean_text(turn)
            if content:
                chunks.append(content)
        if chunks:
            return "\n".join(chunks)
    return clean_text(row)


def _extract_tools(row: dict[str, Any]) -> str:
    for key in ("tools", "functions", "function", "function_list", "apis"):
        value = row.get(key)
        if value:
            return clean_text(value)
    return "calculator(expression), lookup_stub(query), time_stub(), echo(text)"


def _build_replay_calls(
    sample_id: str,
    question: str,
    tools: str,
    max_tokens: int,
    replay_steps: int,
) -> list[CallSpec]:
    system = f"{REACT_SYSTEM_PROMPT}\n\n可用工具定义：\n{tools}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    calls: list[CallSpec] = []
    for step in range(1, max(1, replay_steps) + 1):
        if step > 1:
            messages = messages + [
                {
                    "role": "user",
                    "content": f"Observation: scripted observation for step {step - 1}. 请继续。",
                }
            ]
        calls.append(
            CallSpec(
                call_id=f"bfcl:{sample_id}:step-{step}",
                messages=list(messages),
                max_tokens=max_tokens,
            )
        )
    return calls
