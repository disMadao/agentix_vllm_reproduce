from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_NANO_ROOT = Path(__file__).resolve().parents[1] / "nano-vllm"
if str(_NANO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANO_ROOT))

from nanovllm import LLM, SamplingParams  # noqa: E402

from agentix_app.openai_types import ChatCompletionRequest, ChatMessage


DEFAULT_QWEN_MODEL = "/home/mzj/huggingface/Qwen3-0.6B"


class NanoVLLMChatClient:
    """Small OpenAI-style chat adapter over local nano-vllm."""

    def __init__(
        self,
        model_path: str | None = None,
        scheduler_policy: str = "mlfq_plas",
        max_model_len: int = 4096,
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        enforce_eager: bool = True,
    ):
        self.model_path = os.path.expanduser(model_path or DEFAULT_QWEN_MODEL)
        self.llm = LLM(
            self.model_path,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            scheduler_policy=scheduler_policy,
        )
        self.model = Path(self.model_path).name

    def create(self, request: ChatCompletionRequest) -> dict[str, Any]:
        messages = request["messages"]
        prompt = self._render_messages(messages)
        agentix = dict(request.get("agentix") or {})
        program_id = agentix.get("program_id") or request.get("user") or f"program-{uuid.uuid4().hex[:8]}"
        call_id = agentix.get("call_id") or f"{program_id}:call-{uuid.uuid4().hex[:8]}"
        thread_id = agentix.get("thread_id") or "main"
        max_tokens = int(request.get("max_completion_tokens") or request.get("max_tokens") or 128)
        sampling_params = SamplingParams(
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
            request_metadata=[
                {
                    "program_id": program_id,
                    "call_id": call_id,
                    "thread_id": thread_id,
                }
            ],
        )
        content = outputs[0]["text"]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model") or self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "agentix": {
                "program_id": program_id,
                "call_id": call_id,
                "thread_id": thread_id,
            },
        }

    def process_table_snapshot(self):
        return self.llm.process_table_snapshot()

    def _render_messages(self, messages: list[ChatMessage]) -> str:
        hf_messages = [
            {"role": self._normalize_role(m["role"]), "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "tool"
        ]
        tokenizer = self.llm.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in hf_messages) + "\nassistant:"

    def _normalize_role(self, role: str) -> str:
        if role == "developer":
            return "system"
        if role in ("system", "user", "assistant"):
            return role
        return "user"
