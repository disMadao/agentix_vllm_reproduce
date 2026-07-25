from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in __path__:
    __path__.append(str(_REPO_ROOT))

__all__ = ["ChatbotAgent", "NanoVLLMChatClient", "ReActAgent"]


def __getattr__(name: str) -> Any:
    if name == "ChatbotAgent":
        return import_module("agentix_app.chatbot").ChatbotAgent
    if name == "NanoVLLMChatClient":
        return import_module("agentix_app.nanovllm_client").NanoVLLMChatClient
    if name == "ReActAgent":
        return import_module("agentix_app.react_agent").ReActAgent
    raise AttributeError(name)
