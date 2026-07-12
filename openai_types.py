from typing import Any, Literal, TypedDict


Role = Literal["system", "user", "assistant", "tool", "developer"]


class ChatMessage(TypedDict, total=False):
    role: Role
    content: str
    name: str
    tool_call_id: str


class AgentixMetadata(TypedDict, total=False):
    program_id: str
    call_id: str
    thread_id: str


class ChatCompletionRequest(TypedDict, total=False):
    model: str
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int
    max_completion_tokens: int
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    metadata: dict[str, str]
    user: str
    agentix: AgentixMetadata
