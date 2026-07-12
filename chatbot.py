from __future__ import annotations

from agentix_app.nanovllm_client import NanoVLLMChatClient
from agentix_app.openai_types import ChatMessage


class ChatbotAgent:

    def __init__(
        self,
        client: NanoVLLMChatClient,
        program_id: str,
        system_prompt: str = "你是一个简洁、可靠的中文助手。",
    ):
        self.client = client
        self.program_id = program_id
        self.messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
        ]
        self.turn = 0

    def ask(self, user_text: str, max_tokens: int = 128) -> str:
        self.turn += 1
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.create(
            {
                "model": self.client.model,
                "messages": self.messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "user": self.program_id,
                "agentix": {
                    "program_id": self.program_id,
                    "call_id": f"{self.program_id}:chat-turn-{self.turn}",
                    "thread_id": "main",
                },
            }
        )
        content = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": content})
        return content
