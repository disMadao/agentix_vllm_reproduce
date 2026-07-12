from __future__ import annotations

import re

from agentix_app.nanovllm_client import NanoVLLMChatClient
from agentix_app.openai_types import ChatMessage
from agentix_app.tools import TOOLS


REACT_SYSTEM_PROMPT = """你是一个 ReAct Agent。你可以使用工具解决问题。

请严格使用下面格式之一：

Thought: 你的思考
Action: calculator | lookup_stub | time_stub | echo
Action Input: 工具输入

或者：

Thought: 你的思考
Final: 最终答案
"""


class ReActAgent:

    def __init__(self, client: NanoVLLMChatClient, program_id: str, max_steps: int = 3):
        self.client = client
        self.program_id = program_id
        self.max_steps = max_steps
        self.messages: list[ChatMessage] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
        ]

    def run(self, task: str, max_tokens: int = 128) -> str:
        self.messages.append({"role": "user", "content": task})
        last_content = ""
        for step in range(1, self.max_steps + 1):
            response = self.client.create(
                {
                    "model": self.client.model,
                    "messages": self.messages,
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "user": self.program_id,
                    "agentix": {
                        "program_id": self.program_id,
                        "call_id": f"{self.program_id}:react-step-{step}",
                        "thread_id": "main",
                    },
                }
            )
            last_content = response["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": last_content})
            final = self._parse_final(last_content)
            if final:
                return final
            action = self._parse_action(last_content)
            if action is None:
                return last_content
            name, action_input = action
            observation = self._run_tool(name, action_input)
            self.messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n请继续，必要时给出 Final。",
                }
            )
        return last_content

    def _run_tool(self, name: str, action_input: str) -> str:
        tool = TOOLS.get(name)
        if tool is None:
            return f"unknown tool: {name}"
        return tool(action_input)

    def _parse_final(self, text: str) -> str | None:
        match = re.search(r"Final\s*:\s*(.*)", text, flags=re.I | re.S)
        if not match:
            return None
        return match.group(1).strip()

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        action = re.search(r"Action\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)", text)
        action_input = re.search(r"Action Input\s*:\s*(.*)", text, flags=re.S)
        if not action or not action_input:
            return None
        return action.group(1).strip(), action_input.group(1).strip()
