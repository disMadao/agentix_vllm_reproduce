from __future__ import annotations

import argparse
import json

from agentix_app.chatbot import ChatbotAgent
from agentix_app.nanovllm_client import DEFAULT_QWEN_MODEL, NanoVLLMChatClient
from agentix_app.react_agent import ReActAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simple Agentix app-layer demos on nano-vllm.")
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--scheduler-policy", default="mlfq_plas", choices=("fcfs", "plas", "mlfq_plas"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    client = NanoVLLMChatClient(
        model_path=args.model_path,
        scheduler_policy=args.scheduler_policy,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    chatbot = ChatbotAgent(client, program_id="demo-chatbot")
    print("## Chatbot")
    print(chatbot.ask("你好，用一句话介绍一下你自己。", max_tokens=args.max_tokens))

    react = ReActAgent(client, program_id="demo-react", max_steps=2)
    print("\n## ReAct")
    print(react.run("请计算 12 * 7，并给出最终答案。", max_tokens=args.max_tokens))

    print("\n## Process Table")
    print(json.dumps(client.process_table_snapshot(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
