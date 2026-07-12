from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agentix_app.datasets import ProgramSpec, load_bfcl, load_sharegpt
from agentix_app.metrics import build_summary
from agentix_app.nanovllm_client import DEFAULT_QWEN_MODEL, NanoVLLMChatClient
from nanovllm import SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Agentix dataset workloads on nano-vllm.")
    parser.add_argument("--dataset", choices=("sharegpt", "bfcl"), required=True)
    parser.add_argument("--input", required=True, help="Local JSON/JSONL dataset path.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--mode", choices=("replay",), default="replay")
    parser.add_argument("--scheduler-policy", choices=("fcfs", "plas", "mlfq_plas"), default="mlfq_plas")
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--replay-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True, help="Program-level JSONL output path.")
    parser.add_argument("--summary-out", default=None, help="Summary JSON output path.")
    args = parser.parse_args()

    programs = load_programs(args)
    if not programs:
        raise SystemExit("no runnable programs parsed from dataset")

    result = run_replay(programs, args)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in result["programs"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_out or f"{out_path}.summary.json").expanduser().resolve()
    summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "summary_out": str(summary_path), **result["summary"]}, ensure_ascii=False))
    return 0


def load_programs(args) -> list[ProgramSpec]:
    if args.dataset == "sharegpt":
        return load_sharegpt(args.input, args.limit, args.max_tokens)
    return load_bfcl(args.input, args.limit, args.max_tokens, replay_steps=args.replay_steps)


def run_replay(programs: list[ProgramSpec], args) -> dict[str, Any]:
    client = NanoVLLMChatClient(
        model_path=args.model_path,
        scheduler_policy=args.scheduler_policy,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
    )
    llm = client.llm
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    seq_to_call: dict[int, dict[str, Any]] = {}
    program_rows: dict[str, dict[str, Any]] = {}
    started_at = time.perf_counter()
    for program in programs:
        program_rows[program.program_id] = {
            "program_id": program.program_id,
            "dataset": program.dataset,
            "kind": program.kind,
            "scheduler_policy": args.scheduler_policy,
            "num_calls": len(program.calls),
            "num_finished_calls": 0,
            "output_tokens": 0,
            "started_at": started_at,
            "ended_at": None,
            "program_latency_sec": 0.0,
            "status": "ok",
        }
        for call in program.calls:
            prompt = client._render_messages(call.messages)
            seq_id = llm.add_request(
                prompt,
                sampling_params,
                program_id=program.program_id,
                call_id=call.call_id,
                thread_id=call.thread_id,
            )
            seq_to_call[seq_id] = {
                "program_id": program.program_id,
                "call_id": call.call_id,
                "thread_id": call.thread_id,
                "started_at": started_at,
            }

    while not llm.is_finished():
        outputs, _ = llm.step()
        now = time.perf_counter()
        for seq_id, token_ids in outputs:
            info = seq_to_call[seq_id]
            row = program_rows[info["program_id"]]
            row["num_finished_calls"] += 1
            row["output_tokens"] += len(token_ids)
            row["ended_at"] = now
            row["program_latency_sec"] = now - row["started_at"]

    ended_at = time.perf_counter()
    snapshot = llm.process_table_snapshot()
    for program_id, row in program_rows.items():
        proc = snapshot.get(program_id, {})
        row["service_time"] = proc.get("service_time", 0.0)
        row["wait_time"] = proc.get("wait_time", 0.0)
        row["active_call_ids"] = proc.get("active_call_ids", [])
        if row["ended_at"] is None:
            row["ended_at"] = ended_at
            row["program_latency_sec"] = ended_at - row["started_at"]
            row["status"] = "not_finished"

    rows = list(program_rows.values())
    summary = build_summary(rows, started_at, ended_at)
    summary.update(
        {
            "dataset": args.dataset,
            "mode": args.mode,
            "scheduler_policy": args.scheduler_policy,
            "model_path": args.model_path,
            "max_tokens": args.max_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "ignore_eos": args.ignore_eos,
        }
    )
    return {"programs": rows, "summary": summary}


if __name__ == "__main__":
    raise SystemExit(main())
