from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentix_app.datasets import ProgramSpec, load_bfcl, load_sharegpt
from agentix_app.metrics import build_summary
from agentix_app.nanovllm_client import DEFAULT_QWEN_MODEL, NanoVLLMChatClient
from nanovllm import SamplingParams


@dataclass
class ProgramRuntime:
    program: ProgramSpec
    arrival_offset_sec: float
    next_call_idx: int = 0
    active_seq_id: int | None = None


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
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="Poisson program arrival rate in programs/sec. Non-positive values submit all programs at t=0.",
    )
    parser.add_argument("--arrival-seed", type=int, default=0, help="Seed for program order and Poisson arrivals.")
    parser.add_argument(
        "--shuffle-programs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle loaded programs before assigning arrivals, approximating random program sampling.",
    )
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
    programs = list(programs)
    rng = random.Random(args.arrival_seed)
    if args.shuffle_programs:
        rng.shuffle(programs)

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
    runtimes = _build_program_runtimes(programs, args.arrival_rate, rng)
    seq_to_call: dict[int, dict[str, Any]] = {}
    program_rows: dict[str, dict[str, Any]] = {}
    started_at = time.perf_counter()
    pending_idx = 0
    completed_programs = 0
    runtime_by_program_id = {runtime.program.program_id: runtime for runtime in runtimes}

    while completed_programs < len(runtimes):
        now = time.perf_counter()
        pending_idx, newly_completed = _admit_ready_programs(
            runtimes,
            pending_idx,
            started_at,
            now,
            args,
            client,
            llm,
            sampling_params,
            seq_to_call,
            program_rows,
        )
        completed_programs += newly_completed
        if completed_programs >= len(runtimes):
            break

        if not seq_to_call:
            next_arrival_at = started_at + runtimes[pending_idx].arrival_offset_sec
            time.sleep(min(max(0.0, next_arrival_at - time.perf_counter()), 0.01))
            continue

        outputs, _ = llm.step()
        now = time.perf_counter()
        for seq_id, token_ids in outputs:
            info = seq_to_call.pop(seq_id)
            runtime = runtime_by_program_id[info["program_id"]]
            runtime.active_seq_id = None
            row = program_rows[info["program_id"]]
            row["num_finished_calls"] += 1
            row["output_tokens"] += len(token_ids)
            row["program_latency_sec"] = now - row["started_at"]
            row["last_call_ended_at"] = now
            if runtime.next_call_idx < len(runtime.program.calls):
                _submit_next_call(
                    runtime,
                    now,
                    client,
                    llm,
                    sampling_params,
                    seq_to_call,
                )
                row["num_submitted_calls"] = runtime.next_call_idx
            else:
                row["ended_at"] = now
                completed_programs += 1

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
            "call_submission_mode": "program_sequential",
            "arrival_rate_program_per_sec": args.arrival_rate,
            "arrival_seed": args.arrival_seed,
            "shuffle_programs": args.shuffle_programs,
        }
    )
    return {"programs": rows, "summary": summary}


def _build_program_runtimes(
    programs: list[ProgramSpec],
    arrival_rate: float,
    rng: random.Random,
) -> list[ProgramRuntime]:
    arrival_offset_sec = 0.0
    runtimes: list[ProgramRuntime] = []
    for idx, program in enumerate(programs):
        if arrival_rate > 0.0 and idx > 0:
            arrival_offset_sec += rng.expovariate(arrival_rate)
        runtimes.append(ProgramRuntime(program=program, arrival_offset_sec=arrival_offset_sec))
    return runtimes


def _admit_ready_programs(
    runtimes: list[ProgramRuntime],
    pending_idx: int,
    started_at: float,
    now: float,
    args,
    client: NanoVLLMChatClient,
    llm,
    sampling_params: SamplingParams,
    seq_to_call: dict[int, dict[str, Any]],
    program_rows: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    newly_completed = 0
    elapsed = now - started_at
    while pending_idx < len(runtimes) and runtimes[pending_idx].arrival_offset_sec <= elapsed:
        runtime = runtimes[pending_idx]
        pending_idx += 1
        program = runtime.program
        program_rows[program.program_id] = {
            "program_id": program.program_id,
            "dataset": program.dataset,
            "kind": program.kind,
            "scheduler_policy": args.scheduler_policy,
            "call_submission_mode": "program_sequential",
            "arrival_offset_sec": runtime.arrival_offset_sec,
            "arrival_lag_sec": max(0.0, elapsed - runtime.arrival_offset_sec),
            "num_calls": len(program.calls),
            "num_submitted_calls": 0,
            "num_finished_calls": 0,
            "output_tokens": 0,
            "started_at": now,
            "ended_at": None,
            "last_call_ended_at": None,
            "program_latency_sec": 0.0,
            "status": "ok",
        }
        if program.calls:
            _submit_next_call(runtime, now, client, llm, sampling_params, seq_to_call)
            program_rows[program.program_id]["num_submitted_calls"] = runtime.next_call_idx
        else:
            program_rows[program.program_id]["ended_at"] = now
            newly_completed += 1
    return pending_idx, newly_completed


def _submit_next_call(
    runtime: ProgramRuntime,
    now: float,
    client: NanoVLLMChatClient,
    llm,
    sampling_params: SamplingParams,
    seq_to_call: dict[int, dict[str, Any]],
) -> None:
    program = runtime.program
    call = program.calls[runtime.next_call_idx]
    prompt = client._render_messages(call.messages)
    seq_id = llm.add_request(
        prompt,
        sampling_params,
        program_id=program.program_id,
        call_id=call.call_id,
        thread_id=call.thread_id,
    )
    runtime.next_call_idx += 1
    runtime.active_seq_id = seq_id
    seq_to_call[seq_id] = {
        "program_id": program.program_id,
        "call_id": call.call_id,
        "thread_id": call.thread_id,
        "submitted_at": now,
    }


if __name__ == "__main__":
    raise SystemExit(main())
