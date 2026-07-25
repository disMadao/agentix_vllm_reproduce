from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentix_app.datasets import ProgramSpec, load_bfcl, load_sharegpt
from agentix_app.metrics import build_summary, percentile
from agentix_app.nanovllm_client import DEFAULT_QWEN_MODEL, NanoVLLMChatClient


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
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of runnable programs to select after shuffling and context-length filtering; non-positive means all.",
    )
    parser.add_argument("--mode", choices=("replay",), default="replay")
    parser.add_argument("--scheduler-policy", choices=("fcfs", "plas", "mlfq_plas"), default="mlfq_plas")
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8, help="Maximum output tokens for each call.")
    parser.add_argument(
        "--output-length-mode",
        choices=("fixed", "reference"),
        default="fixed",
        help="Use --max-tokens for every call, or replay each reference response's token length up to that cap.",
    )
    parser.add_argument("--replay-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="Poisson program arrival rate in programs/sec. Non-positive values submit all programs at t=0.",
    )
    parser.add_argument("--arrival-seed", type=int, default=0, help="Seed for Poisson program arrivals.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed used by --shuffle-programs before --limit. Defaults to --arrival-seed.",
    )
    parser.add_argument(
        "--shuffle-programs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle runnable programs before context filtering, --limit, and arrival assignment.",
    )
    parser.add_argument(
        "--skip-overlong-programs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip programs containing a call whose prompt plus target output exceeds --max-model-len.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load, sample, and summarize the workload without importing or running nano-vLLM.",
    )
    parser.add_argument("--out", default=None, help="Program-level JSONL output path.")
    parser.add_argument("--summary-out", default=None, help="Summary JSON output path.")
    args = parser.parse_args()
    _validate_args(parser, args)

    programs = load_programs(args)
    if not programs:
        raise SystemExit("no runnable programs parsed from dataset")
    if args.validate_only:
        ordered_programs = _order_programs(programs, args)
        sampled_programs = _select_programs_without_token_filter(ordered_programs, args)
        print(
            json.dumps(
                build_workload_report(programs, sampled_programs, args),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

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


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.replay_steps <= 0:
        parser.error("--replay-steps must be > 0")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be > 0")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be > 0")
    if args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be > 0")
    if not args.validate_only and not args.out:
        parser.error("--out is required unless --validate-only is set")


def load_programs(args) -> list[ProgramSpec]:
    if args.dataset == "sharegpt":
        return load_sharegpt(args.input, None, args.max_tokens)
    return load_bfcl(args.input, None, args.max_tokens, replay_steps=args.replay_steps)


def _sample_seed(args) -> int:
    return args.sample_seed if args.sample_seed is not None else args.arrival_seed


def _order_programs(programs: list[ProgramSpec], args) -> list[ProgramSpec]:
    programs = list(programs)
    if args.shuffle_programs:
        random.Random(_sample_seed(args)).shuffle(programs)
    return programs


def _select_programs_without_token_filter(programs: list[ProgramSpec], args) -> list[ProgramSpec]:
    limit = max(0, int(args.limit))
    if not limit:
        return list(programs)
    return list(programs[:limit])


def build_workload_report(
    all_programs: list[ProgramSpec],
    selected_programs: list[ProgramSpec],
    args,
) -> dict[str, Any]:
    call_counts = [len(program.calls) for program in selected_programs]
    prompt_chars = [
        sum(len(str(message.get("content", ""))) for message in call.messages)
        for program in selected_programs
        for call in program.calls
    ]
    max_messages_per_call = max(
        (len(call.messages) for program in selected_programs for call in program.calls),
        default=0,
    )
    return {
        "dataset": args.dataset,
        "input": str(Path(args.input).expanduser()),
        "total_runnable_programs": len(all_programs),
        "selected_programs": len(selected_programs),
        "limit": args.limit,
        "shuffle_programs": args.shuffle_programs,
        "sample_seed": _sample_seed(args),
        "arrival_rate_program_per_sec": args.arrival_rate,
        "arrival_seed": args.arrival_seed,
        "total_calls": sum(call_counts),
        "multi_call_programs": sum(1 for count in call_counts if count > 1),
        "avg_calls_per_program": _avg(call_counts),
        "p50_calls_per_program": percentile([float(v) for v in call_counts], 0.50),
        "p95_calls_per_program": percentile([float(v) for v in call_counts], 0.95),
        "max_calls_per_program": max(call_counts, default=0),
        "avg_prompt_chars_per_call": _avg(prompt_chars),
        "p95_prompt_chars_per_call": percentile([float(v) for v in prompt_chars], 0.95),
        "max_prompt_chars_per_call": max(prompt_chars, default=0),
        "max_messages_per_call": max_messages_per_call,
        "max_tokens_per_call": args.max_tokens,
        "context_filter": "not_checked_in_validate_only",
        "note": "Run without --validate-only on the inference machine to apply tokenizer-based --max-model-len filtering.",
    }


def _avg(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_replay(programs: list[ProgramSpec], args) -> dict[str, Any]:
    programs = _order_programs(programs, args)
    num_loaded_programs = len(programs)
    rng = random.Random(args.arrival_seed)

    try:
        client = NanoVLLMChatClient(
            model_path=args.model_path,
            scheduler_policy=args.scheduler_policy,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enforce_eager=True,
        )
    except ImportError as exc:
        raise SystemExit(
            "nanovllm is not importable. Use --validate-only on machines without nano-vLLM; "
            "on the benchmark machine, keep nano-vllm as a sibling directory of this repo."
        ) from exc
    llm = client.llm
    programs, num_skipped_overlong_programs = _select_runnable_programs(
        programs,
        args,
        client,
    )
    if not programs:
        raise SystemExit("no programs fit the configured model context")
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
                    args,
                    client,
                    llm,
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
            "output_length_mode": args.output_length_mode,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "ignore_eos": args.ignore_eos,
            "skip_overlong_programs": args.skip_overlong_programs,
            "num_loaded_programs": num_loaded_programs,
            "num_skipped_overlong_programs": num_skipped_overlong_programs,
            "num_selected_programs": len(programs),
            "call_submission_mode": "program_sequential",
            "arrival_rate_program_per_sec": args.arrival_rate,
            "arrival_seed": args.arrival_seed,
            "sample_seed": _sample_seed(args),
            "shuffle_programs": args.shuffle_programs,
        }
    )
    return {"programs": rows, "summary": summary}


def _select_runnable_programs(
    programs: list[ProgramSpec],
    args,
    client: NanoVLLMChatClient,
) -> tuple[list[ProgramSpec], int]:
    selected: list[ProgramSpec] = []
    skipped_overlong = 0
    limit = max(0, int(args.limit))
    for program in programs:
        if args.skip_overlong_programs and not _program_fits_model(program, args, client):
            skipped_overlong += 1
            continue
        selected.append(program)
        if limit and len(selected) >= limit:
            break
    return selected, skipped_overlong


def _program_fits_model(program: ProgramSpec, args, client: NanoVLLMChatClient) -> bool:
    tokenizer = client.llm.tokenizer
    for call in program.calls:
        prompt = client._render_messages(call.messages)
        prompt_tokens = _encode(tokenizer, prompt)
        output_tokens = _target_output_tokens(call, args, tokenizer)
        if len(prompt_tokens) + output_tokens > args.max_model_len:
            return False
    return True


def _target_output_tokens(call, args, tokenizer) -> int:
    max_tokens = max(1, int(args.max_tokens))
    if args.output_length_mode == "reference" and call.reference:
        return min(max_tokens, max(1, len(_encode(tokenizer, call.reference))))
    return max_tokens


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


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
            _submit_next_call(runtime, now, args, client, llm, seq_to_call)
            program_rows[program.program_id]["num_submitted_calls"] = runtime.next_call_idx
        else:
            program_rows[program.program_id]["ended_at"] = now
            newly_completed += 1
    return pending_idx, newly_completed


def _submit_next_call(
    runtime: ProgramRuntime,
    now: float,
    args,
    client: NanoVLLMChatClient,
    llm,
    seq_to_call: dict[int, dict[str, Any]],
) -> None:
    from nanovllm import SamplingParams

    program = runtime.program
    call = program.calls[runtime.next_call_idx]
    prompt = client._render_messages(call.messages)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=_target_output_tokens(call, args, llm.tokenizer),
        ignore_eos=args.ignore_eos,
    )
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
