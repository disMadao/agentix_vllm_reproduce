#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator, TextIO


DATASET_REPOSITORY = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILENAME}"
)
DATASET_LICENSE = "Apache-2.0"
EXPECTED_RAW_BYTES = 672_837_942
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PARSE_CHUNK_CHARS = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and reproducibly sample the ShareGPT workload used by Agentix."
    )
    parser.add_argument(
        "--source",
        default=DATASET_URL,
        help="Pinned ShareGPT URL or an existing local JSON/JSONL file.",
    )
    parser.add_argument(
        "--raw-path",
        default=f"data/raw/{DATASET_FILENAME}",
        help="Cache path used when --source is a URL.",
    )
    parser.add_argument(
        "--output",
        default="data/sharegpt-real-10000-seed0.jsonl",
        help="Sampled ShareGPT JSONL output path.",
    )
    parser.add_argument("--num-programs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-calls",
        type=int,
        default=1,
        help="Minimum number of user/assistant turns required in a program.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum calls per program; non-positive values disable the limit.",
    )
    parser.add_argument(
        "--max-conversation-chars",
        type=int,
        default=0,
        help="Optional coarse conversation-size filter; zero disables it.",
    )
    parser.add_argument(
        "--stats-out",
        default=None,
        help="Statistics JSON path; defaults to <output>.stats.json.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Discard an existing cached raw file and download it again.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_programs <= 0:
        raise SystemExit("--num-programs must be positive")
    if args.min_calls <= 0:
        raise SystemExit("--min-calls must be positive")
    if args.max_calls > 0 and args.max_calls < args.min_calls:
        raise SystemExit("--max-calls must be greater than or equal to --min-calls")

    source_path = resolve_source(args.source, Path(args.raw_path), args.force_download)
    sampled, scan_stats = sample_programs(
        source_path,
        num_programs=args.num_programs,
        seed=args.seed,
        min_calls=args.min_calls,
        max_calls=args.max_calls,
        max_conversation_chars=args.max_conversation_chars,
    )
    output_path = Path(args.output).expanduser().resolve()
    write_jsonl_atomic(output_path, sampled)

    sample_call_counts = [count_calls(row) for row in sampled]
    stats = {
        "dataset": "sharegpt",
        "source_repository": DATASET_REPOSITORY,
        "source_revision": DATASET_REVISION,
        "source_file": DATASET_FILENAME,
        "source": args.source,
        "license": DATASET_LICENSE,
        "seed": args.seed,
        "requested_programs": args.num_programs,
        "sampled_programs": len(sampled),
        "min_calls_filter": args.min_calls,
        "max_calls_filter": args.max_calls,
        "max_conversation_chars_filter": args.max_conversation_chars,
        **scan_stats,
        "sample_total_calls": sum(sample_call_counts),
        "sample_mean_calls_per_program": statistics.mean(sample_call_counts),
        "sample_median_calls_per_program": statistics.median(sample_call_counts),
        "sample_min_calls_per_program": min(sample_call_counts),
        "sample_max_calls_per_program": max(sample_call_counts),
        "sample_sha256": sha256_file(output_path),
    }
    stats_path = Path(args.stats_out or f"{output_path}.stats.json").expanduser().resolve()
    write_json_atomic(stats_path, stats)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "stats_out": str(stats_path),
                **stats,
            },
            ensure_ascii=False,
        )
    )
    return 0


def resolve_source(source: str, raw_path: Path, force_download: bool) -> Path:
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"dataset source does not exist: {path}")
        return path

    raw_path = raw_path.expanduser().resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = EXPECTED_RAW_BYTES if source == DATASET_URL else None
    if force_download:
        raw_path.unlink(missing_ok=True)
        raw_path.with_name(f"{raw_path.name}.part").unlink(missing_ok=True)
    if raw_path.is_file():
        if expected_bytes is None or raw_path.stat().st_size == expected_bytes:
            print(f"Using cached dataset: {raw_path}", file=sys.stderr)
            return raw_path
        raise SystemExit(
            f"cached dataset has unexpected size: {raw_path} "
            f"({raw_path.stat().st_size} bytes, expected {expected_bytes}); "
            "rerun with --force-download"
        )

    download_with_resume(source, raw_path, expected_bytes)
    return raw_path


def download_with_resume(url: str, destination: Path, expected_bytes: int | None) -> None:
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "agentix-vllm-reproduce/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming download at {format_bytes(offset)}", file=sys.stderr)
    else:
        print(f"Downloading {url}", file=sys.stderr)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        downloaded = offset
        next_report = downloaded + 64 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"Downloaded {format_bytes(downloaded)}", file=sys.stderr)
                    next_report = downloaded + 64 * 1024 * 1024

    actual_bytes = partial.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise SystemExit(
            f"download is incomplete: {actual_bytes} bytes, expected {expected_bytes}; "
            "rerun the same command to resume"
        )
    os.replace(partial, destination)
    print(f"Downloaded dataset to {destination} ({format_bytes(actual_bytes)})", file=sys.stderr)


def sample_programs(
    source_path: Path,
    num_programs: int,
    seed: int,
    min_calls: int,
    max_calls: int,
    max_conversation_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    scanned = 0
    valid = 0
    total_valid_calls = 0
    min_valid_calls: int | None = None
    max_valid_calls = 0

    for row in iter_json_records(source_path):
        scanned += 1
        calls = count_calls(row)
        if calls < min_calls or (max_calls > 0 and calls > max_calls):
            continue
        if max_conversation_chars > 0 and conversation_chars(row) > max_conversation_chars:
            continue

        valid += 1
        total_valid_calls += calls
        min_valid_calls = calls if min_valid_calls is None else min(min_valid_calls, calls)
        max_valid_calls = max(max_valid_calls, calls)
        if len(sample) < num_programs:
            sample.append(row)
        else:
            replacement_index = rng.randrange(valid)
            if replacement_index < num_programs:
                sample[replacement_index] = row

    if valid < num_programs:
        raise SystemExit(
            f"only {valid} valid programs found after scanning {scanned}; "
            f"cannot sample {num_programs}"
        )
    rng.shuffle(sample)
    return sample, {
        "records_scanned": scanned,
        "valid_programs": valid,
        "valid_total_calls": total_valid_calls,
        "valid_mean_calls_per_program": total_valid_calls / valid,
        "valid_min_calls_per_program": min_valid_calls,
        "valid_max_calls_per_program": max_valid_calls,
    }


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        first = read_first_non_whitespace(source)
        if first == "[":
            yield from iter_json_array(source)
            return
        source.seek(0)
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"record on line {line_number} is not an object")
            yield row


def read_first_non_whitespace(source: TextIO) -> str:
    while True:
        char = source.read(1)
        if not char:
            raise ValueError("dataset is empty")
        if not char.isspace():
            return char


def iter_json_array(source: TextIO) -> Iterator[dict[str, Any]]:
    current: list[str] | None = None
    depth = 0
    in_string = False
    escaped = False
    closed = False

    while True:
        chunk = source.read(PARSE_CHUNK_CHARS)
        if not chunk:
            break
        for char in chunk:
            if current is None:
                if char.isspace() or char == ",":
                    continue
                if char == "]":
                    closed = True
                    break
                if char != "{":
                    raise ValueError(f"expected a JSON object in top-level array, found {char!r}")
                current = [char]
                depth = 1
                in_string = False
                escaped = False
                continue

            current.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    row = json.loads("".join(current))
                    if not isinstance(row, dict):
                        raise ValueError("top-level array record is not an object")
                    yield row
                    current = None
        if closed:
            break

    if current is not None:
        raise ValueError("dataset ended in the middle of a JSON object")
    if not closed:
        raise ValueError("dataset JSON array is missing its closing bracket")


def count_calls(row: dict[str, Any]) -> int:
    conversations = extract_conversations(row)
    pending_user = False
    calls = 0
    for message in conversations:
        role = normalize_role(message.get("from") or message.get("role") or message.get("speaker"))
        content = clean_text(message.get("value") or message.get("content") or message.get("text"))
        if not content:
            continue
        if role == "user":
            pending_user = True
        elif role == "assistant" and pending_user:
            calls += 1
            pending_user = False
    if pending_user:
        calls += 1
    return calls


def conversation_chars(row: dict[str, Any]) -> int:
    return sum(
        len(clean_text(message.get("value") or message.get("content") or message.get("text")))
        for message in extract_conversations(row)
    )


def extract_conversations(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("conversations", "messages", "conversation"):
        value = row.get(key)
        if isinstance(value, list):
            return [message for message in value if isinstance(message, dict)]
    return []


def normalize_role(role: Any) -> str:
    role = clean_text(role).lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant", "bot"):
        return "assistant"
    return role


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix.lower() == ".gz":
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=0,
            ) as compressed_output:
                with io.TextIOWrapper(compressed_output, encoding="utf-8") as output:
                    _write_jsonl(output, rows)
    else:
        with temporary.open("w", encoding="utf-8") as output:
            _write_jsonl(output, rows)
    os.replace(temporary, path)


def _write_jsonl(output: TextIO, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
