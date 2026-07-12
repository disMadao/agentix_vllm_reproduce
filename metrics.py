from __future__ import annotations

from statistics import mean
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def build_summary(programs: list[dict[str, Any]], started_at: float, ended_at: float) -> dict[str, Any]:
    latencies = [float(p["program_latency_sec"]) for p in programs if p.get("status") == "ok"]
    total_calls = sum(int(p.get("num_calls", 0)) for p in programs)
    total_output_tokens = sum(int(p.get("output_tokens", 0)) for p in programs)
    elapsed = max(ended_at - started_at, 1e-9)
    return {
        "num_programs": len(programs),
        "num_ok": sum(1 for p in programs if p.get("status") == "ok"),
        "total_calls": total_calls,
        "total_output_tokens": total_output_tokens,
        "elapsed_sec": elapsed,
        "avg_program_latency_sec": mean(latencies) if latencies else 0.0,
        "p50_program_latency_sec": percentile(latencies, 0.50),
        "p95_program_latency_sec": percentile(latencies, 0.95),
        "p99_program_latency_sec": percentile(latencies, 0.99),
        "throughput_program_per_sec": len(programs) / elapsed,
        "throughput_call_per_sec": total_calls / elapsed,
        "throughput_output_tok_per_sec": total_output_tokens / elapsed,
    }
