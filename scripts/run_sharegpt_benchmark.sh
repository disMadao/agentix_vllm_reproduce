#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_ROOT/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi
PREPARE_PROGRAMS="${PREPARE_PROGRAMS:-10000}"
DATASET_SEED="${DATASET_SEED:-0}"
DATASET="${DATASET:-$REPO_ROOT/data/sharegpt-real-${PREPARE_PROGRAMS}-seed${DATASET_SEED}.jsonl}"
LIMIT="${LIMIT:-1000}"
ARRIVAL_RATES="${ARRIVAL_RATES:-2}"
ARRIVAL_SEEDS="${ARRIVAL_SEEDS:-0}"
SCHEDULER_POLICIES="${SCHEDULER_POLICIES:-fcfs plas mlfq_plas}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/sharegpt-real}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  MODEL=/path/to/Qwen3-0.6B ./scripts/run_sharegpt_benchmark.sh

This prepares the real ShareGPT sample when DATASET does not exist, then runs
fcfs, plas, and mlfq_plas through scripts/run_benchmark.sh.

Common optional env:
  PREPARE_PROGRAMS=10000
  DATASET_SEED=0
  DATASET=data/sharegpt-real-10000-seed0.jsonl
  LIMIT=1000
  ARRIVAL_RATES="1 2 3"
  ARRIVAL_SEEDS="0 1 2"
  SCHEDULER_POLICIES="fcfs plas mlfq_plas"
  OUT_DIR=results/sharegpt-real
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="${MODEL:-${MODEL_PATH:-${1:-}}}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required." >&2
  usage
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 2
fi
if [[ ! -d "$WORKSPACE_DIR/nano-vllm" ]]; then
  echo "nano-vllm must be a sibling of this repository: $WORKSPACE_DIR/nano-vllm" >&2
  exit 2
fi

if [[ ! -f "$DATASET" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_sharegpt.py" \
    --raw-path "$REPO_ROOT/data/raw/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --output "$DATASET" \
    --num-programs "$PREPARE_PROGRAMS" \
    --seed "$DATASET_SEED"
fi

for arrival_rate in $ARRIVAL_RATES; do
  DATASET_TYPE=sharegpt \
  DATASET="$DATASET" \
  MODEL="$MODEL" \
  LIMIT="$LIMIT" \
  ARRIVAL_RATE="$arrival_rate" \
  ARRIVAL_SEEDS="$ARRIVAL_SEEDS" \
  SCHEDULER_POLICIES="$SCHEDULER_POLICIES" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_TOKENS="$MAX_TOKENS" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  OUTPUT_LENGTH_MODE=reference \
  TEMPERATURE=0 \
  SHUFFLE_PROGRAMS=1 \
  IGNORE_EOS=1 \
  SKIP_OVERLONG_PROGRAMS=1 \
  RESULT_PREFIX=sharegpt-real \
  OUT_DIR="$OUT_DIR" \
    "$SCRIPT_DIR/run_benchmark.sh"
done

echo "Benchmark complete. Results: $OUT_DIR"
