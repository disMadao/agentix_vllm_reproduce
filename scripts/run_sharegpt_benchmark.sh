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
PREPARE_PROGRAMS="${PREPARE_PROGRAMS:-5000}"
DATASET_SEED="${DATASET_SEED:-0}"
if [[ -z "${DATASET:-}" ]]; then
  if [[ "$PREPARE_PROGRAMS" == "5000" && "$DATASET_SEED" == "0" ]]; then
    DATASET="$REPO_ROOT/benchmark_data/sharegpt-real-5000-seed0.jsonl.gz"
  else
    DATASET="$REPO_ROOT/data/sharegpt-real-${PREPARE_PROGRAMS}-seed${DATASET_SEED}.jsonl.gz"
  fi
fi
LIMIT="${LIMIT:-5000}"
ARRIVAL_RATES="${ARRIVAL_RATES:-1 2 4}"
ARRIVAL_SEEDS="${ARRIVAL_SEEDS:-0 1 2}"
SCHEDULER_POLICIES="${SCHEDULER_POLICIES:-fcfs plas mlfq_plas}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/sharegpt-real}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/run_sharegpt_benchmark.sh /path/to/Qwen3-0.6B

By default this validates the bundled real ShareGPT sample, then runs 5,000
programs with fcfs, plas, and mlfq_plas at rates 1, 2, and 4 using three seeds.

Common optional env:
  PREPARE_PROGRAMS=5000
  DATASET_SEED=0
  DATASET=benchmark_data/sharegpt-real-5000-seed0.jsonl.gz
  LIMIT=5000
  ARRIVAL_RATES="1 2 4"
  ARRIVAL_SEEDS="0 1 2"
  SCHEDULER_POLICIES="fcfs plas mlfq_plas"
  RUN_PREFLIGHT=1
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

echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Programs per run: $LIMIT"
echo "Arrival rates: $ARRIVAL_RATES"
echo "Arrival seeds: $ARRIVAL_SEEDS"
echo "Scheduler policies: $SCHEDULER_POLICIES"
echo "Results: $OUT_DIR"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "Validating workload before inference..."
  DATASET_TYPE=sharegpt \
  DATASET="$DATASET" \
  LIMIT="$LIMIT" \
  ARRIVAL_RATE="${ARRIVAL_RATES%% *}" \
  ARRIVAL_SEEDS="${ARRIVAL_SEEDS%% *}" \
  MAX_TOKENS="$MAX_TOKENS" \
  OUTPUT_LENGTH_MODE=reference \
  SHUFFLE_PROGRAMS=1 \
  VALIDATE_ONLY=1 \
    "$SCRIPT_DIR/run_benchmark.sh"
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
