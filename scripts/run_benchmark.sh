#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi
DATASET_TYPE="${DATASET_TYPE:-sharegpt}"
LIMIT="${LIMIT:-1000}"
MODEL="${MODEL:-}"
DATASET="${DATASET:-}"
OUT_DIR="${OUT_DIR:-results}"
ARRIVAL_RATE="${ARRIVAL_RATE:-2}"
ARRIVAL_SEEDS="${ARRIVAL_SEEDS:-0}"
SCHEDULER_POLICIES="${SCHEDULER_POLICIES:-fcfs plas mlfq_plas}"
MAX_TOKENS="${MAX_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
REPLAY_STEPS="${REPLAY_STEPS:-3}"
SHUFFLE_PROGRAMS="${SHUFFLE_PROGRAMS:-1}"
IGNORE_EOS="${IGNORE_EOS:-1}"
OUTPUT_LENGTH_MODE="${OUTPUT_LENGTH_MODE:-fixed}"
SKIP_OVERLONG_PROGRAMS="${SKIP_OVERLONG_PROGRAMS:-1}"
RESULT_PREFIX="${RESULT_PREFIX:-${DATASET_TYPE}}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  VALIDATE_ONLY=1 DATASET=fixtures/sharegpt_fixture_16.json LIMIT=16 ./scripts/run_benchmark.sh

  DATASET=fixtures/sharegpt_fixture_16.json \
  MODEL=/path/to/Qwen3-0.6B \
  LIMIT=16 \
  ./scripts/run_benchmark.sh

Required for real benchmark:
  DATASET   Local ShareGPT JSON or BFCL JSONL file.
  MODEL     Local model directory, for example /models/Qwen3-0.6B.

Common optional env:
  DATASET_TYPE=sharegpt|bfcl
  LIMIT=1000
  ARRIVAL_RATE=2
  ARRIVAL_SEEDS="0 1 2"
  SCHEDULER_POLICIES="fcfs plas mlfq_plas"
  OUT_DIR=results
  PYTHON_BIN=python3

Outputs:
  results/<dataset>-limit<LIMIT>-rate<RATE>-seed<SEED>-<policy>.jsonl
  results/<dataset>-limit<LIMIT>-rate<RATE>-seed<SEED>-<policy>.jsonl.summary.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  usage
  exit 2
fi

if [[ -z "${DATASET}" ]]; then
  echo "DATASET is required." >&2
  usage
  exit 2
fi

if [[ -z "${MODEL}" && "${VALIDATE_ONLY}" != "1" ]]; then
  echo "MODEL is required unless VALIDATE_ONLY=1." >&2
  usage
  exit 2
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "DATASET does not exist: ${DATASET}" >&2
  exit 2
fi

if [[ "${VALIDATE_ONLY}" != "1" && ! -d "${MODEL}" ]]; then
  echo "MODEL does not exist or is not a directory: ${MODEL}" >&2
  usage
  exit 2
fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

rate_tag="${ARRIVAL_RATE//./p}"

if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  first_seed="${ARRIVAL_SEEDS%% *}"
  args=(
    "${PYTHON_BIN}" -m agentix_app.dataset_runner
    --dataset "${DATASET_TYPE}"
    --input "${DATASET}"
    --limit "${LIMIT}"
    --arrival-rate "${ARRIVAL_RATE}"
    --arrival-seed "${first_seed}"
    --sample-seed "${first_seed}"
    --max-tokens "${MAX_TOKENS}"
    --output-length-mode "${OUTPUT_LENGTH_MODE}"
    --validate-only
  )
  if [[ "${DATASET_TYPE}" == "bfcl" ]]; then
    args+=(--replay-steps "${REPLAY_STEPS}")
  fi
  if [[ "${SHUFFLE_PROGRAMS}" == "1" ]]; then
    args+=(--shuffle-programs)
  else
    args+=(--no-shuffle-programs)
  fi
  "${args[@]}"
  exit 0
fi

for seed in ${ARRIVAL_SEEDS}; do
  for policy in ${SCHEDULER_POLICIES}; do
    out="${OUT_DIR}/${RESULT_PREFIX}-limit${LIMIT}-rate${rate_tag}-seed${seed}-${policy}.jsonl"
    args=(
      "${PYTHON_BIN}" -m agentix_app.dataset_runner
      --dataset "${DATASET_TYPE}"
      --input "${DATASET}"
      --limit "${LIMIT}"
      --model-path "${MODEL}"
      --scheduler-policy "${policy}"
      --arrival-rate "${ARRIVAL_RATE}"
      --arrival-seed "${seed}"
      --sample-seed "${seed}"
      --max-tokens "${MAX_TOKENS}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
      --temperature "${TEMPERATURE}"
      --output-length-mode "${OUTPUT_LENGTH_MODE}"
      --out "${out}"
      --summary-out "${out}.summary.json"
    )

    if [[ "${DATASET_TYPE}" == "bfcl" ]]; then
      args+=(--replay-steps "${REPLAY_STEPS}")
    fi
    if [[ "${SHUFFLE_PROGRAMS}" == "1" ]]; then
      args+=(--shuffle-programs)
    else
      args+=(--no-shuffle-programs)
    fi
    if [[ "${IGNORE_EOS}" == "1" ]]; then
      args+=(--ignore-eos)
    else
      args+=(--no-ignore-eos)
    fi
    if [[ "${SKIP_OVERLONG_PROGRAMS}" == "1" ]]; then
      args+=(--skip-overlong-programs)
    else
      args+=(--no-skip-overlong-programs)
    fi

    echo "Running ${policy}, seed=${seed}, output=${out}"
    "${args[@]}"
  done
done
