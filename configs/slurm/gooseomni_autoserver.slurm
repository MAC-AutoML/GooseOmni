#!/bin/bash
#SBATCH --job-name=gooseomni
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "configs/slurm/logs" "results"

# 可通过 sbatch --export 覆盖：
#   MODEL=qwen3_omni / qwen3_omni_thinking / qwen2_5_omni / miniomni_2 / omnivinci
#   LEVEL=1 或 2
#   MAX_SAMPLES 为空表示跑完整任务；设为 1/10 等用于冒烟测试
LEVEL="${LEVEL:-1}"
MODEL="${MODEL:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-1}"
REUSE_SERVER="${REUSE_SERVER:-0}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
TEST_TIMEOUT="${TEST_TIMEOUT:-86400}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
SERVER_PORT="${SERVER_PORT:-}"

if [[ -z "$SERVER_PORT" ]]; then
  BASE_PORT=5090
  case "$MODEL" in
    qwen3_omni_thinking) BASE_PORT=5091 ;;
    qwen2_5_omni) BASE_PORT=5089 ;;
    miniomni_2) BASE_PORT=5092 ;;
    omnivinci) BASE_PORT=5091 ;;
    vita_1_5) BASE_PORT=5093 ;;
    baichuan_omni_1_5) BASE_PORT=5094 ;;
    ming) BASE_PORT=5095 ;;
  esac
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SERVER_PORT="$((BASE_PORT + (SLURM_JOB_ID % 50)))"
  else
    SERVER_PORT="$BASE_PORT"
  fi
fi

GPU_COUNT_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
GPU_COUNT="$(printf "%s" "$GPU_COUNT_RAW" | tr -cd "0-9")"
if [[ -z "$GPU_COUNT" ]]; then
  GPU_COUNT=1
fi

THREADS_PER_WORKER=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$SLURM_CPUS_PER_TASK" -gt 0 ]]; then
  THREADS_PER_WORKER=$((SLURM_CPUS_PER_TASK / GPU_COUNT))
  if [[ "$THREADS_PER_WORKER" -lt 1 ]]; then
    THREADS_PER_WORKER=1
  fi
fi

export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

HF_MODULES_CACHE_ROOT="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_MODULES_CACHE_ROOT"
export HF_MODULES_CACHE="$HF_MODULES_CACHE_ROOT"

if [[ -z "$MODEL" ]]; then
  MODEL="$(
    .venv/bin/python -u - <<PY
from gooseomni.config.settings import CONFIG
print(CONFIG.benchmark("level${LEVEL}.model", "") or "")
PY
  )"
fi

if [[ -z "$MODEL" ]]; then
  echo "MODEL is empty. Set MODEL via sbatch --export or configs/gooseomni.yaml benchmark.level${LEVEL}.model." >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job Name: ${SLURM_JOB_NAME:-gooseomni}"
echo "Node: $(hostname)"
echo "Submit Dir: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "GPU_COUNT: $GPU_COUNT"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MODEL: $MODEL"
echo "LEVEL: $LEVEL"
echo "MODALITY: ${MODALITY:-${GOOSEOMNI_MODALITY:-config_default}}"
echo "INCLUDE_ASR: ${GOOSEOMNI_LEVEL1_INCLUDE_ASR:-${GOOSEOMNI_INCLUDE_ASR:-config_default}}"
echo "SERVER_PORT: $SERVER_PORT"
echo "HF_MODULES_CACHE: $HF_MODULES_CACHE"
echo "MAX_SAMPLES: ${MAX_SAMPLES:-all}"
echo "REUSE_SERVER: $REUSE_SERVER"
date

nvidia-smi || true

CMD=(
  .venv/bin/python -m gooseomni internal run tools/eval/run_test_with_autoserver.py
  --model "$MODEL"
  --level "$LEVEL"
  --server-timeout "$SERVER_TIMEOUT"
  --test-timeout "$TEST_TIMEOUT"
  --progress-interval "$PROGRESS_INTERVAL"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  CMD+=(--resume)
fi
if [[ "$REUSE_SERVER" == "1" || "$REUSE_SERVER" == "true" ]]; then
  CMD+=(--reuse-server)
fi

echo "Command: ${CMD[*]}"
export GOOSEOMNI_SERVER_PORT="$SERVER_PORT"
export OMNIVINCI_SERVER_PORT="$SERVER_PORT"
export QWEN3_OMNI_SERVER_PORT="$SERVER_PORT"
export QWEN3_OMNI_THINKING_SERVER_PORT="$SERVER_PORT"
export QWEN2_5_OMNI_SERVER_PORT="$SERVER_PORT"
export MINIOMNI2_SERVER_PORT="$SERVER_PORT"
export VITA_1_5_SERVER_PORT="$SERVER_PORT"
export BAICHUAN_OMNI_1_5_SERVER_PORT="$SERVER_PORT"
export MING_SERVER_PORT="$SERVER_PORT"
"${CMD[@]}"
