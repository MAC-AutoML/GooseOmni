#!/bin/bash
#SBATCH --job-name=qwen3-round
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
mkdir -p "configs/slurm/logs" "annotations"

MANIFEST_PATH="${MANIFEST_PATH:-data/processed/clip_manifest_manual_sync_v1.jsonl}"
ANNOTATION_DIR="${ANNOTATION_DIR:-annotations/round_boundaries_manual_sync_v1}"
ERROR_DIR="${ERROR_DIR:-annotations/round_boundary_errors_manual_sync_v1}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((5140 + (${SLURM_JOB_ID:-0} % 50)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-0.5}"
QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-32}"
QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-200704}"
LIMIT="${LIMIT:-}"
GAME_ID="${GAME_ID:-}"
PLAYER_ID="${PLAYER_ID:-}"
RESUME="${RESUME:-1}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS
export QWEN3_OMNI_VIDEO_FPS
export QWEN3_OMNI_VIDEO_MAX_FRAMES
export QWEN3_OMNI_VIDEO_MAX_PIXELS
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

HF_MODULES_CACHE_ROOT="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_MODULES_CACHE_ROOT"
export HF_MODULES_CACHE="$HF_MODULES_CACHE_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "MANIFEST_PATH: $MANIFEST_PATH"
echo "ANNOTATION_DIR: $ANNOTATION_DIR"
echo "ERROR_DIR: $ERROR_DIR"
echo "MODEL_PATH: $MODEL_PATH"
echo "SERVER_URL: $QWEN3_OMNI_SERVER_URL"
echo "VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS"
echo "VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES"
echo "VIDEO_MAX_PIXELS: $QWEN3_OMNI_VIDEO_MAX_PIXELS"
date
nvidia-smi || true

SERVER_LOG="configs/slurm/logs/qwen3-round-server-${SLURM_JOB_ID:-manual}.log"
.venv/bin/python -m gooseomni models serve qwen3_omni \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Waiting for Qwen3-Omni server, log=$SERVER_LOG"
.venv/bin/python - <<PY
import sys
import time
import requests

url = "${QWEN3_OMNI_SERVER_URL}/health"
deadline = time.time() + int("${SERVER_TIMEOUT}")
while time.time() < deadline:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            print("server_ready=true")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server_ready=false url={url}", file=sys.stderr)
sys.exit(1)
PY

ANNOTATE_CMD=(
  .venv/bin/python -m gooseomni internal run tools/annotation/run_qwen_round_boundary_annotation.py
  --backend local
  --server-url "$QWEN3_OMNI_SERVER_URL"
  --manifest-path "$MANIFEST_PATH"
  --output-dir "$ANNOTATION_DIR"
  --error-dir "$ERROR_DIR"
)
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  ANNOTATE_CMD+=(--resume)
fi
if [[ -n "$LIMIT" ]]; then
  ANNOTATE_CMD+=(--limit "$LIMIT")
fi
if [[ -n "$GAME_ID" ]]; then
  ANNOTATE_CMD+=(--game-id "$GAME_ID")
fi
if [[ -n "$PLAYER_ID" ]]; then
  ANNOTATE_CMD+=(--player-id "$PLAYER_ID")
fi

echo "Round boundary command: ${ANNOTATE_CMD[*]}"
"${ANNOTATE_CMD[@]}"
