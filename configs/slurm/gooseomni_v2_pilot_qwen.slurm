#!/bin/bash
#SBATCH --job-name=goose-v2-pilot
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}"
CONFIG_PATH="${CONFIG_PATH:-configs/datasets/gooseomni_v2_pilot.yaml}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import socket

with socket.socket() as handle:
    handle.bind(("127.0.0.1", 0))
    print(handle.getsockname()[1])
PY
)"
FROM_STAGE="${FROM_STAGE:-episode}"
export GOOSEOMNI_PILOT_LIMIT_CLIPS="${GOOSEOMNI_PILOT_LIMIT_CLIPS:-}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export GOOSEOMNI_SERVER_INSTANCE="${SLURM_JOB_ID:-manual}"
export QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-2048}"
export QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-0.5}"
export QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-32}"
export QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-200704}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_MODULES_CACHE="$PROJECT_ROOT/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}"

cd "$PROJECT_ROOT"
mkdir -p "configs/slurm/logs" "$HF_MODULES_CACHE"
nvidia-smi

# Materialize/replay audited alignment before paying the model-loading cost.
.venv/bin/python -m gooseomni data build \
  --config "$CONFIG_PATH" --from-stage align --to-stage align --resume

SERVER_PID=""

wait_for_server() {
  .venv/bin/python - <<'PY'
import os
import time

import requests

url = os.environ["QWEN3_OMNI_SERVER_URL"] + "/health"
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        response = requests.get(url, timeout=5)
        payload = response.json()
        if (
            response.ok
            and payload.get("model_loaded")
            and payload.get("instance_id") == os.environ["GOOSEOMNI_SERVER_INSTANCE"]
        ):
            raise SystemExit(0)
    except requests.RequestException:
        pass
    time.sleep(5)
raise SystemExit(f"Qwen3-Omni service did not become healthy: {url}")
PY
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

start_server() {
  .venv/bin/python -m gooseomni models serve qwen3_omni \
    --host "$SERVER_HOST" --port "$SERVER_PORT" \
    > "configs/slurm/logs/goose-v2-pilot-server-${SLURM_JOB_ID:-manual}.log" 2>&1 &
  SERVER_PID="$!"
  echo "Qwen server instance=$GOOSEOMNI_SERVER_INSTANCE pid=$SERVER_PID url=$QWEN3_OMNI_SERVER_URL"
  wait_for_server
}

trap stop_server EXIT
start_server

ATTEMPT=1
MAX_ATTEMPTS="${GOOSEOMNI_QWEN_RESTART_ATTEMPTS:-3}"
while ! .venv/bin/python -m gooseomni data build \
  --config "$CONFIG_PATH" --from-stage "$FROM_STAGE" \
  --to-stage perceive_audio --resume; do
  if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "Qwen perception failed after $ATTEMPT service attempts" >&2
    exit 1
  fi
  ATTEMPT=$((ATTEMPT + 1))
  echo "Restarting Qwen service after failed perception attempt: $ATTEMPT/$MAX_ATTEMPTS"
  stop_server
  start_server
done
