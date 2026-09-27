#!/bin/bash
#SBATCH --job-name=qwen3-gooseomni-eval
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/xty/workdir/omni_goose}"
cd "$REPO_ROOT"
mkdir -p configs/slurm/logs

GOOSEOMNI_BIN="${GOOSEOMNI_BIN:-.venv/bin/gooseomni}"
BENCHMARK="${BENCHMARK:-gooseomni_v2}"
TRACK="${TRACK:-leaderboard_core}"
RUN_ROOT="${RUN_ROOT:-runs/eval/qwen3_omni_${TRACK}}"
LIMIT="${LIMIT:-}"
SKIP="${SKIP:-${SLURM_ARRAY_TASK_ID:-0}}"
STRIDE="${STRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
RESUME="${RESUME:-1}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((6100 + (${SLURM_JOB_ID:-0} % 300)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"

if [[ -z "${MODALITIES:-}" ]]; then
  if [[ "$TRACK" == "raw_video_smoke" ]]; then
    MODALITIES="av v a text"
  else
    MODALITIES="text"
  fi
fi
MODALITIES="${MODALITIES//,/ }"
read -r -a MODALITY_ARGS <<< "$MODALITIES"

if [[ "$STRIDE" != "1" ]]; then
  RUN_ROOT="${RUN_ROOT}/shard_${SKIP}_of_${STRIDE}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-2048}"
export QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-0.5}"
export QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-32}"
export QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-200704}"
export OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-1200}"
export HF_MODULES_CACHE="$REPO_ROOT/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_eval"
mkdir -p "$HF_MODULES_CACHE" "$RUN_ROOT"

server_log="configs/slurm/logs/qwen3-gooseomni-server-${SLURM_JOB_ID:-manual}.log"
"$GOOSEOMNI_BIN" models serve qwen3_omni --host "$SERVER_HOST" --port "$SERVER_PORT" \
  >"$server_log" 2>&1 &
server_pid="$!"
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

.venv/bin/python - <<PY_HEALTH
import requests
import sys
import time

url = "${QWEN3_OMNI_SERVER_URL}/health"
deadline = time.time() + int("${SERVER_TIMEOUT}")
while time.time() < deadline:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and response.json().get("model_loaded"):
            print("server_ready=true")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server_ready=false url={url}", file=sys.stderr)
sys.exit(1)
PY_HEALTH

command=(
  "$GOOSEOMNI_BIN" eval run
  --benchmark "$BENCHMARK"
  --run "$RUN_ROOT"
  --model qwen3_omni
  --track "$TRACK"
  --modalities "${MODALITY_ARGS[@]}"
  --skip "$SKIP"
  --stride "$STRIDE"
  --workers 1
)
if [[ -n "$LIMIT" ]]; then command+=(--limit "$LIMIT"); fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then command+=(--resume); fi
"${command[@]}"
