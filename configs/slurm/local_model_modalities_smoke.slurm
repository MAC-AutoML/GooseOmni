#!/bin/bash
#SBATCH --job-name=gooseomni-modalities
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:?Set MODEL to one registered local model}"
SAMPLE="${SAMPLE:-data/processed/clips/g001/beigang/g001_beigang_7200_7225.mp4}"
SERVER_HOST="127.0.0.1"
SERVER_PORT="${SERVER_PORT:-$((5400 + (${SLURM_JOB_ID:-0} % 200)))}"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
OUTPUT="${OUTPUT:-runs/local_model_modalities/${MODEL}/${SLURM_JOB_ID:-manual}.json}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p configs/slurm/logs "$(dirname "$OUTPUT")"
SERVER_LOG="configs/slurm/logs/${MODEL}-server-${SLURM_JOB_ID:-manual}.log"

"$PYTHON_BIN" -m gooseomni models serve "$MODEL" \
  --host "$SERVER_HOST" --port "$SERVER_PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" - <<PY
import os
import sys
import time

import requests

url = "$SERVER_URL/health"
deadline = time.time() + int("$SERVER_TIMEOUT")
while time.time() < deadline:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and response.json().get("model_loaded"):
            print("server_ready=true")
            sys.exit(0)
    except Exception:
        pass
    if not os.path.exists("/proc/$SERVER_PID"):
        print("server process exited; inspect $SERVER_LOG", file=sys.stderr)
        sys.exit(1)
    time.sleep(5)
print("server_ready=false", file=sys.stderr)
sys.exit(1)
PY

"$PYTHON_BIN" -m gooseomni internal run \
  tools/eval/smoke_local_model_modalities.py \
  --model "$MODEL" \
  --server-url "$SERVER_URL" \
  --sample "$SAMPLE" \
  --output "$OUTPUT" \
  --timeout-sec "$REQUEST_TIMEOUT"
