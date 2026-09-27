#!/bin/bash
#SBATCH --job-name=gooseomni_q3_q2judge
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "configs/slurm/logs" "results/tmp"

SERVER_PORT="${SERVER_PORT:-$((5090 + (${SLURM_JOB_ID:-0} % 50)))}"
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
SERVER_LOG="results/tmp/qwen3_level2_q2_judge_server_${SLURM_JOB_ID:-manual}.log"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export QWEN3_OMNI_SERVER_URL="$SERVER_URL"
export GOOSEOMNI_SERVER_PORT="$SERVER_PORT"
export HF_MODULES_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_MODULES_CACHE"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "SERVER_URL: $SERVER_URL"
echo "HF_MODULES_CACHE: $HF_MODULES_CACHE"
date
nvidia-smi || true

.venv/bin/python -m gooseomni models serve qwen3_omni \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

deadline=$((SECONDS + ${SERVER_TIMEOUT:-1800}))
until .venv/bin/python - <<PY
import requests
try:
    r = requests.get("$SERVER_URL/health", timeout=2)
    data = r.json()
    raise SystemExit(0 if r.status_code == 200 and data.get("model_loaded") else 1)
except Exception:
    raise SystemExit(1)
PY
do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[FAIL] server exited before ready. See $SERVER_LOG"
    tail -200 "$SERVER_LOG" || true
    exit 1
  fi
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "[FAIL] server startup timeout. See $SERVER_LOG"
    tail -200 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
done

echo "[INFO] server ready, scoring q2"
.venv/bin/python -m gooseomni internal run tools/eval/score_level2_q2_with_omni.py \
  --server-url "$SERVER_URL"

.venv/bin/python -m gooseomni internal run tools/report/report_level2_extended_by_language.py

echo "[DONE] q2 judge and language report generated"
