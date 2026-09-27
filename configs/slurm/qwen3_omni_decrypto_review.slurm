#!/bin/bash
#SBATCH --job-name=goose_decrypto_review
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p configs/slurm/logs

PASS_ROOT="${PASS_ROOT:-runs/gooseomni_decrypto_diagnostic_pass2_reviewed}"
PASS1_ROOT="${PASS1_ROOT:-runs/gooseomni_decrypto_diagnostic_pass1}"
QUEUE="${QUEUE:-$PASS1_ROOT/review/qwen3_omni_high_quality_review_queue.jsonl}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$PASS1_ROOT/review/qwen3_omni_high_quality_review_prompt_template.md}"
RELEASE_ROOT="${RELEASE_ROOT:-runs/gooseomni_gameplay_pass1/release_benchmark_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PASS_ROOT/review/qwen3_omni_high_quality_results}"
LIMIT="${LIMIT:-4}"
SKIP="${SKIP:-0}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((6200 + (${SLURM_JOB_ID:-0} % 300)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"

export TOKENIZERS_PARALLELISM=false
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
export QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"
export QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"
export QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-128}"
export QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"
export OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-900}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export HF_MODULES_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_decrypto_review"
mkdir -p "$HF_MODULES_CACHE" "$OUTPUT_ROOT"

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
Node: $(hostname)
PASS_ROOT: $PASS_ROOT
QUEUE: $QUEUE
LIMIT: $LIMIT
SKIP: $SKIP
OUTPUT_ROOT: $OUTPUT_ROOT
SERVER_URL: $QWEN3_OMNI_SERVER_URL
MAX_TOKENS: $QWEN3_OMNI_MAX_TOKENS
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
HTTP_TIMEOUT_SEC: $OMNI_HTTP_TIMEOUT_SEC
INFO
nvidia-smi || true

server_log="configs/slurm/logs/qwen3-omni-decrypto-review-server-${SLURM_JOB_ID:-manual}.log"
.venv/bin/python -m gooseomni models serve qwen3_omni --host "$SERVER_HOST" --port "$SERVER_PORT" > "$server_log" 2>&1 &
server_pid="$!"
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

.venv/bin/python - <<'HEALTHPY'
import os, sys, time, requests
url = os.environ["QWEN3_OMNI_SERVER_URL"] + "/health"
deadline = time.time() + int(os.environ.get("SERVER_TIMEOUT", "2400"))
while time.time() < deadline:
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200 and r.json().get("model_loaded"):
            print("server_ready=true")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server_ready=false url={url}", file=sys.stderr)
sys.exit(1)
HEALTHPY

common=(--queue "$QUEUE" --prompt-template "$PROMPT_TEMPLATE" --release-root "$RELEASE_ROOT" --output-root "$OUTPUT_ROOT" --backend local --server-url "$QWEN3_OMNI_SERVER_URL" --limit "$LIMIT" --skip "$SKIP")
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then common+=(--resume); fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then common+=(--overwrite); fi
.venv/bin/python -m gooseomni internal run tools/annotation/run_decrypto_high_quality_review.py "${common[@]}"
