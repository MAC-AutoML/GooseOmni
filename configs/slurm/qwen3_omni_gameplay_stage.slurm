#!/bin/bash
#SBATCH --job-name=qwen3-omni-gameplay
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p configs/slurm/logs runs/gooseomni_gameplay_pass1

STAGE="${STAGE:-phase_window}"
DATASET_ROOT="${DATASET_ROOT:-data/gooseomni}"
PHASE_ROOT="${PHASE_ROOT:-runs/gooseomni_gameplay_pass1/phase_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/gooseomni_gameplay_pass1/phase_window_annotations}"
GAME_ID="${GAME_ID:-g001}"
SEGMENT_ID="${SEGMENT_ID:-}"
PHASE_ID="${PHASE_ID:-}"
PLAYER_ID="${PLAYER_ID:-}"
LIMIT="${LIMIT:-}"
SKIP="${SKIP:-0}"
STRIDE="${STRIDE:-1}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((6100 + (${SLURM_JOB_ID:-0} % 300)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-0.5}"
QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"
QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-200704}"

export TOKENIZERS_PARALLELISM=false
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS
export QWEN3_OMNI_VIDEO_FPS
export QWEN3_OMNI_VIDEO_MAX_FRAMES
export QWEN3_OMNI_VIDEO_MAX_PIXELS
export SERVER_TIMEOUT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export HF_MODULES_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_gameplay"
mkdir -p "$HF_MODULES_CACHE"

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
Node: $(hostname)
STAGE: $STAGE
DATASET_ROOT: $DATASET_ROOT
PHASE_ROOT: $PHASE_ROOT
OUTPUT_ROOT: $OUTPUT_ROOT
GAME_ID: $GAME_ID
SEGMENT_ID: ${SEGMENT_ID:-auto}
PHASE_ID: ${PHASE_ID:-auto}
PLAYER_ID: ${PLAYER_ID:-all}
LIMIT: ${LIMIT:-none}
SKIP: $SKIP
STRIDE: $STRIDE
RESUME: $RESUME
OVERWRITE: $OVERWRITE
SERVER_URL: $QWEN3_OMNI_SERVER_URL
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
INFO
nvidia-smi || true

server_log="configs/slurm/logs/qwen3-omni-gameplay-server-${SLURM_JOB_ID:-manual}.log"
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

common=(--backend local --server-url "$QWEN3_OMNI_SERVER_URL" --game-id "$GAME_ID" --skip "$SKIP" --stride "$STRIDE")
if [[ -n "$LIMIT" ]]; then common+=(--limit "$LIMIT"); fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then common+=(--resume); fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then common+=(--overwrite); fi
if [[ -n "$PLAYER_ID" ]]; then common+=(--player-id "$PLAYER_ID"); fi

case "$STAGE" in
  phase_window)
    if [[ -n "$SEGMENT_ID" ]]; then common+=(--segment-id "$SEGMENT_ID"); fi
    .venv/bin/python -m gooseomni internal run tools/annotation/run_gameplay_phase_window_annotation.py --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" "${common[@]}"
    ;;
  phase_video)
    if [[ -n "$PHASE_ID" ]]; then common+=(--phase-id "$PHASE_ID"); fi
    .venv/bin/python -m gooseomni internal run tools/annotation/run_gameplay_phase_video_annotation.py --phase-root "$PHASE_ROOT" --output-root "$OUTPUT_ROOT" "${common[@]}"
    ;;
  *)
    echo "Unsupported STAGE=$STAGE" >&2
    exit 2
    ;;
esac
