#!/bin/bash
#SBATCH --job-name=qwen3-omni-tom-smoke
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
mkdir -p "configs/slurm/logs" "runs/gooseomni_benchmark_smoke" "annotations" "benchmark"

DATASET_ROOT="${DATASET_ROOT:-data/gooseomni}"
SEGMENTS_JSONL="${SEGMENTS_JSONL:-$DATASET_ROOT/segments.jsonl}"
GAME_ID="${GAME_ID:-g001}"
LIMIT="${LIMIT:-1}"
TARGET_PLAYER="${TARGET_PLAYER:-}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-runs/gooseomni_benchmark_smoke/annotations}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-runs/gooseomni_benchmark_smoke/benchmark}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((5200 + (${SLURM_JOB_ID:-0} % 200)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-0.5}"
QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"
QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-200704}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS
export QWEN3_OMNI_VIDEO_FPS
export QWEN3_OMNI_VIDEO_MAX_FRAMES
export QWEN3_OMNI_VIDEO_MAX_PIXELS
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export GOOSEOMNI_ANNOTATION_ROOT="$ANNOTATION_ROOT"

HF_MODULES_CACHE_ROOT="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_MODULES_CACHE_ROOT"
export HF_MODULES_CACHE="$HF_MODULES_CACHE_ROOT"

COMMON_ARGS=(--dataset-root "$DATASET_ROOT" --segments-jsonl "$SEGMENTS_JSONL" --output-root "$ANNOTATION_ROOT" --backend local --server-url "$QWEN3_OMNI_SERVER_URL" --game-id "$GAME_ID" --limit "$LIMIT" --overwrite)
if [[ -n "$TARGET_PLAYER" ]]; then
  TARGET_ARGS=(--target-player "$TARGET_PLAYER")
else
  TARGET_ARGS=()
fi

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
Node: $(hostname)
DATASET_ROOT: $DATASET_ROOT
SEGMENTS_JSONL: $SEGMENTS_JSONL
GAME_ID: $GAME_ID
LIMIT: $LIMIT
ANNOTATION_ROOT: $ANNOTATION_ROOT
BENCHMARK_ROOT: $BENCHMARK_ROOT
MODEL_PATH: $MODEL_PATH
SERVER_URL: $QWEN3_OMNI_SERVER_URL
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
VIDEO_MAX_PIXELS: $QWEN3_OMNI_VIDEO_MAX_PIXELS
INFO
date
nvidia-smi || true

SERVER_LOG="configs/slurm/logs/qwen3-omni-tom-smoke-server-${SLURM_JOB_ID:-manual}.log"
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
import sys, time, requests
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

run_step() {
  echo "\n===== $* ====="
  "$@"
}

run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_pov_event_annotation.py "${COMMON_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_utterance_annotation.py "${COMMON_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_phase_event_annotation.py "${COMMON_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_global_merge.py "${COMMON_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_information_state.py "${COMMON_ARGS[@]}" "${TARGET_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_memory_state_update.py "${COMMON_ARGS[@]}" "${TARGET_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_belief_state_annotation.py "${COMMON_ARGS[@]}" "${TARGET_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_candidate_trial_generation.py "${COMMON_ARGS[@]}"
run_step .venv/bin/python -m gooseomni internal run tools/annotation/run_annotation_checkers.py --dataset-root "$DATASET_ROOT" --segments-jsonl "$SEGMENTS_JSONL" --output-root "$ANNOTATION_ROOT" --game-id "$GAME_ID" --limit "$LIMIT"
run_step .venv/bin/python -m gooseomni internal run tools/package/export_tom_benchmark.py --dataset-root "$DATASET_ROOT" --annotation-root "$ANNOTATION_ROOT" --benchmark-root "$BENCHMARK_ROOT"
run_step .venv/bin/python -m gooseomni internal run tools/report/report_benchmark_quality.py --dataset-root "$DATASET_ROOT" --benchmark-root "$BENCHMARK_ROOT"

run_step .venv/bin/python - <<PY
from pathlib import Path
import json
root=Path("$ANNOTATION_ROOT")
bench=Path("$BENCHMARK_ROOT")
for sub in ["pov_events", "utterances", "phase_events", "global_events", "information_states", "memory_states", "belief_states", "candidate_trials", "checker_reports", "errors"]:
    p=root/sub
    n=sum(1 for _ in p.rglob("*.json")) + sum(1 for _ in p.rglob("*.jsonl")) if p.exists() else 0
    print(sub, n)
for path in [bench/"weak"/"trials.jsonl", bench/"weak"/"gold_weak.jsonl", bench/"weak"/"metadata.json"]:
    print(path, path.exists(), path.stat().st_size if path.exists() else 0)
PY
