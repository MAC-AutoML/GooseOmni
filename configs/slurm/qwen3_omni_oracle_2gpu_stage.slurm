#!/bin/bash
#SBATCH --job-name=qwen3-omni-2gpu-stage
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
mkdir -p "configs/slurm/logs" "runs/gooseomni_oracle"

DATASET_ROOT="${DATASET_ROOT:-data/gooseomni}"
SEGMENTS_JSONL="${SEGMENTS_JSONL:-$DATASET_ROOT/segments.jsonl}"
GAME_ID="${GAME_ID:-g001}"
STAGE="${STAGE:-pov_events}"
SEGMENT_ID="${SEGMENT_ID:-}"
SEGMENT_IDS="${SEGMENT_IDS:-}"
LIMIT="${LIMIT:-1}"
SKIP="${SKIP:-0}"
STRIDE="${STRIDE:-1}"
TARGET_PLAYER="${TARGET_PLAYER:-}"
PLAYER_ID="${PLAYER_ID:-}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-runs/gooseomni_oracle/annotations}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((5900 + (${SLURM_JOB_ID:-0} % 400)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"
QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"
QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"
QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"
OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-900}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS
export QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS
export QWEN3_OMNI_VIDEO_FPS
export QWEN3_OMNI_VIDEO_MAX_FRAMES
export QWEN3_OMNI_VIDEO_MAX_PIXELS
export OMNI_HTTP_TIMEOUT_SEC
export GOOSEOMNI_ANNOTATION_ROOT="$ANNOTATION_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export HF_MODULES_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_2gpu"

mkdir -p "$HF_MODULES_CACHE"

manifest_root="${ANNOTATION_ROOT%/annotations}/job_manifests"
mkdir -p "$manifest_root"
.venv/bin/python - <<PY
import json, os, time
from pathlib import Path

payload = {
    "job_id": os.environ.get("SLURM_JOB_ID", "manual"),
    "job_name": os.environ.get("SLURM_JOB_NAME", ""),
    "node": os.uname().nodename,
    "stage": "$STAGE",
    "dataset_root": "$DATASET_ROOT",
    "segments_jsonl": "$SEGMENTS_JSONL",
    "game_id": "$GAME_ID",
    "segment_id": "$SEGMENT_ID",
    "segment_ids": "$SEGMENT_IDS",
    "limit": "$LIMIT",
    "skip": "$SKIP",
    "stride": "$STRIDE",
    "player_id": "$PLAYER_ID",
    "target_player": "$TARGET_PLAYER",
    "annotation_root": "$ANNOTATION_ROOT",
    "model_path": "$MODEL_PATH",
    "server_url": "$QWEN3_OMNI_SERVER_URL",
    "video_fps": "$QWEN3_OMNI_VIDEO_FPS",
    "video_max_frames": "$QWEN3_OMNI_VIDEO_MAX_FRAMES",
    "video_max_pixels": "$QWEN3_OMNI_VIDEO_MAX_PIXELS",
    "resume": "$RESUME",
    "overwrite": "$OVERWRITE",
    "created_at_unix": time.time(),
}
path = Path("$manifest_root") / f"{payload['job_id']}.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"job_manifest={path.as_posix()}")
PY

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
Node: $(hostname)
STAGE: $STAGE
DATASET_ROOT: $DATASET_ROOT
GAME_ID: $GAME_ID
SEGMENT_ID: ${SEGMENT_ID:-auto}
SEGMENT_IDS: ${SEGMENT_IDS:-none}
LIMIT: $LIMIT
SKIP: $SKIP
STRIDE: $STRIDE
PLAYER_ID: ${PLAYER_ID:-all}
TARGET_PLAYER: ${TARGET_PLAYER:-all}
ANNOTATION_ROOT: $ANNOTATION_ROOT
RESUME: $RESUME
OVERWRITE: $OVERWRITE
MODEL_PATH: $MODEL_PATH
SERVER_URL: $QWEN3_OMNI_SERVER_URL
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
VIDEO_MAX_PIXELS: $QWEN3_OMNI_VIDEO_MAX_PIXELS
INFO
date
nvidia-smi || true

server_log="configs/slurm/logs/qwen3-omni-2gpu-stage-server-${SLURM_JOB_ID:-manual}.log"
.venv/bin/python -m gooseomni models serve qwen3_omni --host "$SERVER_HOST" --port "$SERVER_PORT" > "$server_log" 2>&1 &
server_pid="$!"

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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

run_stage() {
  local segment_id="$1"
  local common=(--dataset-root "$DATASET_ROOT" --segments-jsonl "$SEGMENTS_JSONL" --output-root "$ANNOTATION_ROOT" --backend local --server-url "$QWEN3_OMNI_SERVER_URL" --game-id "$GAME_ID" --skip "$SKIP" --stride "$STRIDE")
  if [[ -n "$segment_id" ]]; then
    common+=(--segment-id "$segment_id")
  fi
  if [[ -n "$LIMIT" ]]; then
    common+=(--limit "$LIMIT")
  fi
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    common+=(--resume)
  fi
  if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
    common+=(--overwrite)
  fi

  echo "run_stage stage=$STAGE segment_id=${segment_id:-auto}"
  case "$STAGE" in
    pov|pov_events)
      if [[ -n "$PLAYER_ID" ]]; then common+=(--player-id "$PLAYER_ID"); fi
      .venv/bin/python -m gooseomni internal run tools/annotation/run_pov_event_annotation.py "${common[@]}"
      ;;
    utterance|utterances)
      if [[ -n "$PLAYER_ID" ]]; then common+=(--player-id "$PLAYER_ID"); fi
      .venv/bin/python -m gooseomni internal run tools/annotation/run_utterance_annotation.py "${common[@]}"
      ;;
    phase|phase_events)
      .venv/bin/python -m gooseomni internal run tools/annotation/run_phase_event_annotation.py "${common[@]}"
      ;;
    upstream_chain|oracle_upstream_chain)
      .venv/bin/python -m gooseomni internal run tools/annotation/run_pov_event_annotation.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_utterance_annotation.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_phase_event_annotation.py "${common[@]}"
      ;;
    global|global_merge|global_events)
      .venv/bin/python -m gooseomni internal run tools/annotation/run_global_merge.py "${common[@]}"
      ;;
    information|information_state|information_states)
      if [[ -n "$TARGET_PLAYER" ]]; then common+=(--target-player "$TARGET_PLAYER"); fi
      .venv/bin/python -m gooseomni internal run tools/annotation/run_information_state.py "${common[@]}"
      ;;
    memory|memory_state|memory_states)
      if [[ -n "$TARGET_PLAYER" ]]; then common+=(--target-player "$TARGET_PLAYER"); fi
      .venv/bin/python -m gooseomni internal run tools/annotation/run_memory_state_update.py "${common[@]}"
      ;;
    belief|belief_state|belief_states)
      if [[ -n "$TARGET_PLAYER" ]]; then common+=(--target-player "$TARGET_PLAYER"); fi
      .venv/bin/python -m gooseomni internal run tools/annotation/run_belief_state_annotation.py "${common[@]}"
      ;;
    trial|candidate_trial|candidate_trials)
      .venv/bin/python -m gooseomni internal run tools/annotation/run_candidate_trial_generation.py "${common[@]}"
      ;;
    downstream_chain|oracle_downstream_chain)
      .venv/bin/python -m gooseomni internal run tools/annotation/run_global_merge.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_information_state.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_memory_state_update.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_belief_state_annotation.py "${common[@]}"
      .venv/bin/python -m gooseomni internal run tools/annotation/run_candidate_trial_generation.py "${common[@]}"
      ;;
    *)
      echo "Unsupported STAGE: $STAGE" >&2
      exit 2
      ;;
  esac
}

if [[ -n "$SEGMENT_IDS" ]]; then
  IFS=', ' read -r -a segment_list <<< "$SEGMENT_IDS"
  for batch_segment_id in "${segment_list[@]}"; do
    if [[ -n "$batch_segment_id" ]]; then
      run_stage "$batch_segment_id"
    fi
  done
else
  run_stage "$SEGMENT_ID"
fi

.venv/bin/python - <<PY
from pathlib import Path
root=Path("$ANNOTATION_ROOT")
for sub in ["pov_events", "utterances", "phase_events", "global_events", "information_states", "memory_states", "belief_states", "candidate_trials", "errors", "review_queue"]:
    p=root/sub
    n=sum(1 for _ in p.rglob("*.json")) + sum(1 for _ in p.rglob("*.jsonl")) if p.exists() else 0
    print(sub, n)
PY
