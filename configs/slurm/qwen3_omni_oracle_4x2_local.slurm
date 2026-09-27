#!/bin/bash
#SBATCH --job-name=qwen3-omni-oracle-4x2
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "configs/slurm/logs" "runs/gooseomni_oracle_pass1"

DATASET_ROOT="${DATASET_ROOT:-data/gooseomni}"
SEGMENTS_JSONL="${SEGMENTS_JSONL:-$DATASET_ROOT/segments.jsonl}"
GAME_ID="${GAME_ID:-g001}"
LIMIT_PER_WORKER="${LIMIT_PER_WORKER:-}"
WORKERS="${WORKERS:-4}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-runs/gooseomni_oracle_pass1/annotations}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-$((5600 + (${SLURM_JOB_ID:-0} % 200)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"
QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"
QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"
QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"
OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-900}"
STAGE_SET="${STAGE_SET:-local}"
STAGE_PLAN="${STAGE_PLAN:-}"
PLAYER_ID="${PLAYER_ID:-}"
TARGET_PLAYER="${TARGET_PLAYER:-}"
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

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
Node: $(hostname)
WORKERS: $WORKERS
STAGE_SET: $STAGE_SET
STAGE_PLAN: ${STAGE_PLAN:-auto}
DATASET_ROOT: $DATASET_ROOT
GAME_ID: $GAME_ID
LIMIT_PER_WORKER: $LIMIT_PER_WORKER
ANNOTATION_ROOT: $ANNOTATION_ROOT
RESUME: $RESUME
OVERWRITE: $OVERWRITE
MODEL_PATH: $MODEL_PATH
BASE_PORT: $BASE_PORT
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
VIDEO_MAX_PIXELS: $QWEN3_OMNI_VIDEO_MAX_PIXELS
INFO
date
nvidia-smi || true

PIDS=()
SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

launch_worker() {
  local worker_id="$1"
  local gpu_a=$((worker_id * 2))
  local gpu_b=$((worker_id * 2 + 1))
  local port=$((BASE_PORT + worker_id))
  local server_url="http://${SERVER_HOST}:${port}"
  local worker_log="configs/slurm/logs/qwen3-omni-oracle-4x2-worker-${SLURM_JOB_ID:-manual}_${worker_id}.log"
  local server_log="configs/slurm/logs/qwen3-omni-oracle-4x2-server-${SLURM_JOB_ID:-manual}_${worker_id}.log"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}"
    export QWEN3_OMNI_SERVER_URL="$server_url"
    export HF_MODULES_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_${worker_id}"
    mkdir -p "$HF_MODULES_CACHE"

    echo "worker=${worker_id} cuda=${CUDA_VISIBLE_DEVICES} server=${server_url} log=${worker_log}"
    .venv/bin/python -m gooseomni models serve qwen3_omni --host "$SERVER_HOST" --port "$port" > "$server_log" 2>&1 &
    local server_pid="$!"
    echo "$server_pid" > "configs/slurm/logs/qwen3-omni-oracle-4x2-server-${SLURM_JOB_ID:-manual}_${worker_id}.pid"

    .venv/bin/python - <<PY2
import sys, time, requests
url = "${server_url}/health"
deadline = time.time() + int("${SERVER_TIMEOUT}")
while time.time() < deadline:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            print("server_ready=true worker=${worker_id}")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server_ready=false worker=${worker_id} url={url}", file=sys.stderr)
sys.exit(1)
PY2

    local plan_text="$STAGE_PLAN"
    if [[ -z "$plan_text" ]]; then
      case "$STAGE_SET" in
        local|upstream)
          plan_text="upstream_chain"
          ;;
        downstream)
          plan_text="downstream_chain"
          ;;
        all|balanced)
          plan_text="upstream_chain,downstream_chain"
          ;;
        *)
          plan_text="$STAGE_SET"
          ;;
      esac
    fi
    IFS=',' read -r -a stage_plan <<< "$plan_text"
    local stage_count="${#stage_plan[@]}"
    local stage_index=$((worker_id % stage_count))
    local stage="${stage_plan[$stage_index]}"
    local stage_worker_index=0
    local stage_worker_count=0
    local idx=0
    for idx in $(seq 0 $((WORKERS - 1))); do
      local idx_stage="${stage_plan[$((idx % stage_count))]}"
      if [[ "$idx_stage" == "$stage" ]]; then
        if [[ "$idx" -lt "$worker_id" ]]; then
          stage_worker_index=$((stage_worker_index + 1))
        fi
        stage_worker_count=$((stage_worker_count + 1))
      fi
    done

    echo "worker_stage=${worker_id} stage=${stage} stage_worker=${stage_worker_index}/${stage_worker_count}"
    common=(--dataset-root "$DATASET_ROOT" --segments-jsonl "$SEGMENTS_JSONL" --output-root "$ANNOTATION_ROOT" --backend local --server-url "$server_url" --game-id "$GAME_ID" --skip "$stage_worker_index" --stride "$stage_worker_count")
    if [[ -n "$LIMIT_PER_WORKER" ]]; then
      common+=(--limit "$LIMIT_PER_WORKER")
    fi
    if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
      common+=(--resume)
    fi
    if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
      common+=(--overwrite)
    fi

    case "$stage" in
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
      upstream|upstream_chain|oracle_upstream_chain)
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
      downstream|downstream_chain|oracle_downstream_chain)
        .venv/bin/python -m gooseomni internal run tools/annotation/run_global_merge.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_information_state.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_memory_state_update.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_belief_state_annotation.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_candidate_trial_generation.py "${common[@]}"
        ;;
      local|all)
        .venv/bin/python -m gooseomni internal run tools/annotation/run_pov_event_annotation.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_utterance_annotation.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_phase_event_annotation.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_global_merge.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_information_state.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_memory_state_update.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_belief_state_annotation.py "${common[@]}"
        .venv/bin/python -m gooseomni internal run tools/annotation/run_candidate_trial_generation.py "${common[@]}"
        ;;
      *)
        echo "Unsupported stage: $stage" >&2
        exit 2
        ;;
    esac

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "worker_done=${worker_id}"
  ) > "$worker_log" 2>&1 &
  PIDS+=("$!")
}

for worker_id in $(seq 0 $((WORKERS - 1))); do
  launch_worker "$worker_id"
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

.venv/bin/python - <<PY2
from pathlib import Path
root=Path("$ANNOTATION_ROOT")
for sub in ["pov_events", "utterances", "phase_events", "global_events", "information_states", "memory_states", "belief_states", "candidate_trials", "errors", "review_queue"]:
    p=root/sub
    n=sum(1 for _ in p.rglob("*.json")) + sum(1 for _ in p.rglob("*.jsonl")) if p.exists() else 0
    print(sub, n)
PY2

exit "$status"
