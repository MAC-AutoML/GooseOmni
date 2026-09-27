#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="${GOOSEOMNI_REMOTE_ROOT:-/public/home/xty/workdir/omni_goose}"
CONFIG=""
RUN_ROOT=""
NODELIST="${GOOSEOMNI_QWEN_NODELIST:-gpu8}"
MODEL="${GOOSEOMNI_CODEX_MODEL:-gpt-5.6-sol}"
SKIP_QWEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run) RUN_ROOT="$2"; shift 2 ;;
    --nodelist) NODELIST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --skip-qwen) SKIP_QWEN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$RUN_ROOT" ]]; then
  echo "usage: $0 --config CONFIG --run RUN_ROOT [--nodelist gpu8] [--skip-qwen]" >&2
  exit 2
fi
for value in "$CONFIG" "$RUN_ROOT" "$NODELIST"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_./-]+$ ]]; then
    echo "unsafe argument: $value" >&2
    exit 2
  fi
done
if [[ "$CONFIG" == /* || "$RUN_ROOT" == /* || "$CONFIG" == *".."* || "$RUN_ROOT" == *".."* ]]; then
  echo "config and run paths must be repository-relative without '..'" >&2
  exit 2
fi

remote() {
  hpc run "cd \"$REMOTE_ROOT\" && $1"
}

stage_complete() {
  remote "test -f \"$RUN_ROOT/run_manifest.json\" && jq -e '.stages.$1.status == \"complete\"' \"$RUN_ROOT/run_manifest.json\" >/dev/null"
}

if [[ "$SKIP_QWEN" == "0" ]] && ! stage_complete perceive_audio; then
  JOB_ID="$(remote "sbatch --parsable --nodelist=\"$NODELIST\" --gpus-per-node=2 --export=ALL,CONFIG_PATH=\"$CONFIG\",FROM_STAGE=episode configs/slurm/gooseomni_v2_pilot_qwen.slurm" | tail -n 1)"
  echo "submitted Qwen perception job: $JOB_ID"
  while remote "squeue -h -j \"$JOB_ID\" | grep -q ."; do
    remote "squeue -h -j \"$JOB_ID\" -o '%i|%T|%M|%R'"
    sleep 20
  done
  STATE="$(remote "sacct -j \"$JOB_ID\" --format=State -n -X | head -n 1 | xargs")"
  if [[ "$STATE" != "COMPLETED" ]]; then
    echo "Qwen perception job did not complete: $JOB_ID state=$STATE" >&2
    exit 1
  fi
fi

if [[ ! -s "$PROJECT_ROOT/$RUN_ROOT/local_codex/trajectory_decisions.jsonl" ]]; then
  if ! remote ".venv/bin/python -m gooseomni data build --config \"$CONFIG\" --from-stage trajectory_fusion --to-stage trajectory_fusion --resume"; then
    remote "test -s \"$RUN_ROOT/local_codex/trajectory_review_queue.jsonl\""
  fi
  bash "$PROJECT_ROOT/tools/automation/run_local_codex_review.sh" \
    --run "$RUN_ROOT" --kind trajectory --model "$MODEL"
fi

remote ".venv/bin/python -m gooseomni data build --config \"$CONFIG\" --from-stage trajectory_fusion --to-stage tom_trial_build --resume"

if [[ ! -s "$PROJECT_ROOT/$RUN_ROOT/local_codex/decisions.jsonl" ]]; then
  if remote ".venv/bin/python -m gooseomni data build --config \"$CONFIG\" --from-stage codex_review --to-stage codex_review --resume"; then
    echo "Codex review skipped: no candidate probe groups"
  else
    remote "test -s \"$RUN_ROOT/local_codex/review_queue.jsonl\""
    bash "$PROJECT_ROOT/tools/automation/run_local_codex_review.sh" \
      --run "$RUN_ROOT" --kind tom --model "$MODEL"
  fi
fi

remote ".venv/bin/python -m gooseomni data build --config \"$CONFIG\" --from-stage codex_review --to-stage validate --resume"
remote ".venv/bin/python -m gooseomni data status --run \"$RUN_ROOT\""
