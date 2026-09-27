#!/bin/bash
#SBATCH --job-name=omni-goose-promote
#SBATCH --partition=com300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

cd "$SLURM_SUBMIT_DIR"

DATASET_ROOT="${DATASET_ROOT:-data/gooseomni}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-runs/gooseomni_oracle_pass1/annotations}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-82}"
UPSTREAM_START_INDEX="${UPSTREAM_START_INDEX:-35}"
UPSTREAM_END_INDEX="${UPSTREAM_END_INDEX:-$END_INDEX}"
MAX_UPSTREAM_JOBS="${MAX_UPSTREAM_JOBS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_DOWNSTREAM_JOBS="${MAX_DOWNSTREAM_JOBS:-20}"
MAX_REPAIR_JOBS="${MAX_REPAIR_JOBS:-8}"
BALANCED_UPSTREAM_JOBS="${BALANCED_UPSTREAM_JOBS:-4}"
BALANCED_DOWNSTREAM_JOBS="${BALANCED_DOWNSTREAM_JOBS:-8}"
BALANCED_REPAIR_JOBS="${BALANCED_REPAIR_JOBS:-4}"
MAX_TOTAL_JOBS="${MAX_TOTAL_JOBS:-60}"
MAX_PENDING_JOBS="${MAX_PENDING_JOBS:-55}"
PREFER_4X2="${PREFER_4X2:-1}"
MULTI_WORKER_STAGE_PLAN="${MULTI_WORKER_STAGE_PLAN:-upstream_chain,downstream_chain,upstream_chain,downstream_chain}"
MULTI_WORKER_LIMIT_PER_WORKER="${MULTI_WORKER_LIMIT_PER_WORKER:-}"
SLEEP_SEC="${SLEEP_SEC:-900}"
ITERATIONS="${ITERATIONS:-48}"

echo "DATASET_ROOT=$DATASET_ROOT"
echo "ANNOTATION_ROOT=$ANNOTATION_ROOT"
echo "START_INDEX=$START_INDEX END_INDEX=$END_INDEX"
echo "UPSTREAM_START_INDEX=$UPSTREAM_START_INDEX UPSTREAM_END_INDEX=$UPSTREAM_END_INDEX MAX_UPSTREAM_JOBS=$MAX_UPSTREAM_JOBS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_DOWNSTREAM_JOBS=$MAX_DOWNSTREAM_JOBS MAX_REPAIR_JOBS=$MAX_REPAIR_JOBS"
echo "BALANCED_UPSTREAM_JOBS=$BALANCED_UPSTREAM_JOBS BALANCED_DOWNSTREAM_JOBS=$BALANCED_DOWNSTREAM_JOBS BALANCED_REPAIR_JOBS=$BALANCED_REPAIR_JOBS"
echo "MAX_TOTAL_JOBS=$MAX_TOTAL_JOBS MAX_PENDING_JOBS=$MAX_PENDING_JOBS"
echo "PREFER_4X2=$PREFER_4X2 MULTI_WORKER_STAGE_PLAN=$MULTI_WORKER_STAGE_PLAN"
echo "SLEEP_SEC=$SLEEP_SEC ITERATIONS=$ITERATIONS"
date

run_step() {
  local name="$1"
  shift
  echo "----- $name -----"
  set +e
  "$@"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "step_failed name=$name status=$status"
  fi
  return 0
}

for iteration in $(seq 1 "$ITERATIONS"); do
  echo "===== iteration=$iteration ====="
  date
  status_log="configs/slurm/logs/omni-goose-promote-status-${SLURM_JOB_ID:-manual}.txt"
  .venv/bin/python -m gooseomni internal run tools/annotation/submit_gooseomni_oracle_jobs.py     --dataset-root "$DATASET_ROOT"     --annotation-root "$ANNOTATION_ROOT"     --mode status     --start-index "$START_INDEX"     --end-index "$END_INDEX" > "$status_log"
  sed -n 1,12p "$status_log"

  if [[ "$PREFER_4X2" == "1" || "$PREFER_4X2" == "true" ]]; then
    run_step submit-4x2-balanced .venv/bin/python -m gooseomni internal run tools/annotation/submit_gooseomni_oracle_jobs.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"       --mode submit-4x2-balanced       --max-total-jobs "$MAX_TOTAL_JOBS"       --max-pending-jobs "$MAX_PENDING_JOBS"       --multi-worker-stage-plan "$MULTI_WORKER_STAGE_PLAN"       --multi-worker-limit-per-worker "$MULTI_WORKER_LIMIT_PER_WORKER"
  fi

  run_step submit-balanced     .venv/bin/python -m gooseomni internal run tools/annotation/submit_gooseomni_oracle_jobs.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"       --mode submit-balanced       --start-index "$START_INDEX"       --end-index "$END_INDEX"       --upstream-start-index "$UPSTREAM_START_INDEX"       --upstream-end-index "$UPSTREAM_END_INDEX"       --max-jobs "$MAX_UPSTREAM_JOBS"       --batch-size "$BATCH_SIZE"       --balanced-upstream-jobs "$BALANCED_UPSTREAM_JOBS"       --balanced-downstream-jobs "$BALANCED_DOWNSTREAM_JOBS"       --balanced-repair-jobs "$BALANCED_REPAIR_JOBS"       --max-total-jobs "$MAX_TOTAL_JOBS"       --max-pending-jobs "$MAX_PENDING_JOBS"

  run_step recover-errors .venv/bin/python -m gooseomni internal run tools/annotation/recover_gooseomni_error_outputs.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"

  run_step export     .venv/bin/python -m gooseomni internal run tools/annotation/submit_gooseomni_oracle_jobs.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"       --mode export

  run_step report     .venv/bin/python -m gooseomni internal run tools/annotation/report_gooseomni_progress.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"       --benchmark-root "${ANNOTATION_ROOT%/annotations}/benchmark"       --output "${ANNOTATION_ROOT%/annotations}/progress_report.json"

  run_step validate-completion .venv/bin/python -m gooseomni internal run tools/validate/validate_gooseomni_completion.py       --dataset-root "$DATASET_ROOT"       --annotation-root "$ANNOTATION_ROOT"       --benchmark-root "${ANNOTATION_ROOT%/annotations}/benchmark"       --output "${ANNOTATION_ROOT%/annotations}/completion_validation.json"

  if [[ "$iteration" -lt "$ITERATIONS" ]]; then
    sleep "$SLEEP_SEC"
  fi
done
