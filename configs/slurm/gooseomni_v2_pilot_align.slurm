#!/bin/bash
#SBATCH --job-name=goose-pilot-cut
#SBATCH --partition=com300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}"
CONFIG_PATH="${CONFIG_PATH:-configs/datasets/gooseomni_v2_pilot.yaml}"

cd "$PROJECT_ROOT"
mkdir -p "configs/slurm/logs"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m gooseomni data build \
  --config "$CONFIG_PATH" --from-stage align --to-stage align --resume
