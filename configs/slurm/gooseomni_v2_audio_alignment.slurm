#!/bin/bash
#SBATCH --job-name=goose-align
#SBATCH --partition=com300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}"
CONFIG_PATH="${CONFIG_PATH:-configs/datasets/gooseomni_v2_pilot.yaml}"
REFERENCE_ZERO_SEC="${REFERENCE_ZERO_SEC:?REFERENCE_ZERO_SEC must be semantically audited}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/gooseomni_v2_pilot/alignment_candidates/audio_anchors.json}"
OFFSET_HINTS="${OFFSET_HINTS:-}"

cd "$PROJECT_ROOT"
mkdir -p "configs/slurm/logs"
COMMAND=(.venv/bin/python tools/build/extract_audio_alignment_anchors.py \
  --config "$CONFIG_PATH" \
  --reference-zero-sec "$REFERENCE_ZERO_SEC" \
  --output "$OUTPUT_PATH")
if [[ -n "$OFFSET_HINTS" ]]; then
  COMMAND+=(--offset-hints "$OFFSET_HINTS")
fi
"${COMMAND[@]}"
