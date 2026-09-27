#!/bin/bash
#SBATCH --job-name=goose_decrypto_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/goose_decrypto_diag_%j.out
#SBATCH --error=logs/goose_decrypto_diag_%j.err

set -euo pipefail
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}/src:${SLURM_SUBMIT_DIR:-/public/home/xty/workdir/omni_goose}${PYTHONPATH:+:$PYTHONPATH}"

REPO_ROOT="${REPO_ROOT:-/public/home/xty/workdir/omni_goose}"
cd "${REPO_ROOT}"

mkdir -p logs

# Quality-first defaults for Decrypto-style diagnostic annotation.
# Do not fall back to 4096-token coarse annotation settings for this layer.
export QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
export QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"
export QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"
export QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-96}"
export QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"

RELEASE_ROOT="${RELEASE_ROOT:-runs/gooseomni_gameplay_pass1/release_benchmark_v2}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-annotations}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-benchmark/gooseomni_v1}"
LIMIT="${LIMIT:-240}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m gooseomni internal run tools/build/build_oracle_trajectory_ledger.py \
  --release-root "${RELEASE_ROOT}" \
  --output-root "${ANNOTATION_ROOT}" \
  --game-id g001

"${PYTHON_BIN}" -m gooseomni benchmark build -- \
  --annotation-root "${ANNOTATION_ROOT}" \
  --limit "${LIMIT}"

"${PYTHON_BIN}" -m gooseomni internal run tools/package/export_gooseomni_benchmark.py \
  --annotation-root "${ANNOTATION_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT}"

"${PYTHON_BIN}" -m gooseomni internal run tools/validate/validate_decrypto_style_outputs.py \
  --annotation-root "${ANNOTATION_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --output "${BENCHMARK_ROOT}/reports/validation.json"
