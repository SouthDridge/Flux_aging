#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the source image directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for filtered images}"

args=(
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --occlusion_threshold "${OCCLUSION_THRESHOLD:-0.5}"
  --transfer_mode "${TRANSFER_MODE:-symlink}"
)
if [[ "${REJECT_CLEAR_GLASSES:-0}" == "1" ]]; then args+=(--reject_clear_glasses); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi

"$PYTHON_BIN" -u "$PROJECT_DIR/filter_ffhq_age_groups_visibility_v1.py" "${args[@]}" "$@"
