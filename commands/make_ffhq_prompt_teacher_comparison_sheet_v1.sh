#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/output/ffhq_35_prompt_teacher_demo_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/visual_compare_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -u "$PROJECT_DIR/make_ffhq_prompt_teacher_comparison_sheet_v1.py" \
  --eval_dir "$EVAL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_sources "${NUM_SOURCES:-5}" \
  --seed "${SEED:-42}" \
  --tile_size "${TILE_SIZE:-256}" \
  "$@"
