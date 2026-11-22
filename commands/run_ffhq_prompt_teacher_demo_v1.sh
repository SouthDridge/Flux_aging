#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/ffhq_35_prompt_teacher_demo_v1}"
mkdir -p "${PROJECT_DIR}/logs"
LOG="${PROJECT_DIR}/logs/ffhq_prompt_teacher_$(date +%Y%m%d_%H%M%S)_$$.log"
EXTRA=()
if [[ "${CPU_OFFLOAD:-0}" == "1" ]]; then EXTRA+=(--cpu_offload); fi
echo "START: full log at ${LOG}"
if FFHQ_PROGRESS_FD=3 "$PYTHON_BIN" -u "${PROJECT_DIR}/run_ffhq_prompt_teacher_demo_v1.py" \
  --data_dir "${DATA_DIR:?Set DATA_DIR to the source dataset}" \
  --model_dir "${MODEL_DIR:?Set MODEL_DIR to the local FLUX checkpoint}" \
  --output_dir "$OUTPUT_DIR" \
  --max_sources "${MAX_SOURCES:-30}" \
  --target_ages "${TARGET_AGES:-25,35,45,55,65}" \
  --min_age_confidence "${MIN_AGE_CONFIDENCE:-0.6}" \
  --resolution "${RESOLUTION:-256}" \
  --strength "${STRENGTH:-0.50}" \
  --guidance_scale "${GUIDANCE_SCALE:-3.5}" \
  --steps "${NUM_INFERENCE_STEPS:-28}" \
  --seed "${SEED:-42}" "${EXTRA[@]}" "$@" 3>&1 >"$LOG" 2>&1; then
  tail -n 4 "$LOG"
else
  echo "FAILED: ${LOG}"
  tail -n 60 "$LOG"
  exit 1
fi
