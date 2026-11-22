#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
META="${TEACHER_META:-$ROOT/output/ffhq_mediapipe_prompt_adapter_v2_steps5000/teacher_cache.pt}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local FLUX checkpoint directory}"
MASK_ROOT="${MASK_ROOT:-$ROOT/output/ffhq_mediapipe_adapter_masks_v1}"
OUTPUT="${OUTPUT_CACHE:-$ROOT/output/ffhq_prompt_teacher_v1/prompt_id_cache.pt}"

exec "$PYTHON" -u "$ROOT/build_prompt_velocity_cache_v1.py" \
  --teacher_meta "$META" \
  --model_dir "$MODEL_DIR" \
  --mask_root "$MASK_ROOT" \
  --output "$OUTPUT" \
  --device "${DEVICE:-cuda}" \
  --max_records "${MAX_RECORDS:-0}" \
  --sigma "${SIGMA:-0.45}"
