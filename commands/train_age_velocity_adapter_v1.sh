#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INDEX="${INDEX:?Set INDEX to the prepared age velocity index JSON}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for adapter checkpoints}"

exec "$PYTHON_BIN" "$ROOT/train_age_velocity_adapter_v1.py" \
  --index "$INDEX" \
  --output_dir "$OUTPUT_DIR" \
  --steps "${STEPS:-5000}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --lr "${LR:-2e-4}" \
  --lambda_mse "${LAMBDA_MSE:-1.0}" \
  --lambda_cos "${LAMBDA_COS:-0.1}" \
  --lambda_rms "${LAMBDA_RMS:-0.1}" \
  --lambda_magnitude "${LAMBDA_MAGNITUDE:-1.0}" \
  --seed "${SEED:-42}" \
  --use_detail_input \
  --device "${DEVICE:-cuda}"
