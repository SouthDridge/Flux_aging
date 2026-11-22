#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST_CSV="${MANIFEST_CSV:-}"
SOURCE_DIR="${SOURCE_DIR:-}"
GENERATED_DIR="${GENERATED_DIR:-}"
REAL_DIR="${REAL_DIR:-}"
FAKE_DIR="${FAKE_DIR:-${GENERATED_DIR}}"
TARGET_AGE="${TARGET_AGE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/aging_metrics_eval}"
INSIGHTFACE_PROVIDER="${INSIGHTFACE_PROVIDER:-cpu}"
FID_BACKEND="${FID_BACKEND:-auto}"
MAX_IMAGES="${MAX_IMAGES:-0}"

test -f "${PROJECT_DIR}/metric/evaluate_aging_metrics.py"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

args=(
  --output_dir "${OUTPUT_DIR}"
  --insightface_provider "${INSIGHTFACE_PROVIDER}"
  --fid_backend "${FID_BACKEND}"
  --max_images "${MAX_IMAGES}"
)

if [[ -n "${MANIFEST_CSV}" ]]; then
  test -f "${MANIFEST_CSV}"
  args+=(--manifest_csv "${MANIFEST_CSV}")
fi
if [[ -n "${SOURCE_DIR}" ]]; then
  test -d "${SOURCE_DIR}"
  args+=(--source_dir "${SOURCE_DIR}")
fi
if [[ -n "${GENERATED_DIR}" ]]; then
  test -d "${GENERATED_DIR}"
  args+=(--generated_dir "${GENERATED_DIR}")
fi
if [[ -n "${TARGET_AGE}" ]]; then
  args+=(--target_age "${TARGET_AGE}")
fi
if [[ -n "${REAL_DIR}" ]]; then
  test -d "${REAL_DIR}"
  args+=(--real_dir "${REAL_DIR}")
fi
if [[ -n "${FAKE_DIR}" ]]; then
  test -d "${FAKE_DIR}"
  args+=(--fake_dir "${FAKE_DIR}")
fi

python -u "${PROJECT_DIR}/metric/evaluate_aging_metrics.py" "${args[@]}"
