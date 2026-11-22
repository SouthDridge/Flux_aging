#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PY=${PY:-python}
ROOT=${ROOT:-output/result}
OUT=${OUT:-metric/fid_agegroup}
BACKEND=${BACKEND:-auto}
MAX_IMAGES=${MAX_IMAGES:-500}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}

# Set these to real images from the corresponding target-age groups.
: "${REAL_25_35:?set REAL_25_35 to the real target-age directory}"
: "${REAL_25_45:?set REAL_25_45 to the real target-age directory}"
: "${REAL_25_55:?set REAL_25_55 to the real target-age directory}"
: "${REAL_25_65:?set REAL_25_65 to the real target-age directory}"

declare -A REAL_DIRS=(
  [25_35]="$REAL_25_35"
  [25_45]="$REAL_25_45"
  [25_55]="$REAL_25_55"
  [25_65]="$REAL_25_65"
)

variants=(base detail scale1.5 scale2.0 scale2.5 scale3.0)

for group in 25_35 25_45 25_55 25_65; do
  for variant in "${variants[@]}"; do
    fake_dir="$ROOT/age_group_${group}/$variant"
    output_json="$OUT/$group/$variant/fid_summary.json"

    if [[ ! -d "$fake_dir" ]]; then
      echo "SKIP missing fake directory: $fake_dir" >&2
      continue
    fi
    if [[ -f "$output_json" ]]; then
      echo "SKIP completed: $group/$variant"
      continue
    fi

    echo "===== FID $group/$variant ====="
    "$PY" metric/evaluate_fid.py \
      --real_dir "${REAL_DIRS[$group]}" \
      --fake_dir "$fake_dir" \
      --output_json "$output_json" \
      --backend "$BACKEND" \
      --max_images "$MAX_IMAGES" \
      --seed "$SEED" \
      --device "$DEVICE"
  done
done

echo "FID results: $OUT"
