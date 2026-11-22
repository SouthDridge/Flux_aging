#!/usr/bin/env bash
set -u

# Batch inference for the 500-source paired aging evaluation.
# Use PART_INDEX/NUM_PARTS to split the same manifest across GPUs.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

PY=${PY:-python}
GPU_ID=${GPU_ID:-7}
PART_ID=${PART_ID:-01}
PART_INDEX=${PART_INDEX:-0}
NUM_PARTS=${NUM_PARTS:-1}
TOTAL_SOURCES=${TOTAL_SOURCES:-500}
SEED=${SEED:-20261020}
SCALES=${SCALES:-"1.5 2.0"}

MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the local FLUX checkpoint}
CKPT=${CKPT:?set CKPT to the trained adapter checkpoint}
DATASET_DIR=${DATASET_DIR:?set DATASET_DIR to the source age-group directory}
SCRIPT=${SCRIPT:-infer_age_velocity_adapter_v1.py}
ROOT=${ROOT:-output/age_velocity_eval500_new}
MANIFEST=${MANIFEST:-$ROOT/part01/selected_sources.tsv}

mkdir -p "$(dirname "$MANIFEST")"

if [ ! -f "$MANIFEST" ]; then
    "$PY" - "$TOTAL_SOURCES" "$SEED" "$MANIFEST" "$PROJECT_DIR" "$DATASET_DIR" <<'PY'
import json
import random
import sys
from pathlib import Path

total = int(sys.argv[1])
seed = int(sys.argv[2])
manifest = Path(sys.argv[3])
project = Path(sys.argv[4]).resolve()
dataset = Path(sys.argv[5]).resolve()

used = set()
for path in (project / "output").rglob("*.json"):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue

    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "source_image" and isinstance(item, str):
                    used.add(Path(item).stem)
                else:
                    scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)

    scan(data)

candidates = sorted(
    path for path in dataset.glob("*.png")
    if path.stem not in used
)
if len(candidates) < total:
    raise RuntimeError(f"only {len(candidates)} unused images; need {total}")

selected = random.Random(seed).sample(candidates, total)
with manifest.open("w") as handle:
    handle.write("image_id\tsource_image\n")
    for index, path in enumerate(selected):
        handle.write(f"{index:03d}\t{path}\n")

print(f"selected={len(selected)} manifest={manifest}")
PY
else
    echo "reuse manifest=$MANIFEST"
fi

line_index=0
while IFS=$'\t' read -r image_id source_image; do
    if [ "$image_id" = "image_id" ]; then
        continue
    fi

    if [ $((line_index % NUM_PARTS)) -ne "$PART_INDEX" ]; then
        line_index=$((line_index + 1))
        continue
    fi
    line_index=$((line_index + 1))

    for scale in $SCALES; do
        tag=$(printf '%s' "$scale" | tr -d '.')
        out="$ROOT/part01/scale${tag}/source_${image_id}"
        log="$ROOT/part01/logs/source_${image_id}_scale${tag}.log"
        mkdir -p "$out" "$(dirname "$log")"

        if [ -f "$out/diagnostics.json" ]; then
            echo "SKIP source=$image_id scale=$scale"
            continue
        fi

        echo "===== source=$image_id scale=$scale gpu=$GPU_ID ====="
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$SCRIPT" \
            --model_dir "$MODEL_DIR" \
            --checkpoint "$CKPT" \
            --source_image "$source_image" \
            --output_dir "$out" \
            --source_age 25 \
            --target_age 65 \
            --steps 40 \
            --strength 0.5 \
            --adapter_scale "$scale" \
            --adapter_timestep 0.45 \
            --guidance_scale 3.5 \
            --seed 42 \
            --device cuda \
            --dtype bf16 \
            --skip_if_complete \
            2>&1 | tee "$log"
    done
done < "$MANIFEST"

echo "DONE part=$PART_ID index=$PART_INDEX/$NUM_PARTS root=$ROOT/part01"
