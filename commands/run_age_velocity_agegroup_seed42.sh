#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

PY=${PY:-python}
GPU_ID=${GPU_ID:?set GPU_ID}
TARGET_AGE=${TARGET_AGE:?set TARGET_AGE}
SOURCE_AGE=${SOURCE_AGE:-25}
SEED=${SEED:-42}
SCALES=${SCALES:-"1.5 2.0 2.5 3.0"}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the local FLUX checkpoint}
CKPT=${CKPT:?set CKPT to the trained adapter checkpoint}
SCRIPT=${SCRIPT:-infer_age_velocity_adapter_v1.py}
ROOT=${ROOT:-output/result}
MANIFEST=${MANIFEST:-output/age_velocity_eval500_new/part01/selected_sources.tsv}

GROUP="${SOURCE_AGE}_${TARGET_AGE}"
GROUP_ROOT="$ROOT/age_group_${GROUP}"
TMP_ROOT="$ROOT/_tmp/$GROUP"
mkdir -p "$GROUP_ROOT/source" "$GROUP_ROOT/base" "$GROUP_ROOT/detail" \
    "$GROUP_ROOT/diagnostics" "$TMP_ROOT"

if [ ! -f "$MANIFEST" ]; then
    echo "missing manifest: $MANIFEST" >&2
    exit 1
fi

echo "START group=$GROUP gpu=$GPU_ID seed=$SEED scales=$SCALES"

while IFS=$'\t' read -r image_id source_image; do
    [ "$image_id" = "image_id" ] && continue
    index=$((10#$image_id + 1))
    stem=$(printf '%05d_%s_to_%s_seed%s' "$index" "$SOURCE_AGE" "$TARGET_AGE" "$SEED")

    [ -f "$GROUP_ROOT/source/$stem.png" ] || cp "$source_image" "$GROUP_ROOT/source/$stem.png"

    for scale in $SCALES; do
        tag="$scale"
        scale_dir="$GROUP_ROOT/scale${tag}"
        tmp="$TMP_ROOT/source_${image_id}/scale${tag}"
        mkdir -p "$scale_dir" "$tmp"

        if [ -f "$scale_dir/$stem.png" ]; then
            echo "SKIP group=$GROUP source=$image_id scale=$scale"
            continue
        fi

        echo "===== group=$GROUP source=$image_id scale=$scale gpu=$GPU_ID ====="
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$SCRIPT" \
            --model_dir "$MODEL_DIR" \
            --checkpoint "$CKPT" \
            --source_image "$source_image" \
            --output_dir "$tmp" \
            --source_age "$SOURCE_AGE" \
            --target_age "$TARGET_AGE" \
            --steps 40 \
            --strength 0.5 \
            --adapter_scale "$scale" \
            --adapter_timestep 0.45 \
            --guidance_scale 3.5 \
            --seed "$SEED" \
            --device cuda \
            --dtype bf16

        if [ ! -f "$tmp/aging_adapter.png" ]; then
            echo "FAILED missing adapter image: $tmp/aging_adapter.png" >&2
            continue
        fi

        cp "$tmp/aging_adapter.png" "$scale_dir/$stem.png"
        cp "$tmp/diagnostics.json" "$GROUP_ROOT/diagnostics/${stem}_scale${tag}.json"

        if [ "$scale" = "1.5" ]; then
            cp "$tmp/aging_base.png" "$GROUP_ROOT/base/$stem.png"
            cp "$tmp/aging_detail.png" "$GROUP_ROOT/detail/$stem.png"
        fi
        echo "DONE group=$GROUP source=$image_id scale=$scale"
    done
done < "$MANIFEST"

echo "DONE group=$GROUP root=$GROUP_ROOT"
