#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
META_ROOT="${META_ROOT:-$ROOT/output/ffhq_teacher_shards_v1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/ffhq_age_adapter_cache_v1}"
PY="${PY:-python}"
# Comma-separated shard names to defer, for example:
# SKIP_SHARDS=20_29_part_03,20_29_part_07 bash build_all_age_velocity_caches.sh
SKIP_SHARDS="${SKIP_SHARDS:-}"
MODEL="${MODEL:?Set MODEL to the local FLUX.1-dev checkpoint directory}"

mkdir -p "$OUT_ROOT"
cd "$ROOT"

for meta in "$META_ROOT"/*/generation_manifest.csv; do
  shard=$(basename "$(dirname "$meta")")
  case ",$SKIP_SHARDS," in
    *,"$shard",*)
      echo "SKIP deferred $shard"
      continue
      ;;
  esac
  out="$OUT_ROOT/${shard}_base_detail_velocity_512.pt"
  existing=$(find "$OUT_ROOT" -maxdepth 1 -type f \
    -name "${shard}*.pt" -print -quit)
  if [[ -n "$existing" ]]; then
    echo "SKIP existing $existing"
    continue
  fi
  echo "==== building $shard ===="
  "$PY" build_prompt_velocity_cache_v1.py \
    --teacher_meta "$meta" \
    --output "$out" \
    --model_dir "$MODEL" \
    --project_root "$ROOT" \
    --device cuda \
    --dtype bf16 \
    --resolution 512 \
    --max_sequence_length 512 \
    --guidance_scale 3.5 \
    --sigma 0.45 \
    --no_masks
done

echo "DONE all velocity caches: $OUT_ROOT"
