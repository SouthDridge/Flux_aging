# Learning Aging Residuals from Prompt-Paired Velocity Fields

This repository contains the v1 training, inference, preprocessing, and
evaluation code for the paper's FLUX-based facial age transformation method.
The method learns an aging-related residual from paired base/detail FLUX
velocity predictions and injects the predicted residual with a controllable
scale.

## Repository scope

The release follows this pipeline:

```text
filtered source manifest
  -> same-state base/detail velocity cache
  -> train/validation index
  -> residual adapter training
  -> FLUX image-to-image inference
  -> Age-MAE / ID cosine / valid-face rate / FID
```

The repository does not include FFHQ images, FLUX weights, InsightFace model
weights, velocity caches, checkpoints, or baseline repositories. Obtain these
from their respective official sources and follow their licenses.

## Main files

- `build_prompt_velocity_cache_v1.py`: builds fixed-timestep paired velocity
  caches and the teacher residual.
- `prepare_age_velocity_index.py`: validates cache shards and creates the
  train/validation index.
- `train_age_velocity_adapter_v1.py`: trains the lightweight residual adapter.
- `infer_age_velocity_adapter_v1.py`: runs base, detail, and adapted FLUX
  trajectories.
- `flux_inference_utils.py`: shared FLUX prompt, latent, transformer,
  scheduler, and decoding utilities used by the v1 inference script.
- `metric/evaluate_aging_metrics.py`: computes age error, identity cosine,
  and valid-face rate with InsightFace.
- `metric/evaluate_fid.py`: computes FID using equal-sized image subsets.

The v1 cache recipe used in the reported experiments fixes the cache and
adapter timestep to `0.45`. During inference, FLUX velocities are evaluated at
the current solver timestep, while the adapter timestep condition remains
fixed at `0.45`.

## Installation

Create an environment with the packages in `requirements-paper.txt`. Install a
PyTorch build compatible with the target CUDA version, then install the
remaining packages. Download a compatible FLUX.1-dev checkpoint and an
InsightFace `buffalo_l` model separately.

## Typical workflow

Set the paths for your machine before running these commands.

```bash
python build_prompt_velocity_cache_v1.py \
  --teacher_meta /path/to/generation_manifest.csv \
  --model_dir /path/to/FLUX.1-dev-hf \
  --output /path/to/cache/part_01.pt \
  --resolution 512 --max_sequence_length 512 \
  --guidance_scale 3.5 --sigma 0.45 --no_masks

python prepare_age_velocity_index.py \
  --cache_root /path/to/cache \
  --output /path/to/age_velocity_index.json \
  --val_fraction 0.1 --seed 42

python train_age_velocity_adapter_v1.py \
  --index /path/to/age_velocity_index.json \
  --output_dir /path/to/checkpoint_dir \
  --steps 5000 --batch_size 8 --lr 2e-4 \
  --lambda_mse 1.0 --lambda_cos 0.1 \
  --lambda_rms 0.1 --lambda_magnitude 1.0 \
  --use_detail_input --seed 42
```

For image generation, use `commands/run_age_velocity_agegroup_seed42.sh` with
`MODEL_DIR`, `CKPT`, `MANIFEST`, `TARGET_AGE`, and `GPU_ID` set explicitly.
For metrics, use the scripts under `commands/` and record the exact evaluator,
detector settings, real-image reference directories, and image counts.

## Reproducibility notes

The training cache, source split, prompt manifests, checkpoint metadata, and
evaluation manifests are part of the experiment provenance and must be saved
alongside a reproduction run. The adapter was trained at a fixed timestep and
is reused across solver timesteps during inference; this cross-timestep
application should not be described as timestep-adaptive training.

See `PAPER_CODE_RELEASE_INVENTORY.md` for the dependency graph, known caveats,
and artifacts that must be recovered from the training server before claiming
full reproduction.
The source code for Flux_aging project
