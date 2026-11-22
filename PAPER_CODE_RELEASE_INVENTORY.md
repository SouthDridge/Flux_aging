# Paper code release inventory

Audited on 2026-11-22 against the local workspace. Scope: the paper's
detail-conditioned `train_age_velocity_adapter_v1.py` /
`infer_age_velocity_adapter_v1.py` route, not later v3-v50 experiments.

This is a source inventory, NOT a tested, portable release bundle. Actual
server checkpoints, training indexes, manifests, package versions and baseline
repositories still need to be matched to the reported experiments.

## 1. Core files to include

Preserve the current relative paths until imports and launchers are deliberately
refactored and tested.

| File | Role |
| --- | --- |
| `build_prompt_velocity_cache_v1.py` | Construct same-state base/detail velocity pairs and teacher residuals; supports CSV prompt manifests and legacy metadata. |
| `prepare_age_velocity_index.py` | Validate cache shards and build source-image-disjoint training/validation index. |
| `train_age_velocity_adapter_v1.py` | Adapter architecture, four training losses, optimization and checkpoint saving. |
| `evaluate_age_velocity_adapter_v1.py` | Evaluate adapter residual predictions on indexed cache records. |
| `infer_age_velocity_adapter_v1.py` | Final detail-conditioned image inference; separate base/detail/adapted trajectories. |
| `flux_inference_utils.py` | Shared FLUX helpers: prompt encoding, transformer calls, VAE packing/decoding, scheduler setup and contact sheets. |
| `metric/evaluate_aging_metrics.py` | Age-MAE, identity cosine and face validity; optional FID support. |
| `metric/evaluate_fid.py` | Dedicated FID with reproducible equal-sized real/fake subsets. |

The former prompt-ID helper contained several historical adapter versions.
They were not methods used in this paper and have been removed from this
release. The v1 inference entry point now imports only the shared utilities in
`flux_inference_utils.py` and loads `AdaptiveAgeVelocityAdapter` directly from
`train_age_velocity_adapter_v1.py`.

## 2. Existing launchers to include after path cleanup

| File | Role / caveat |
| --- | --- |
| `build_all_age_velocity_caches.sh` | Batch cache construction from `ffhq_teacher_shards_v1/*/generation_manifest.csv`; explicitly uses sigma 0.45, resolution 512 and `--no_masks`. Hard-coded server paths. |
| `commands/build_prompt_velocity_cache_v1.sh` | Alternative single-cache wrapper; defaults to legacy teacher metadata/mask inputs. Not identical to the batch recipe. |
| `commands/run_age_velocity_agegroup_seed42.sh` | Per-target-age scale sweep on a selected source manifest; final v1 inference entry point. |
| `commands/run_age_velocity_eval500_part01.sh` | Source selection and batch inference launcher; preserve the actual selected manifest instead of resampling for the paper. |
| `commands/train_age_velocity_adapter_v1.sh` | Portable training wrapper; requires an explicit index and output directory. |
| `commands/evaluate_agegroup_fid.sh` | Base/detail/scale FID sweep across four target groups. Explicitly select CleanFID for the published protocol. |
| `commands/evaluate_aging_metrics.sh` | Age/ID launcher calling the released `metric/evaluate_aging_metrics.py` implementation. |

The included v1 training wrapper is a portable interface, not proof of the
historical command used to create the reported checkpoint. Recover the actual
training command/log from the server. Do not substitute
`commands/train_age_velocity_adapter_v3.sh`.

## 3. Data preparation: relevant local files, provenance still to verify

These files provide preprocessing capabilities. Their presence does not prove
that every one was executed in the final experiment.

| File | Role |
| --- | --- |
| `insightface_check/check_ffhq_insightface.py` | Face screening and diagnostic CSV. |
| `insightface_select.py` | Select identity-valid FFHQ images / reuse a face-screening CSV. |
| `age_group_select.py` | Age-group selection from age labels and a face-validity CSV. |
| `filter_ffhq_age_groups_visibility_v1.py` | Visibility/occlusion filtering from available metadata. |
| `commands/filter_ffhq_age_groups_visibility_v1.sh` | Visibility-filter launcher. |
| `make_ffhq_source_manifest_v1.py` | Source manifest field normalization. |
| `make_ffhq_source_shards_v1.py` | Deterministic source sharding. |
| `run_ffhq_prompt_teacher_demo_v1.py` | Constructs prompt jobs and `generation_manifest.csv`; supports preparation without image generation. |
| `commands/run_ffhq_prompt_teacher_demo_v1.sh` | Prompt job launcher. |

Important: the current prompt-demo script uses templates that are not identical
to final v1 inference prompts. Inspect the ACTUAL training generation manifests
and saved cache `records` before declaring identical training/inference prompts.
Do not regenerate historical manifests blindly from today's script defaults.

Recover the original age/occlusion metadata source, filter thresholds, filtering
order, and exact single-face exclusion rule. Some validity checks accept any
detected face; a valid-face CSV alone does not establish exclusion of all
multi-face images. Publish metadata provenance and allowed source IDs/splits,
not private absolute paths or unrestricted copies of FFHQ photographs.

## 4. Optional inspection, figures and tests

- `compare_inference_metrics.py`: additional source/baseline/adapter comparisons;
  not a substitute for the paper's final age/ID evaluator.
- `make_ffhq_prompt_teacher_comparison_sheet_v1.py`
- `commands/make_ffhq_prompt_teacher_comparison_sheet_v1.sh`
- `test_filter_ffhq_age_groups_visibility_v1.py`
- `test_ffhq_prompt_teacher_demo_v1.py`
- `test_ffhq_comparison_sheet_v1.py`

The comparison-sheet utility uses the prompt-demo manifest format. It is not
confirmed to be the script that assembled the final SAM/FADING/MyTM/ours panels.
Recover the actual paper montage/table scripts separately. Overview drawing
scripts are figure-authoring assets, not experiment dependencies.

## 5. Dependency graph

```text
source IDs / age and visibility metadata
  -> filtering and source manifests [verify exact historical recipe]
  -> source shards / generation_manifest.csv [verify actual prompts]
  -> build_prompt_velocity_cache_v1.py
  -> prepare_age_velocity_index.py
  -> train_age_velocity_adapter_v1.py
  -> checkpoint
     -> evaluate_age_velocity_adapter_v1.py
     -> infer_age_velocity_adapter_v1.py
        -> train_age_velocity_adapter_v1.AdaptiveAgeVelocityAdapter
        -> flux_inference_utils.py
        -> base/detail/adapter images and diagnostics
           -> metric/evaluate_aging_metrics.py
           -> metric/evaluate_fid.py
```

With `--no_masks`, the cache building route does not require mask-preparation
scripts. The final adapter loss/injection does not consume region masks. Do not
advertise region-aware supervision based only on optional cached mask fields.

## 6. Release blockers / missing server artifacts

1. Recover the EXACT checkpoint used for each reported result, its SHA256,
   config, step, training index and command. Wrapper default
   `age_velocity_adapter_v1_balanced_1k` is not proof of the actual run or steps.
2. Recover train/validation index and split IDs yielding the reported 38,140 /
   4,236 counts. Audit timestep values on the indexed training rows, not arbitrary
   caches elsewhere under output. Report fixed-time training and cross-time
   inference honestly; do not claim a completed cross-time validation experiment.
3. Recover actual cache prompt records and original generation manifests.
4. Recover the shared 500-source evaluation manifest and baseline filename-to-ID
   mapping. Resolve the historical FADING 412 versus 500 sample discrepancy.
5. Obtain SAM, CUSP, FADING and MyTimeMachine server launch/extraction commands,
   upstream repository URLs/commits, checkpoint identifiers and local patches.
   Those complete baseline repositories were not found in this local inventory.
6. Use identical real reference subsets and evaluation backend across methods.
   Historical CUSP source-reference FID is NOT comparable to target-age-reference
   FID. Publish reference IDs, age bins and real/fake counts per result.
7. Recover final table-generation and qualitative-panel scripts; several steps
   were previously run as inline terminal snippets, not saved local files.
8. Export the actual working server environment and pin tested versions. No
   end-to-end GPU reproduction was performed in this inventory task.

## 7. Two evaluation scripts must not be mixed silently

`evaluate_aging_metrics.py` and `metric/evaluate_aging_metrics.py` are different:

| Setting | Root version | metric/ version |
| --- | --- | --- |
| Default detector size | 640 | 1536 |
| Default detector threshold | 0.2 | 0.05 |
| Path resolution | Basic | Additional mount aliases and diagnostics |
| Manifest image ID | Generated stem | Explicit manifest ID when available |

Keep the root version only if needed to reproduce its historical runs. Choose
the actual final evaluator using logs/result metadata, specify detector settings
explicitly, and update the launcher only after that decision. Do not silently
replace historical evaluator defaults.

The local CleanFID branch in `metric/evaluate_fid.py` does not forward the CLI
`--batch_size` and `--device` arguments to `fid.compute_fid`; those options only
reach its TorchMetrics branch. Document/test device selection and use a single
visible GPU for the historical NCCL workaround. Avoid automatic backend
fallback when claiming a consistent CleanFID protocol.

## 8. External packages and release documentation

Core dependencies to version from the real environment: torch, diffusers,
transformers, accelerate, safetensors, sentencepiece, protobuf, numpy, Pillow,
pandas and tqdm. Evaluation/preprocessing additionally uses insightface,
onnxruntime or onnxruntime-gpu, OpenCV, and clean-fid (plus its dependencies).
The optional TorchMetrics FID route needs torchmetrics, torch-fidelity and
torchvision. This is not a tested version lockfile.

Add a public README with exact commands, data access instructions, actual prompt
templates, cache/index schema, train/inference configuration, evaluation sample
selection and known limitations. Add requirements/environment files, a license
appropriate to your own code, CITATION metadata and a .gitignore. Respect
upstream FLUX, InsightFace, baseline and dataset licenses separately.

Do not upload the entire workspace. Exclude raw datasets, large velocity caches,
FLUX/InsightFace weights, logs with private paths, credentials, .env files,
__pycache__, temporary outputs, unrelated B3FD/v3-v50 experiments, patches/rejects
and obsolete checkpoints. Publish allowed adapter weights separately with a
checksum and provenance; choose and verify the actual final checkpoint first.

No code was moved, no model/data files were packaged, and nothing was uploaded
to GitHub by this inventory task.
