"""Build a same-state base/detail FLUX velocity cache.

The old V2 ``teacher_cache.pt`` stores a plan and records, not tensors. This
program replays those records and saves the tensors required by the AGE+ID
adapter:

    base_velocity, age_velocity, teacher_delta, editable, protected, background

Both prompts are evaluated at exactly the same source latent, noise and sigma.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_meta", required=True)
    p.add_argument("--output", required=False,
                   default="output/ffhq_prompt_teacher_v1/prompt_id_cache.pt")
    p.add_argument("--model_dir", default=None)
    p.add_argument("--project_root", default=".")
    p.add_argument("--mask_root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--sigma", type=float, default=0.45)
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect_only", action="store_true")
    p.add_argument(
        "--no_masks",
        action="store_true",
        help="Skip editable/protected/background mask lookup for pure velocity caches.",
    )
    return p.parse_args()


def load_object(path: Path):
    return torch.load(path, map_location="cpu")


def load_teacher_records(path: Path):
    """Load either the old metadata PT or a prompt-teacher CSV manifest."""
    if path.suffix.lower() != ".csv":
        return load_object(path)

    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    grouped = {}
    for row in rows:
        condition = str(row.get("condition", "")).strip().lower()
        source_id = row.get("source_id") or row.get("source_image") or row.get("source_age")
        target_age = row.get("target_age") or row.get("target_age_midpoint")
        key = (str(source_id), str(target_age))
        record = grouped.setdefault(key, {})
        record.update({
            "source_id": row.get("source_id", source_id),
            "source_image": row.get("source_image") or row.get("source_path"),
            "source_age": row.get("source_age") or row.get("source_age_midpoint", 35.0),
            "target_age": target_age,
            "sigma": row.get("sigma", row.get("timestep")),
        })
        prompt = row.get("prompt") or row.get("text_prompt")
        generated = (
            row.get("image_path") or row.get("output_path") or
            row.get("generated_image") or row.get("output_image") or
            row.get("result_path")
        )
        if condition == "base":
            record["base_prompt"] = prompt
            record["base_image"] = generated
        elif condition == "detail":
            record["detail_prompt"] = prompt
            record["detail_image"] = generated

    records = [record for record in grouped.values()
               if record.get("base_prompt") and record.get("detail_prompt")]
    return {"version": "csv_manifest", "records": records}


def first_value(obj: Dict[str, Any], names: Iterable[str], default=None):
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return default


def flatten_record(record):
    if not isinstance(record, dict):
        return {"value": record}
    out = dict(record)
    for nested_name in ("metadata", "case", "source", "conditioning", "sample"):
        nested = out.get(nested_name)
        if isinstance(nested, dict):
            for k, v in nested.items():
                out.setdefault(k, v)
    return out


def print_record_schema(obj):
    records = obj.get("records", []) if isinstance(obj, dict) else []
    print(f"top_level_type={type(obj).__name__}")
    if isinstance(obj, dict):
        print("top_level_keys=" + ",".join(sorted(obj.keys())))
    print(f"record_count={len(records)}")
    for i, raw in enumerate(records[:3]):
        record = flatten_record(raw)
        print(f"record_{i}_type={type(raw).__name__}")
        print("record_%d_keys=%s" % (i, ",".join(sorted(record.keys()))))
        for key, value in sorted(record.items()):
            if hasattr(value, "shape"):
                print(f"record_{i}.{key}=tensor shape={tuple(value.shape)}")
            else:
                print(f"record_{i}.{key}={repr(value)[:240]}")


def resolve_path(value, project_root: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    candidate = project_root / path
    return candidate if candidate.is_file() else None


def get_source_image(record, project_root: Path) -> Path:
    value = first_value(record, (
        "source_image", "source_path", "image_path", "image", "file_path",
        "source_file", "relative_path", "input_image",
    ))
    path = resolve_path(value, project_root)
    if path is None:
        raise FileNotFoundError(f"source image not found for record: {value!r}")
    return path


def get_age(record, names, default):
    value = first_value(record, names, default)
    return float(value)


def make_prompts(record, source_age: float, target_age: float):
    base = first_value(record, ("base_prompt", "neutral_prompt", "prompt_base"))
    detail = first_value(record, ("detail_prompt", "aging_prompt", "age_prompt", "prompt_detail"))
    if base is None:
        base = "a realistic close-up portrait photograph of a person"
    if detail is None:
        detail = (
            f"a realistic close-up portrait photograph of a {target_age:.0f}-year-old person, "
            "with natural age-related facial changes, realistic wrinkles, mature facial features, "
            "slightly sagging skin, and subtle age-related texture"
        )
    return str(base), str(detail)


def dtype_from_name(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _mask_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip().lower()
    return str(value).strip().lower()


def _named_masks(payload):
    """Expand an NPZ payload containing stacked ``masks`` and ``names``."""
    if not isinstance(payload, dict) or "masks" not in payload or "names" not in payload:
        return {}
    stacked = np.asarray(payload["masks"])
    names = [_mask_name(x) for x in np.asarray(payload["names"]).reshape(-1)]
    if stacked.ndim != 3 or len(names) == 0:
        return {}
    if stacked.shape[0] == len(names):
        stacked = stacked
    elif stacked.shape[-1] == len(names):
        stacked = np.moveaxis(stacked, -1, 0)
    else:
        return {}
    return {name: stacked[index] for index, name in enumerate(names)}


def _merge_named_region_masks(payload):
    """Map MediaPipe named regions to the three training masks."""
    named = _named_masks(payload)
    if not named:
        return

    protected_terms = (
        "eye", "iris", "pupil", "eyelid", "lip", "mouth", "nose",
        "brow", "eyebrow", "identity", "protected",
    )
    background_terms = ("background", "bg", "hair", "outside")
    editable_terms = (
        "editable", "age", "skin", "cheek", "forehead", "temple",
        "chin", "jaw", "face", "wrinkle",
    )

    def union(names):
        selected = [np.asarray(named[name]).astype(np.float32) for name in names]
        if not selected:
            return None
        result = np.zeros_like(selected[0], dtype=np.float32)
        for value in selected:
            result = np.maximum(result, value)
        return result

    protected_names = [
        name for name in named
        if any(term in name for term in protected_terms)
    ]
    background_names = [
        name for name in named
        if any(term in name for term in background_terms)
    ]
    editable_names = [
        name for name in named
        if any(term in name for term in editable_terms)
        and name not in protected_names
        and name not in background_names
    ]

    protected = union(protected_names)
    background = union(background_names)
    editable = union(editable_names)

    if protected is not None:
        payload.setdefault("protected", protected)
    if background is not None:
        payload.setdefault("background", background)
    if editable is not None:
        payload.setdefault("editable", editable)

    # If the NPZ has named face regions but no explicit editable/background
    # masks, derive them from the available regions without overlapping eyes,
    # lips, nose, brows, or the background.
    if payload.get("background") is None and payload.get("editable") is not None and payload.get("protected") is not None:
        payload["background"] = 1.0 - np.maximum(
            np.asarray(payload["editable"]), np.asarray(payload["protected"])
        )
    if payload.get("editable") is None and payload.get("protected") is not None:
        if background is not None:
            payload["editable"] = 1.0 - np.maximum(
                np.asarray(payload["protected"]), np.asarray(background)
            )
        else:
            payload["editable"] = 1.0 - np.asarray(payload["protected"])


def load_pipeline(model_dir: str, device: torch.device, dtype: torch.dtype):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_image(pipe, image_path: Path, resolution: int, device, dtype):
    image = Image.open(image_path).convert("RGB")
    processor = getattr(pipe, "image_processor", None)
    if processor is not None:
        pixels = processor.preprocess(image, height=resolution, width=resolution)
    else:
        pixels = torch.from_numpy(__import__("numpy").array(image.resize((resolution, resolution))))
        pixels = pixels.permute(2, 0, 1).float() / 127.5 - 1.0
        pixels = pixels.unsqueeze(0)
    pixels = pixels.to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior = pipe.vae.encode(pixels).latent_dist
        latents = posterior.mode()
    scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift) * scaling
    return pack_latents(latents)


def pack_latents(latents):
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError(f"VAE latent size must be even, got {h}x{w}")
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * 4)


def encode_prompt(pipe, prompt, device, max_sequence_length):
    result = pipe.encode_prompt(
        prompt=[prompt], prompt_2=[prompt], device=device,
        num_images_per_prompt=1, max_sequence_length=max_sequence_length,
    )
    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected encode_prompt result: {type(result)}")
    if len(result) < 3:
        raise ValueError("encode_prompt returned fewer than three values")
    prompt_embeds, pooled, text_ids = result[:3]
    return prompt_embeds, pooled, text_ids


def image_ids(pipe, latents, device, dtype):
    if hasattr(pipe, "_prepare_latent_image_ids"):
        h = max(1, int(latents.shape[1] ** 0.5))
        while h > 1 and latents.shape[1] % h:
            h -= 1
        w = latents.shape[1] // h
        # The Diffusers helper expects the unpacked VAE latent height/width
        # and internally divides both by two when creating packed token IDs.
        return pipe._prepare_latent_image_ids(1, h * 2, w * 2, device, dtype)
    return torch.zeros((latents.shape[1], 3), device=device, dtype=dtype)


def call_transformer(pipe, latents, sigma, prompt_state, img_ids, guidance_scale):
    kwargs = {
        "hidden_states": latents,
        "timestep": torch.full((latents.shape[0],), float(sigma), device=latents.device, dtype=latents.dtype),
        "encoder_hidden_states": prompt_state[0],
        "pooled_projections": prompt_state[1],
        "txt_ids": prompt_state[2],
        "img_ids": img_ids,
        "return_dict": False,
    }
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        kwargs["guidance"] = torch.full((latents.shape[0],), guidance_scale,
                                         device=latents.device, dtype=latents.dtype)
    try:
        output = pipe.transformer(**kwargs)
    except TypeError as error:
        if "guidance" not in str(error):
            raise
        kwargs.pop("guidance", None)
        output = pipe.transformer(**kwargs)
    return output[0] if isinstance(output, tuple) else output.sample


def load_masks(record, mask_root: Optional[Path], velocity, latent_hw, allow_missing=False):
    direct = record.get("masks") if isinstance(record.get("masks"), dict) else record
    values = {k: direct.get(k) for k in ("editable", "protected", "background")}
    payload = None
    payload_path = None
    image_values = []
    if not all(v is not None for v in values.values()) and mask_root is not None:
        manifest_path = mask_root / "manifest.json"
        image_values = []
        for name in (
            "base_image", "base_path", "base_output", "base_generated_image",
            "detail_image", "detail_path", "detail_output", "detail_generated_image",
            "generated_image", "output_image", "source_image", "source_path",
            "image_path", "image", "file_path", "source_file", "relative_path",
            "input_image",
        ):
            value = record.get(name)
            if value not in (None, ""):
                image_values.append(str(value).replace("\\", "/"))
        manifest_rows = []
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_rows = manifest.get("images", []) if isinstance(manifest, dict) else manifest
        exact_matches = []
        for row in manifest_rows:
            manifest_image = str(row.get("image", "")).replace("\\", "/")
            if manifest_image in image_values:
                exact_matches.append(row)
        selected = exact_matches[0] if exact_matches else None

        if selected is None:
            record_names = {Path(value).name for value in image_values}
            basename_matches = [
                row for row in manifest_rows
                if Path(str(row.get("image", ""))).name in record_names
            ]
            selected = basename_matches[0] if basename_matches else None

        if selected is None:
            source_id = str(first_value(
                record, ("source_id", "image_number", "file_name"), ""
            ))
            target_age = first_value(
                record, ("target_age", "target_age_midpoint"), None
            )
            target_token = None
            if target_age is not None:
                target_token = f"age_{float(target_age):g}"
            source_matches = []
            for row in manifest_rows:
                image = str(row.get("image", ""))
                image_name = Path(image).name
                matches_source = source_id and source_id in image_name
                matches_age = target_token is None or target_token in image
                if matches_source and matches_age:
                    source_matches.append(row)
            selected = source_matches[0] if source_matches else None
        if selected is not None and selected.get("mask"):
            payload_path = mask_root / str(selected["mask"])
        if payload_path is not None and payload_path.is_file():
            if payload_path.suffix == ".npz":
                with np.load(payload_path, allow_pickle=False) as loaded:
                    payload = {key: loaded[key] for key in loaded.files}
            else:
                loaded = load_object(payload_path)
                if isinstance(loaded, dict):
                    payload = loaded
        if payload is None:
            source_id = first_value(record, ("source_id", "image_number", "file_name", "source_image"), "")
            stem = Path(str(source_id)).stem
            candidates = sorted(mask_root.rglob(f"*{stem}*"))
            for candidate in candidates:
                if candidate.suffix in (".pt", ".pth"):
                    loaded = load_object(candidate)
                    if isinstance(loaded, dict):
                        payload = loaded
                        break
    if payload is not None:
        _merge_named_region_masks(payload)
        aliases = {
            "editable": (
                "editable", "editable_mask", "editable_region", "age_region",
                "age", "age_mask", "local_editable",
            ),
            "protected": (
                "protected", "protected_mask", "identity", "identity_mask",
                "identity_protected", "face_protected",
            ),
            "background": (
                "background", "background_mask", "bg", "bg_mask",
                "background_region",
            ),
        }
        for name, candidates in aliases.items():
            if values[name] is None:
                for candidate in candidates:
                    if candidate in payload:
                        values[name] = payload[candidate]
                        break
        if values["background"] is None and values["editable"] is not None and values["protected"] is not None:
            values["background"] = 1.0 - np.maximum(
                np.asarray(values["editable"]), np.asarray(values["protected"])
            )
    if not all(v is not None for v in values.values()) and allow_missing:
        token_count = velocity.shape[1] if velocity.ndim == 3 else velocity.shape[-1]
        editable = torch.ones((1, token_count, 1), dtype=velocity.dtype)
        zero = torch.zeros((1, token_count, 1), dtype=velocity.dtype)
        return editable, zero, zero.clone()
    if not all(v is not None for v in values.values()):
        available = sorted(payload.keys()) if isinstance(payload, dict) else []
        named = sorted(_named_masks(payload).keys()) if isinstance(payload, dict) else []
        raise FileNotFoundError(
            "could not find editable/protected/background masks; "
            f"record_image_candidates={image_values if mask_root is not None else []}; "
            f"mask_keys={available}; mask_names={named}; "
            "provide --mask_root or add mask paths to each teacher record"
        )
    def to_mask(value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.float()
        if value.ndim == 2:
            value = value[None, None]
        elif value.ndim == 3:
            value = value[None] if value.shape[0] not in (1, 3) else value[:, None]
        if value.ndim == 4 and velocity.ndim == 3:
            value = F.interpolate(value, size=latent_hw, mode="area").flatten(2).transpose(1, 2)
        elif value.ndim == 4 and velocity.ndim == 4:
            value = F.interpolate(value, size=velocity.shape[-2:], mode="area")
        return value.to(dtype=velocity.dtype)
    editable = to_mask(values["editable"])
    protected = to_mask(values["protected"])
    background = to_mask(values["background"])
    return editable, protected, background


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    metadata_path = resolve_path(args.teacher_meta, project_root) or Path(args.teacher_meta)
    obj = load_teacher_records(metadata_path)
    if args.inspect_only:
        print_record_schema(obj)
        return
    if not args.model_dir:
        raise ValueError("--model_dir is required when building velocity tensors")
    if not isinstance(obj, dict) or not isinstance(obj.get("records"), list):
        raise ValueError("teacher_meta must be a dict containing a records list")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    pipe = load_pipeline(args.model_dir, device, dtype)
    mask_root = Path(args.mask_root).resolve() if args.mask_root else None
    records = obj["records"]
    if args.max_records > 0:
        records = records[:args.max_records]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    base_rows = []; age_rows = []; delta_rows = []
    editable_rows = []; protected_rows = []; background_rows = []
    metadata = []
    for index, raw in enumerate(records, 1):
        record = flatten_record(raw)
        source_age = get_age(record, ("source_age", "source_age_midpoint", "age", "source_age_group_midpoint"), 35.0)
        target_age = get_age(record, ("target_age", "target_age_midpoint", "eval_age_midpoint"), 65.0)
        sigma = float(first_value(record, ("sigma", "noise_level", "timestep"), args.sigma))
        image_path = get_source_image(record, project_root)
        base_prompt, age_prompt = make_prompts(record, source_age, target_age)
        source_latents = encode_image(pipe, image_path, args.resolution, device, dtype)
        noise = torch.randn(source_latents.shape, generator=generator, device=device, dtype=source_latents.dtype)
        noisy = (1.0 - sigma) * source_latents + sigma * noise
        with torch.no_grad():
            base_state = encode_prompt(pipe, base_prompt, device, args.max_sequence_length)
            age_state = encode_prompt(pipe, age_prompt, device, args.max_sequence_length)
            ids = image_ids(pipe, noisy, device, noisy.dtype)
            base_velocity = call_transformer(pipe, noisy, sigma, base_state, ids, args.guidance_scale).float().cpu()
            age_velocity = call_transformer(pipe, noisy, sigma, age_state, ids, args.guidance_scale).float().cpu()
        latent_hw = (max(1, int(noisy.shape[1] ** 0.5)), max(1, int(noisy.shape[1] ** 0.5)))
        if latent_hw[0] * latent_hw[1] != noisy.shape[1]:
            latent_hw = (1, noisy.shape[1])
        editable, protected, background = load_masks(
            record,
            mask_root,
            base_velocity,
            latent_hw,
            allow_missing=args.no_masks,
        )
        base_rows.append(base_velocity)
        age_rows.append(age_velocity)
        delta_rows.append(age_velocity - base_velocity)
        editable_rows.append(editable)
        protected_rows.append(protected)
        background_rows.append(background)
        metadata.append({"index": index, "source_image": str(image_path),
                         "source_age": source_age, "target_age": target_age,
                         "sigma": sigma, "base_prompt": base_prompt,
                         "age_prompt": age_prompt})
        print(f"record={index}/{len(records)} source={image_path.name} target_age={target_age:.1f}", flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "base_velocity": torch.cat(base_rows, dim=0),
        "age_velocity": torch.cat(age_rows, dim=0),
        "detail_velocity": torch.cat(age_rows, dim=0),
        "teacher_delta": torch.cat(delta_rows, dim=0),
        "editable": torch.cat(editable_rows, dim=0),
        "protected": torch.cat(protected_rows, dim=0),
        "background": torch.cat(background_rows, dim=0),
        "source_age": torch.tensor([m["source_age"] for m in metadata]),
        "target_age": torch.tensor([m["target_age"] for m in metadata]),
        "timestep": torch.tensor([m["sigma"] for m in metadata]),
        "records": metadata,
        "definition": "same source latent/noise/sigma; detail_velocity - base_velocity",
    }
    torch.save(result, output)
    output.with_suffix(".json").write_text(json.dumps({"records": metadata}, indent=2), encoding="utf-8")
    print(f"DONE records={len(metadata)} output={output}")


if __name__ == "__main__":
    main()
