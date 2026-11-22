#!/usr/bin/env python3
"""Visualize the prompt-aging route with and without the trained ID adapter.

This script is intentionally separate from training and cache construction.  It
uses the same FLUX transformer twice at every denoising step:

    base prompt -> base velocity
    aging prompt -> age velocity
    age velocity + ID adapter correction -> final velocity

The two image trajectories therefore share the same initial latent, scheduler,
seed, and source image.  The only difference is the identity correction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import inspect
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")

import numpy as np
import torch
from PIL import Image, ImageDraw

from age_prompt_id_adapter_v2 import PromptIDAdapterV2Config, PromptIdentityAdapterV2
from age_prompt_id_adapter_v6 import (
    PromptIDAdapterV6Config,
    PromptIdentityAdapterV6,
    landmark_face_gate,
)
from age_prompt_id_adapter_v7 import (
    PromptIDAdapterV7Config,
    PromptIdentityAdapterV7,
    landmark_structure_gate,
)
from age_prompt_id_adapter_v8 import PromptIDAdapterV8Config, PromptIdentityAdapterV8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--landmark_cache", required=True)
    p.add_argument("--identity_cache", required=True)
    p.add_argument("--source_image", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source_age", type=float, default=None)
    p.add_argument("--target_age", type=float, default=65.0)
    p.add_argument("--base_prompt", default=None)
    p.add_argument("--age_prompt", default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--max_injection_ratio", type=float, default=0.35)
    p.add_argument(
        "--age_boost",
        type=float,
        default=0.0,
        help="Extrapolate the source-to-target prompt velocity difference: "
        "target_velocity + age_boost * (target_velocity - source_velocity)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_first_record(cache: Dict[str, Any], source_age: Optional[float], target_age: float) -> Dict[str, Any]:
    records = cache.get("records", [])
    if not records:
        return {}
    best = None
    best_score = float("inf")
    for row in records:
        try:
            sa = float(row.get("source_age", source_age if source_age is not None else 0.0))
            ta = float(row.get("target_age", target_age))
        except (TypeError, ValueError):
            continue
        score = abs(ta - target_age) + (0.0 if source_age is None else abs(sa - source_age))
        if score < best_score:
            best, best_score = row, score
    return best or records[0]


def find_condition_index(cache: Dict[str, Any], source_image: str, target_age: float) -> int:
    records = cache.get("records", [])
    wanted = Path(source_image).resolve()
    best_index = 0
    best_score = float("inf")
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        candidate = row.get("source_image")
        if not candidate:
            continue
        same_source = Path(str(candidate)).resolve() == wanted or Path(str(candidate)).name == wanted.name
        if not same_source:
            continue
        try:
            score = abs(float(row.get("target_age", target_age)) - target_age)
        except (TypeError, ValueError):
            score = 0.0
        if score < best_score:
            best_index, best_score = index, score
    return best_index


def prompt_pair(args: argparse.Namespace, record: Dict[str, Any]) -> Tuple[str, str]:
    source_age = args.source_age
    if source_age is None:
        source_age = float(record.get("source_age", 35.0))
    base = args.base_prompt or f"a realistic close-up portrait photograph of a person at age {source_age:g}"
    age = args.age_prompt or (
        f"a realistic close-up portrait photograph of a person at age {args.target_age:g}, "
        "preserving recognizable identity, eye shape, nose shape, mouth shape, and face structure"
    )
    return base, age


def to_device(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype if x.is_floating_point() else x.dtype)
    return x


def unwrap_output(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Unsupported transformer output type: {type(out)!r}")


def call_transformer(transformer: torch.nn.Module, hidden_states: torch.Tensor, cond: Dict[str, Any], timestep: torch.Tensor) -> torch.Tensor:
    kwargs = dict(cond)
    kwargs["hidden_states"] = hidden_states
    kwargs["timestep"] = timestep
    # FLUX versions differ in whether guidance is required by the transformer.
    try:
        return unwrap_output(transformer(**kwargs))
    except TypeError as exc:
        if "guidance" in kwargs:
            kwargs.pop("guidance", None)
            return unwrap_output(transformer(**kwargs))
        raise exc


def encode_prompt(pipe: Any, prompt: str, device: torch.device, dtype: torch.dtype, max_length: int) -> Dict[str, Any]:
    result = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_length,
    )
    if not isinstance(result, (tuple, list)):
        raise TypeError("Unexpected encode_prompt result")
    names = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")
    out = {name: value for name, value in zip(names, result)}
    out = {k: to_device(v, device, dtype) for k, v in out.items() if v is not None}

    # Some diffusers releases return text_ids from the tokenizer path with a
    # length different from prompt_embeds.  FLUX concatenates text and image
    # ids before rotary embedding, so the lengths must agree exactly.
    prompt_embeds = out["prompt_embeds"]
    text_length = int(prompt_embeds.shape[1])
    out["text_ids"] = torch.zeros(
        (text_length, 3), device=device, dtype=prompt_embeds.dtype
    )
    return out


def condition_kwargs(encoded: Dict[str, Any], latent_ids: torch.Tensor, guidance: float) -> Dict[str, Any]:
    out = {
        "encoder_hidden_states": encoded["prompt_embeds"],
        "pooled_projections": encoded["pooled_prompt_embeds"],
        "txt_ids": encoded["text_ids"],
        "img_ids": latent_ids,
    }
    out["txt_ids"] = out["txt_ids"].to(latent_ids.device)
    out["guidance"] = torch.tensor([guidance], device=latent_ids.device, dtype=encoded["prompt_embeds"].dtype)
    return out


def pack_image_latents(pipe: Any, image: Image.Image, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    if hasattr(pipe, "_encode_vae_image"):
        latents = pipe._encode_vae_image(image_tensor, generator=generator)
    else:
        posterior = pipe.vae.encode(image_tensor).latent_dist
        latents = posterior.sample(generator=generator)
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    if latents.shape[1] == 16 and hasattr(pipe, "_pack_latents"):
        packed = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
    else:
        packed = latents
    token_count = int(packed.shape[1])
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise RuntimeError(
            "Expected a square packed FLUX latent grid, "
            f"got latent_tokens={token_count}"
        )
    # Do not use _prepare_latent_image_ids here.  Its expected height/width
    # convention changed between diffusers versions and can apply a second
    # 2x2 reduction.  The transformer only needs one row per packed token.
    latent_ids = torch.zeros((token_count, 3), device=device, dtype=dtype)
    latent_ids[:, 1] = torch.arange(side, device=device, dtype=dtype).repeat_interleave(side)
    latent_ids[:, 2] = torch.arange(side, device=device, dtype=dtype).repeat(side)
    if int(latent_ids.shape[0]) != token_count:
        raise RuntimeError(
            "FLUX image rotary ids do not match packed latent tokens: "
            f"latent_tokens={token_count}, image_ids={int(latent_ids.shape[0])}"
        )
    return packed.to(dtype), latent_ids


def unpack_and_decode(pipe: Any, packed: torch.Tensor, height: int, width: int) -> Image.Image:
    if packed.ndim == 3 and hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    else:
        latents = packed
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def scheduler_shift_kwargs(scheduler: Any, image_sequence_length: int) -> Dict[str, float]:
    """Return the required dynamic-shifting parameter for FLUX schedulers."""
    config = scheduler.config
    if not bool(getattr(config, "use_dynamic_shifting", False)):
        return {}

    base_len = float(getattr(config, "base_image_seq_len", 256))
    max_len = float(getattr(config, "max_image_seq_len", 4096))
    base_shift = float(getattr(config, "base_shift", 0.5))
    max_shift = float(getattr(config, "max_shift", 1.15))
    if max_len <= base_len:
        raise ValueError(
            "Invalid FLUX scheduler config for dynamic shifting: "
            f"base_image_seq_len={base_len}, max_image_seq_len={max_len}"
        )

    slope = (max_shift - base_shift) / (max_len - base_len)
    mu = slope * float(image_sequence_length) + base_shift - slope * base_len
    return {"mu": mu}


def load_adapter(path: str, device: torch.device) -> Optional[torch.nn.Module]:
    """Load the repository adapter when available.

    The adapter module has changed names during experiments, so this keeps the
    visualization script compatible with the existing checkpoint layout.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    module = importlib.import_module("age_prompt_id_adapter_v1")
    module_classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj is torch.nn.Module or not issubclass(obj, torch.nn.Module):
            continue
        if obj.__module__ == module.__name__:
            module_classes.append((name, obj))

    if not module_classes:
        raise ImportError(
            "age_prompt_id_adapter_v1.py does not define a torch.nn.Module Adapter. "
            "Available module classes: "
            + ", ".join(name for name, _ in inspect.getmembers(module, inspect.isclass))
        )

    config = state.get("config", {}) if isinstance(state, dict) else {}
    if not isinstance(config, dict):
        config = {}
    extra_config = state.get("adapter_config", {}) if isinstance(state, dict) else {}
    if isinstance(extra_config, dict):
        config = {**extra_config, **config}
    weights = state.get("state_dict", state.get("model_state_dict", state.get("model", state))) \
        if isinstance(state, dict) else state

    # Prefer the class recorded in the checkpoint, then conventional names,
    # then the only local nn.Module.  This handles the naming drift across the
    # smoke, 10k-step, and image-inference experiments.
    recorded_name = ""
    if isinstance(state, dict):
        recorded_name = str(state.get("class_name", state.get("adapter_class", "")))
    preferred = [recorded_name, "PromptIDAdapter", "AgePromptIDAdapter", "PromptIDFluxAdapter"]
    ordered = []
    for wanted in preferred:
        for name, cls in module_classes:
            if wanted and name == wanted and (name, cls) not in ordered:
                ordered.append((name, cls))
    for item in module_classes:
        if item not in ordered:
            ordered.append(item)

    config_values = [config]
    for config_name, config_cls in inspect.getmembers(module, inspect.isclass):
        if "config" not in config_name.lower() or config_cls.__module__ != module.__name__:
            continue
        try:
            config_values.append(config_cls(**config))
        except Exception:
            continue

    def constructor_attempts(cls):
        # PromptIdentityAdapterV1 uses Adapter(config), while earlier smoke
        # versions used Adapter(**config).  Keep both forms compatible.
        for config_value in config_values:
            yield lambda config_value=config_value: cls(config_value)
        yield lambda: cls(**config)
        yield lambda: cls()

    errors = []
    for name, cls in ordered:
        for make_model in constructor_attempts(cls):
            try:
                model = make_model()
                model.load_state_dict(weights, strict=True)
                return model.to(device).eval()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        # A strict load can fail only because the checkpoint contains harmless
        # metadata prefixes; allow the repository's current model to proceed
        # when the parameter names otherwise match.
        for make_model in constructor_attempts(cls):
            try:
                model = make_model()
                missing, unexpected = model.load_state_dict(weights, strict=False)
                loaded = len(model.state_dict()) - len(missing)
                if loaded > 0 and loaded >= max(1, len(model.state_dict()) // 2):
                    print(f"Loaded adapter class={name} missing={len(missing)} unexpected={len(unexpected)}")
                    return model.to(device).eval()
            except Exception as exc:
                errors.append(f"{name} relaxed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not instantiate an Adapter from the checkpoint. "
        "Module classes=" + ", ".join(name for name, _ in module_classes) +
        "\n" + "\n".join(errors[-8:])
    )


def load_adapter_v2(path: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=False)
    config_data = dict(state["config"])
    is_v8 = str(state.get("format", "")).startswith("prompt_id_adapter_v8") or {
        "identity_gain_init", "learned_gate_floor"
    }.issubset(config_data)
    is_v7 = str(state.get("format", "")).startswith("prompt_id_adapter_v7") or {
        "structure_sigma", "structure_floor", "orthogonal_eps"
    }.issubset(config_data)
    is_v6 = str(state.get("format", "")).startswith("prompt_id_adapter_v6") or {
        "face_gate_pad", "face_gate_temperature"
    }.issubset(config_data)
    if is_v8:
        config = PromptIDAdapterV8Config(**config_data)
        model = PromptIdentityAdapterV8(config)
    elif is_v7:
        config = PromptIDAdapterV7Config(**config_data)
        model = PromptIdentityAdapterV7(config)
    elif is_v6:
        config = PromptIDAdapterV6Config(**config_data)
        model = PromptIdentityAdapterV6(config)
    else:
        config = PromptIDAdapterV2Config(**config_data)
        model = PromptIdentityAdapterV2(config)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def adapter_delta(
    adapter: Optional[torch.nn.Module],
    base_v: torch.Tensor,
    age_v: torch.Tensor,
    cache: Dict[str, Any],
    args: argparse.Namespace,
    source_age: float,
    target_age: float,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if adapter is None:
        return torch.zeros_like(age_v)
    source_age_t = torch.full(
        (base_v.shape[0],), float(source_age), device=base_v.device, dtype=base_v.dtype
    )
    target_age_t = torch.full(
        (base_v.shape[0],), float(target_age), device=base_v.device, dtype=base_v.dtype
    )
    timestep_t = timestep.to(device=base_v.device, dtype=base_v.dtype).reshape(-1)
    if timestep_t.numel() == 1:
        timestep_t = timestep_t.expand(base_v.shape[0])

    # PromptIdentityAdapterV1's actual interface is
    # (base_velocity, age_velocity, source_age, target_age, timestep).
    # Do not pass cache masks here: they belong to the training loss, not the
    # learned Adapter forward path.
    kwargs = {
        "base_velocity": base_v,
        "age_velocity": age_v,
        "source_age": source_age_t,
        "target_age": target_age_t,
        "timestep": timestep_t,
    }
    try:
        out = adapter(**kwargs)
    except TypeError:
        out = adapter(base_v, age_v, source_age_t, target_age_t, timestep_t)
    delta = unwrap_output(out).to(age_v.dtype)
    if delta.shape != age_v.shape:
        delta = torch.broadcast_to(delta, age_v.shape)
    ratio = delta.square().mean().sqrt() / age_v.square().mean().sqrt().clamp_min(1e-6)
    if float(ratio) > args.max_injection_ratio:
        delta = delta * (args.max_injection_ratio / float(ratio))
    return delta * args.adapter_scale


def adapter_delta_v2(
    adapter: torch.nn.Module,
    base_v: torch.Tensor,
    age_v: torch.Tensor,
    source_age: float,
    target_age: float,
    timestep: torch.Tensor,
    landmark_xy: torch.Tensor,
    identity_embedding: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    model_dtype = next(adapter.parameters()).dtype
    source_age_t = torch.full((base_v.shape[0],), source_age, device=base_v.device, dtype=model_dtype)
    target_age_t = torch.full((base_v.shape[0],), target_age, device=base_v.device, dtype=model_dtype)
    timestep_t = timestep.to(device=base_v.device, dtype=model_dtype).reshape(-1)
    if timestep_t.numel() == 1:
        timestep_t = timestep_t.expand(base_v.shape[0])
    landmark_xy = landmark_xy.to(device=base_v.device, dtype=model_dtype)
    identity_embedding = identity_embedding.to(device=base_v.device, dtype=model_dtype)
    if landmark_xy.shape[0] == 1 and base_v.shape[0] > 1:
        landmark_xy = landmark_xy.expand(base_v.shape[0], -1, -1)
    if identity_embedding.shape[0] == 1 and base_v.shape[0] > 1:
        identity_embedding = identity_embedding.expand(base_v.shape[0], -1)
    strength = (target_age_t - source_age_t).abs() / 30.0
    adapter_kwargs = {
        "base": base_v.to(dtype=model_dtype),
        "age": age_v.to(dtype=model_dtype),
        "source_age": source_age_t,
        "target_age": target_age_t,
        "timestep": timestep_t,
        "strength": strength,
        "landmark_xy": landmark_xy,
        "identity_embedding": identity_embedding,
    }
    if getattr(adapter, "requires_spatial_gate", False):
        if getattr(adapter, "requires_structure_gate", False):
            adapter_kwargs["spatial_gate"] = landmark_structure_gate(
                landmark_xy,
                base_v.shape[1],
                image_size=float(getattr(adapter.config, "landmark_image_size", 512.0)),
                sigma=float(getattr(adapter.config, "structure_sigma", 0.16)),
                floor=float(getattr(adapter.config, "structure_floor", 0.05)),
                face_pad=float(getattr(adapter.config, "face_gate_pad", 0.12)),
                face_temperature=float(getattr(adapter.config, "face_gate_temperature", 0.08)),
            ).to(device=base_v.device, dtype=model_dtype)
        else:
            adapter_kwargs["spatial_gate"] = landmark_face_gate(
                landmark_xy,
                base_v.shape[1],
                pad=float(getattr(adapter.config, "face_gate_pad", 0.12)),
                temperature=float(getattr(adapter.config, "face_gate_temperature", 0.08)),
                image_size=float(getattr(adapter.config, "landmark_image_size", 512.0)),
            ).to(device=base_v.device, dtype=model_dtype)
    delta = adapter(**adapter_kwargs).to(age_v.dtype)
    ratio = delta.square().mean().sqrt() / age_v.square().mean().sqrt().clamp_min(1e-6)
    if float(ratio) > args.max_injection_ratio:
        delta = delta * (args.max_injection_ratio / float(ratio))
    return delta * args.adapter_scale


def make_sheet(images: Iterable[Tuple[str, Image.Image]], path: Path) -> None:
    items = list(images)
    if not items:
        return
    w, h = items[0][1].size
    sheet = Image.new("RGB", (w * len(items), h + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(items):
        sheet.paste(image.convert("RGB"), (i * w, 28))
        draw.text((i * w + 6, 6), label, fill="black")
    sheet.save(path, quality=95)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cuda" and dtype == torch.float32:
        dtype = torch.bfloat16

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    record = load_first_record(cache, args.source_age, args.target_age)
    base_prompt, age_prompt = prompt_pair(args, record)
    source_age = args.source_age if args.source_age is not None else float(record.get("source_age", 35.0))
    condition_index = find_condition_index(cache, args.source_image, args.target_age)
    landmark_obj = torch.load(args.landmark_cache, map_location="cpu", weights_only=False)
    identity_obj = torch.load(args.identity_cache, map_location="cpu", weights_only=False)
    landmark_condition = landmark_obj["landmark_xy"][condition_index:condition_index + 1]
    identity_condition = identity_obj["identity_embedding"][condition_index:condition_index + 1]
    if not bool(landmark_obj.get("valid", torch.ones(1, dtype=torch.bool))[condition_index]):
        raise ValueError(f"no valid landmark condition at cache row {condition_index}")
    if not bool(identity_obj.get("valid", torch.ones(1, dtype=torch.bool))[condition_index]):
        raise ValueError(f"no valid identity condition at cache row {condition_index}")

    from diffusers import FluxImg2ImgPipeline
    pipe = FluxImg2ImgPipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    pipe.transformer.eval()
    pipe.vae.eval()

    base_cond = encode_prompt(pipe, base_prompt, device, dtype, args.max_sequence_length)
    age_cond = encode_prompt(pipe, age_prompt, device, dtype, args.max_sequence_length)
    adapter = load_adapter_v2(args.checkpoint, device)
    image = Image.open(args.source_image).convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    packed, latent_ids = pack_image_latents(pipe, image, device, dtype, generator)

    # Flow-Match schedulers keep a mutable step_index.  The two visualization
    # trajectories must therefore use independent scheduler instances; calling
    # step() twice on one instance advances it twice per denoising timestep.
    scheduler_no_adapter = copy.deepcopy(pipe.scheduler)
    scheduler_with_adapter = copy.deepcopy(pipe.scheduler)
    scheduler_no_adapter.set_timesteps(
        args.steps,
        device=device,
        **scheduler_shift_kwargs(scheduler_no_adapter, int(packed.shape[1])),
    )
    scheduler_with_adapter.set_timesteps(
        args.steps,
        device=device,
        **scheduler_shift_kwargs(scheduler_with_adapter, int(packed.shape[1])),
    )
    timesteps = scheduler_no_adapter.timesteps
    start = max(0, int(args.steps * (1.0 - args.strength)))
    timesteps = timesteps[start:]
    noise = torch.randn(packed.shape, generator=generator, device=device, dtype=dtype)
    if len(timesteps):
        # Use the scheduler's actual sigma.  With FLUX dynamic shifting,
        # timestep / 1000 is not generally the interpolation coefficient.
        sigma = scheduler_no_adapter.sigmas[start].to(device=device, dtype=dtype)
        packed = packed * (1.0 - sigma) + noise * sigma
    if len(timesteps) < 8:
        print(
            f"WARNING: only {len(timesteps)} denoising steps remain after "
            f"strength={args.strength}; use --steps 20 or higher for visual output."
        )

    no_adapter_img = packed.clone()
    with_adapter_img = packed.clone()
    age_rms = []
    age_delta_rms = []
    corr_rms = []
    for t in timesteps:
        # Diffusers' FLUX pipeline passes normalized timesteps to the
        # transformer, while scheduler.step() receives the original value.
        ts = (t / 1000.0).expand(packed.shape[0]).to(dtype)
        no_adapter_v = call_transformer(
            pipe.transformer,
            no_adapter_img,
            condition_kwargs(age_cond, latent_ids, args.guidance_scale),
            ts,
        )
        no_base_v = call_transformer(
            pipe.transformer,
            no_adapter_img,
            condition_kwargs(base_cond, latent_ids, args.guidance_scale),
            ts,
        )
        boosted_no_adapter_v = no_adapter_v + args.age_boost * (
            no_adapter_v - no_base_v
        )
        with_base_v = call_transformer(
            pipe.transformer,
            with_adapter_img,
            condition_kwargs(base_cond, latent_ids, args.guidance_scale),
            ts,
        )
        with_age_v = call_transformer(
            pipe.transformer,
            with_adapter_img,
            condition_kwargs(age_cond, latent_ids, args.guidance_scale),
            ts,
        )
        boosted_age_v = with_age_v + args.age_boost * (
            with_age_v - with_base_v
        )
        delta = adapter_delta_v2(
            adapter,
            with_base_v,
            boosted_age_v,
            source_age,
            args.target_age,
            ts,
            landmark_condition, identity_condition, args,
        )
        no_adapter_img = scheduler_no_adapter.step(
            boosted_no_adapter_v, t, no_adapter_img
        ).prev_sample
        with_adapter_img = scheduler_with_adapter.step(
            boosted_age_v + delta, t, with_adapter_img
        ).prev_sample
        age_rms.append(float(boosted_age_v.float().square().mean().sqrt()))
        age_delta_rms.append(
            float((boosted_age_v - with_base_v).float().square().mean().sqrt())
        )
        corr_rms.append(float(delta.float().square().mean().sqrt()))

    height = width = args.resolution
    no_adapter = unpack_and_decode(pipe, no_adapter_img, height, width)
    with_adapter = unpack_and_decode(pipe, with_adapter_img, height, width)
    source_path = out_dir / "source.png"
    no_adapter_path = out_dir / "aging_no_adapter.png"
    with_adapter_path = out_dir / "aging_with_id_adapter.png"
    source = image.convert("RGB")
    source.save(source_path)
    no_adapter.save(no_adapter_path)
    with_adapter.save(with_adapter_path)
    make_sheet([("source", source), ("aging", no_adapter), ("aging + ID", with_adapter)], out_dir / "contact_sheet.jpg")

    diagnostics = {
        "source_image": str(Path(args.source_image).resolve()),
        "source_age": source_age,
        "target_age": args.target_age,
        "age_boost": args.age_boost,
        "base_prompt": base_prompt,
        "age_prompt": age_prompt,
        "steps": int(len(timesteps)),
        "seed": args.seed,
        "age_velocity_rms_mean": float(np.mean(age_rms)) if age_rms else 0.0,
        "age_delta_rms_mean": float(np.mean(age_delta_rms)) if age_delta_rms else 0.0,
        "applied_correction_rms_mean": float(np.mean(corr_rms)) if corr_rms else 0.0,
        "correction_to_age_delta_ratio": (
            float(np.mean(corr_rms)) / max(float(np.mean(age_delta_rms)), 1e-8)
            if age_delta_rms else 0.0
        ),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(f"DONE output={out_dir}")


if __name__ == "__main__":
    main()
