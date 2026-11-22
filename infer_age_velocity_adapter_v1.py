"""Image inference for the detail-conditioned velocity adapter."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

from train_age_velocity_adapter_v1 import AdaptiveAgeVelocityAdapter
from flux_inference_utils import (
    call_transformer,
    condition_kwargs,
    dtype_from_name,
    encode_prompt,
    make_sheet,
    pack_image_latents,
    scheduler_shift_kwargs,
    unpack_and_decode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source_image", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source_age", type=float, required=True)
    p.add_argument("--target_age", type=float, default=65.0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--adapter_timestep", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument(
        "--skip_if_complete",
        action="store_true",
        help="Skip inference when output_dir already contains diagnostics.json.",
    )
    return p.parse_args()


def load_adapter(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(state["config"])
    adapter = AdaptiveAgeVelocityAdapter(**config)
    adapter.load_state_dict(state["model"], strict=True)
    return adapter.to(device=device).eval()


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.skip_if_complete and (out / "diagnostics.json").is_file():
        print(f"SKIP complete output={out}")
        return
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cuda" and dtype == torch.float32:
        dtype = torch.bfloat16

    base_prompt = (
        f"a realistic close-up portrait photograph of a person at age {args.source_age:g}"
    )
    detail_prompt = (
        f"a realistic close-up portrait photograph of a person at age {args.target_age:g}, "
        "preserving recognizable identity, eye shape, nose shape, mouth shape, and face structure"
    )

    pipe = FluxImg2ImgPipeline.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    base_cond = encode_prompt(pipe, base_prompt, device, dtype, args.max_sequence_length)
    detail_cond = encode_prompt(pipe, detail_prompt, device, dtype, args.max_sequence_length)
    adapter = load_adapter(args.checkpoint, device)
    if not getattr(adapter, "use_detail_input", False):
        raise ValueError("This inference script requires a checkpoint trained with --use_detail_input")

    source = Image.open(args.source_image).convert("RGB").resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    packed, latent_ids = pack_image_latents(pipe, source, device, dtype, generator)

    schedulers = [copy.deepcopy(pipe.scheduler) for _ in range(3)]
    for scheduler in schedulers:
        scheduler.set_timesteps(
            args.steps, device=device,
            **scheduler_shift_kwargs(scheduler, int(packed.shape[1])),
        )
    timesteps = schedulers[0].timesteps
    start = max(0, int(args.steps * (1.0 - args.strength)))
    timesteps = timesteps[start:]
    noise = torch.randn(packed.shape, generator=generator, device=device, dtype=dtype)
    sigma = schedulers[0].sigmas[start].to(device=device, dtype=dtype)
    initial = packed * (1.0 - sigma) + noise * sigma
    base_img, detail_img, adapter_img = (initial.clone(), initial.clone(), initial.clone())

    base_out = []
    detail_out = []
    correction_out = []
    applied_correction_out = []
    prompt_delta_out = []
    for t in timesteps:
        ts = (t / 1000.0).expand(packed.shape[0]).to(dtype)
        adapter_ts = torch.full_like(ts, args.adapter_timestep)

        base_v = call_transformer(
            pipe.transformer, base_img,
            condition_kwargs(base_cond, latent_ids, args.guidance_scale), ts,
        )
        detail_v = call_transformer(
            pipe.transformer, detail_img,
            condition_kwargs(detail_cond, latent_ids, args.guidance_scale), ts,
        )
        adapter_base_v = call_transformer(
            pipe.transformer, adapter_img,
            condition_kwargs(base_cond, latent_ids, args.guidance_scale), ts,
        )
        adapter_detail_v = call_transformer(
            pipe.transformer, adapter_img,
            condition_kwargs(detail_cond, latent_ids, args.guidance_scale), ts,
        )
        source_age = torch.full((1,), args.source_age, device=device, dtype=torch.float32)
        target_age = torch.full((1,), args.target_age, device=device, dtype=torch.float32)
        predicted_delta = adapter(
            adapter_base_v, source_age, target_age, adapter_ts, adapter_detail_v
        ).to(adapter_base_v.dtype)
        adapter_v = adapter_base_v + args.adapter_scale * predicted_delta

        base_img = schedulers[0].step(base_v, t, base_img).prev_sample
        detail_img = schedulers[1].step(detail_v, t, detail_img).prev_sample
        adapter_img = schedulers[2].step(adapter_v, t, adapter_img).prev_sample
        base_out.append(float(base_v.float().square().mean().sqrt()))
        detail_out.append(float(detail_v.float().square().mean().sqrt()))
        correction_out.append(float(predicted_delta.float().square().mean().sqrt()))
        applied_correction_out.append(
            float((args.adapter_scale * predicted_delta).float().square().mean().sqrt())
        )
        prompt_delta_out.append(
            float((detail_v - base_v).float().square().mean().sqrt())
        )

    base_image = unpack_and_decode(pipe, base_img, args.resolution, args.resolution)
    detail_image = unpack_and_decode(pipe, detail_img, args.resolution, args.resolution)
    adapter_image = unpack_and_decode(pipe, adapter_img, args.resolution, args.resolution)
    source.save(out / "source.png")
    base_image.save(out / "aging_base.png")
    detail_image.save(out / "aging_detail.png")
    adapter_image.save(out / "aging_adapter.png")
    make_sheet(
        [("source", source), ("base aging", base_image),
         ("detail aging", detail_image), ("adapter", adapter_image)],
        out / "contact_sheet.jpg",
    )
    diagnostics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_image": str(Path(args.source_image).resolve()),
        "source_age": args.source_age,
        "target_age": args.target_age,
        "steps": len(timesteps),
        "total_scheduler_steps": args.steps,
        "strength": args.strength,
        "adapter_scale": args.adapter_scale,
        "adapter_timestep": args.adapter_timestep,
        "seed": args.seed,
        "base_velocity_rms_mean": sum(base_out) / max(len(base_out), 1),
        "detail_velocity_rms_mean": sum(detail_out) / max(len(detail_out), 1),
        "predicted_delta_rms_mean": sum(correction_out) / max(len(correction_out), 1),
        "applied_correction_rms_mean": sum(applied_correction_out) / max(len(applied_correction_out), 1),
        "prompt_detail_minus_base_rms_mean": sum(prompt_delta_out) / max(len(prompt_delta_out), 1),
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(f"DONE output={out}")


if __name__ == "__main__":
    main()
