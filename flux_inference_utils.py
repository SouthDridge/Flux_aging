"""Small FLUX image-to-image utilities used by the paper's v1 inference."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from PIL import Image, ImageDraw


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    return value


def unwrap_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported transformer output type: {type(output)!r}")


def call_transformer(
    transformer: torch.nn.Module,
    hidden_states: torch.Tensor,
    conditioning: Dict[str, Any],
    timestep: torch.Tensor,
) -> torch.Tensor:
    kwargs = dict(conditioning)
    kwargs["hidden_states"] = hidden_states
    kwargs["timestep"] = timestep
    try:
        return unwrap_output(transformer(**kwargs))
    except TypeError as exc:
        if "guidance" in kwargs:
            kwargs.pop("guidance")
            return unwrap_output(transformer(**kwargs))
        raise exc


def encode_prompt(
    pipe: Any,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> Dict[str, Any]:
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
    output = {name: value for name, value in zip(names, result)}
    output = {
        key: to_device(value, device, dtype)
        for key, value in output.items()
        if value is not None
    }

    prompt_embeds = output["prompt_embeds"]
    text_length = int(prompt_embeds.shape[1])
    output["text_ids"] = torch.zeros(
        (text_length, 3), device=device, dtype=prompt_embeds.dtype
    )
    return output


def condition_kwargs(
    encoded: Dict[str, Any], latent_ids: torch.Tensor, guidance: float
) -> Dict[str, Any]:
    output = {
        "encoder_hidden_states": encoded["prompt_embeds"],
        "pooled_projections": encoded["pooled_prompt_embeds"],
        "txt_ids": encoded["text_ids"].to(latent_ids.device),
        "img_ids": latent_ids,
        "guidance": torch.tensor(
            [guidance],
            device=latent_ids.device,
            dtype=encoded["prompt_embeds"].dtype,
        ),
    }
    return output


def pack_image_latents(
    pipe: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    if hasattr(pipe, "_encode_vae_image"):
        latents = pipe._encode_vae_image(image_tensor, generator=generator)
    else:
        latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    if latents.shape[1] == 16 and hasattr(pipe, "_pack_latents"):
        packed = pipe._pack_latents(
            latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]
        )
    else:
        packed = latents

    token_count = int(packed.shape[1])
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise RuntimeError(f"Expected a square packed FLUX latent grid, got {token_count} tokens")

    latent_ids = torch.zeros((token_count, 3), device=device, dtype=dtype)
    latent_ids[:, 1] = torch.arange(side, device=device, dtype=dtype).repeat_interleave(side)
    latent_ids[:, 2] = torch.arange(side, device=device, dtype=dtype).repeat(side)
    return packed.to(dtype), latent_ids


def unpack_and_decode(pipe: Any, packed: torch.Tensor, height: int, width: int) -> Image.Image:
    if packed.ndim == 3 and hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    else:
        latents = packed
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def scheduler_shift_kwargs(scheduler: Any, image_sequence_length: int) -> Dict[str, float]:
    config = scheduler.config
    if not bool(getattr(config, "use_dynamic_shifting", False)):
        return {}
    base_len = float(getattr(config, "base_image_seq_len", 256))
    max_len = float(getattr(config, "max_image_seq_len", 4096))
    base_shift = float(getattr(config, "base_shift", 0.5))
    max_shift = float(getattr(config, "max_shift", 1.15))
    if max_len <= base_len:
        raise ValueError("Invalid FLUX scheduler dynamic-shifting configuration")
    slope = (max_shift - base_shift) / (max_len - base_len)
    mu = slope * float(image_sequence_length) + base_shift - slope * base_len
    return {"mu": mu}


def make_sheet(images: Iterable[Tuple[str, Image.Image]], path: Path) -> None:
    items = list(images)
    if not items:
        return
    width, height = items[0][1].size
    sheet = Image.new("RGB", (width * len(items), height + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        sheet.paste(image.convert("RGB"), (index * width, 28))
        draw.text((index * width + 6, 6), label, fill="black")
    sheet.save(path, quality=95)
