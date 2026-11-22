#!/usr/bin/env python3
"""Compute FID for two image directories using a fixed, reproducible subset."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "cleanfid", "torchmetrics"),
        default="auto",
    )
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def choose_paths(paths: list[Path], max_images: int, seed: int) -> list[Path]:
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    return sorted(random.Random(seed).sample(paths, max_images))


def select_equal_subsets(
    real_dir: Path, fake_dir: Path, max_images: int, seed: int
) -> tuple[list[Path], list[Path]]:
    real = image_files(real_dir)
    fake = image_files(fake_dir)
    if not real:
        raise RuntimeError(f"no images found in real_dir: {real_dir}")
    if not fake:
        raise RuntimeError(f"no images found in fake_dir: {fake_dir}")

    count = min(len(real), len(fake))
    if max_images > 0:
        count = min(count, max_images)
    return choose_paths(real, count, seed), choose_paths(fake, count, seed + 1)


def link_subset(paths: list[Path], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        target = root / f"{index:06d}{path.suffix.lower()}"
        target.symlink_to(path.resolve())


def compute_cleanfid(real_paths: list[Path], fake_paths: list[Path]) -> float:
    from cleanfid import fid

    with tempfile.TemporaryDirectory(prefix="fid_subset_") as tmp:
        root = Path(tmp)
        real_subset = root / "real"
        fake_subset = root / "fake"
        link_subset(real_paths, real_subset)
        link_subset(fake_paths, fake_subset)
        return float(fid.compute_fid(str(real_subset), str(fake_subset)))


def compute_torchmetrics(
    real_paths: list[Path],
    fake_paths: list[Path],
    batch_size: int,
    device: str,
) -> float:
    import torch
    from PIL import Image, ImageOps
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    transform = transforms.Compose(
        [transforms.Resize((299, 299), antialias=True), transforms.PILToTensor()]
    )

    def update(paths: list[Path], real: bool) -> None:
        batch = []
        for path in paths:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            batch.append(transform(image))
            if len(batch) == batch_size:
                metric.update(torch.stack(batch).to(device), real=real)
                batch.clear()
        if batch:
            metric.update(torch.stack(batch).to(device), real=real)

    with torch.inference_mode():
        update(real_paths, real=True)
        update(fake_paths, real=False)
        return float(metric.compute().detach().cpu())


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real_dir).expanduser().resolve()
    fake_dir = Path(args.fake_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise SystemExit(f"real_dir is not a directory: {real_dir}")
    if not fake_dir.is_dir():
        raise SystemExit(f"fake_dir is not a directory: {fake_dir}")

    real_paths, fake_paths = select_equal_subsets(
        real_dir, fake_dir, args.max_images, args.seed
    )

    backends = (
        ["cleanfid", "torchmetrics"]
        if args.backend == "auto"
        else [args.backend]
    )
    errors = []
    value = None
    used_backend = None
    for backend in backends:
        try:
            if backend == "cleanfid":
                value = compute_cleanfid(real_paths, fake_paths)
            else:
                value = compute_torchmetrics(
                    real_paths, fake_paths, args.batch_size, args.device
                )
            used_backend = backend
            break
        except Exception as exc:  # Try the next optional backend in auto mode.
            errors.append(f"{backend}: {exc}")

    output = {
        "fid": value,
        "backend": used_backend,
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "real_images": len(real_paths),
        "fake_images": len(fake_paths),
        "max_images": args.max_images,
        "seed": args.seed,
        "errors": errors,
        "real_files": [str(path) for path in real_paths],
        "fake_files": [str(path) for path in fake_paths],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if value is None:
        raise SystemExit("FID computation failed; see errors in the output JSON")


if __name__ == "__main__":
    main()
