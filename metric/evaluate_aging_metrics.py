"""Evaluate aging edit outputs with age error, identity similarity, and FID."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
GENERATED_COLUMNS = (
    "generated_image",
    "generated_path",
    "edited_image",
    "edited_path",
    "adapter_image",
    "adapter_path",
    "result_path",
    "image_path",
    "output_path",
)
SOURCE_COLUMNS = (
    "source_image",
    "source_path",
    "input_image",
    "input_path",
)
TARGET_COLUMNS = (
    "target_age",
    "requested_target_age",
    "age_target",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute first-pass aging baseline metrics: Age MSE/MAE with an "
            "external age estimator, ArcFace identity cosine, and optional FID."
        )
    )
    parser.add_argument("--manifest_csv", default=None)
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--generated_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_age", type=float, default=None)
    parser.add_argument("--real_dir", default=None, help="Reference real image directory for FID.")
    parser.add_argument("--fake_dir", default=None, help="Generated image directory for FID; defaults to --generated_dir.")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--insightface_model", default="buffalo_l")
    parser.add_argument("--insightface_provider", choices=["cpu", "cuda"], default="cpu")
    # Match the detector settings used by the existing paired identity evals.
    parser.add_argument("--insightface_det_size", type=int, default=1536)
    parser.add_argument("--insightface_det_thresh", type=float, default=0.05)
    parser.add_argument(
        "--fid_backend",
        choices=["auto", "cleanfid", "torchmetrics", "none"],
        default="auto",
    )
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def first_existing(row: pd.Series, columns: Iterable[str]):
    for column in columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return row[column]
    return None


def resolve_path(value, base_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None:
        return None
    raw = str(value).strip()
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path])
        if base_dir is not None:
            candidates.extend([base_dir / path, base_dir / path.name])
    elif base_dir is not None:
        candidates.append(base_dir / path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def image_files(directory: Path) -> List[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def load_manifest(args) -> List[Dict[str, object]]:
    if args.manifest_csv:
        dataframe = pd.read_csv(args.manifest_csv)
        rows = []
        source_base = Path(args.source_dir) if args.source_dir else None
        generated_base = Path(args.generated_dir) if args.generated_dir else None
        for _, row in dataframe.iterrows():
            generated_path = resolve_path(
                first_existing(row, GENERATED_COLUMNS), generated_base
            )
            if generated_path is None or not generated_path.is_file():
                continue
            source_path = resolve_path(first_existing(row, SOURCE_COLUMNS), source_base)
            target_age = first_existing(row, TARGET_COLUMNS)
            image_id = first_existing(row, ("image_id", "sample_id", "id"))
            rows.append(
                {
                    "image_id": str(image_id) if image_id is not None else generated_path.stem,
                    "source_image": source_path if source_path and source_path.is_file() else None,
                    "generated_image": generated_path,
                    "target_age": float(target_age)
                    if target_age is not None and str(target_age).strip()
                    else args.target_age,
                }
            )
        if not rows:
            examples = []
            for _, row in dataframe.head(5).iterrows():
                value = first_existing(row, GENERATED_COLUMNS)
                examples.append(str(value) if value is not None else "<missing generated_image>")
            raise ValueError(
                "No valid generated images were found from --manifest_csv. "
                f"Manifest rows={len(dataframe)}; examples={examples}"
            )
        return rows[: args.max_images or None]

    if not args.generated_dir:
        raise ValueError("Provide either --manifest_csv or --generated_dir.")
    generated_dir = Path(args.generated_dir)
    generated_paths = image_files(generated_dir)
    source_map = {}
    if args.source_dir:
        source_map = {path.name: path for path in image_files(Path(args.source_dir))}
    rows = [
        {
            "image_id": path.stem,
            "source_image": source_map.get(path.name),
            "generated_image": path,
            "target_age": args.target_age,
        }
        for path in generated_paths
    ]
    return rows[: args.max_images or None]


def load_rgb_uint8(path: Path, resize: Optional[int]):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if resize:
        image = image.resize((resize, resize), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1].copy()


def build_face_app(args):
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if args.insightface_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    app = FaceAnalysis(name=args.insightface_model, providers=providers)
    app.prepare(
        ctx_id=ctx_id,
        det_size=(args.insightface_det_size, args.insightface_det_size),
        det_thresh=args.insightface_det_thresh,
    )
    return app


def extract_face(app, rgb_image: np.ndarray):
    faces = app.get(rgb_to_bgr(rgb_image))
    if not faces:
        return {
            "valid": 0,
            "num_faces": 0,
            "det_score": np.nan,
            "age": np.nan,
            "gender": np.nan,
            "embedding": None,
        }
    face = max(
        faces,
        key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
    )
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return {
        "valid": 1,
        "num_faces": len(faces),
        "det_score": float(getattr(face, "det_score", np.nan)),
        "age": float(getattr(face, "age", np.nan)),
        "gender": float(getattr(face, "gender", np.nan)),
        "embedding": embedding,
    }


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def summarize_rows(rows: List[Dict[str, object]]):
    numeric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"image_id", "source_image", "generated_image"}
    ]
    summary = {"samples": len(rows)}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if np.isfinite(values).any():
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def compute_fid_cleanfid(real_dir: Path, fake_dir: Path):
    from cleanfid import fid

    return float(fid.compute_fid(str(real_dir), str(fake_dir)))


def compute_fid_torchmetrics(real_dir: Path, fake_dir: Path, batch_size: int, device: str):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299), antialias=True),
            transforms.PILToTensor(),
        ]
    )

    def update(paths, real: bool):
        batch = []
        for path in paths:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            batch.append(transform(image))
            if len(batch) == batch_size:
                metric.update(torch.stack(batch).to(device), real=real)
                batch = []
        if batch:
            metric.update(torch.stack(batch).to(device), real=real)

    update(image_files(real_dir), real=True)
    update(image_files(fake_dir), real=False)
    return float(metric.compute().detach().cpu())


def compute_fid(args, summary):
    if args.fid_backend == "none":
        return
    if not args.real_dir:
        return
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir or args.generated_dir or "")
    if not real_dir.is_dir() or not fake_dir.is_dir():
        summary["fid_error"] = "real_dir or fake_dir is not a directory"
        return

    backends = (
        ["cleanfid", "torchmetrics"]
        if args.fid_backend == "auto"
        else [args.fid_backend]
    )
    errors = []
    for backend in backends:
        try:
            if backend == "cleanfid":
                summary["fid"] = compute_fid_cleanfid(real_dir, fake_dir)
                summary["fid_backend"] = "cleanfid"
                return
            if backend == "torchmetrics":
                summary["fid"] = compute_fid_torchmetrics(
                    real_dir, fake_dir, args.fid_batch_size, args.device
                )
                summary["fid_backend"] = "torchmetrics"
                return
        except Exception as exc:  # FID deps are optional and environment-dependent.
            errors.append(f"{backend}: {exc}")
    summary["fid_error"] = " | ".join(errors)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args)
    face_app = build_face_app(args)

    rows = []
    for item in tqdm(items, desc="Evaluating aging metrics"):
        generated = load_rgb_uint8(Path(item["generated_image"]), args.resize)
        generated_face = extract_face(face_app, generated)
        source_face = None
        if item["source_image"]:
            source = load_rgb_uint8(Path(item["source_image"]), args.resize)
            source_face = extract_face(face_app, source)
        target_age = item["target_age"]
        predicted_age = generated_face["age"]
        age_error = (
            float(predicted_age - target_age)
            if target_age is not None and np.isfinite(predicted_age)
            else np.nan
        )
        row = {
            "image_id": item["image_id"],
            "source_image": str(item["source_image"]) if item["source_image"] else "",
            "generated_image": str(item["generated_image"]),
            "target_age": target_age if target_age is not None else np.nan,
            "predicted_age": predicted_age,
            "age_error": age_error,
            "age_abs_error": abs(age_error) if np.isfinite(age_error) else np.nan,
            "age_squared_error": age_error * age_error if np.isfinite(age_error) else np.nan,
            "generated_face_valid": generated_face["valid"],
            "generated_num_faces": generated_face["num_faces"],
            "generated_det_score": generated_face["det_score"],
            "source_face_valid": source_face["valid"] if source_face else np.nan,
            "source_num_faces": source_face["num_faces"] if source_face else np.nan,
            "source_det_score": source_face["det_score"] if source_face else np.nan,
            "id_cosine": cosine(
                source_face["embedding"] if source_face else None,
                generated_face["embedding"],
            ),
        }
        rows.append(row)

    metrics_csv = output_dir / "aging_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    summary["age_mse"] = summary.get("age_squared_error_mean", float("nan"))
    summary["age_mae"] = summary.get("age_abs_error_mean", float("nan"))
    summary["id_cosine"] = summary.get("id_cosine_mean", float("nan"))
    compute_fid(args, summary)

    summary_json = output_dir / "aging_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"metrics csv: {metrics_csv}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
