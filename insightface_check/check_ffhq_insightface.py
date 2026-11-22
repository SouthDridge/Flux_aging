import argparse
import csv
import os
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Check InsightFace detection coverage on an image folder.")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing face images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for CSV and summary output.",
    )
    parser.add_argument("--model", type=str, default="buffalo_l")
    parser.add_argument("--provider", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to check.",
    )
    return parser.parse_args()


def build_face_app(args):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "Install InsightFace and ONNXRuntime first: `pip install insightface onnxruntime-gpu`."
        ) from exc

    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    app = FaceAnalysis(name=args.model, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    return app


def list_images(image_dir, limit=None):
    image_dir = Path(image_dir)
    paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        paths = paths[:limit]
    return paths


def main():
    args = parse_args()
    image_paths = list_images(args.image_dir, args.limit)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = Path(args.output_dir) / "insightface_detection_results.csv"
    summary_path = Path(args.output_dir) / "summary.txt"

    app = build_face_app(args)

    rows = []
    success = 0
    multi_face = 0
    read_fail = 0

    for path in tqdm(image_paths, desc="Checking InsightFace"):
        image = cv2.imread(str(path))
        if image is None:
            read_fail += 1
            rows.append(
                {
                    "file_name": path.name,
                    "read_ok": 0,
                    "num_faces": 0,
                    "selected_score": "",
                    "selected_bbox": "",
                    "embedding_norm": "",
                }
            )
            continue

        faces = app.get(image)
        num_faces = len(faces)
        if num_faces > 0:
            success += 1
        if num_faces > 1:
            multi_face += 1

        if num_faces == 0:
            selected_score = ""
            selected_bbox = ""
            embedding_norm = ""
        else:
            selected = max(
                faces,
                key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            )
            selected_score = float(getattr(selected, "det_score", 0.0))
            selected_bbox = ",".join(f"{float(value):.2f}" for value in selected.bbox)
            embedding = getattr(selected, "normed_embedding", None)
            embedding_norm = "" if embedding is None else float((embedding**2).sum() ** 0.5)

        rows.append(
            {
                "file_name": path.name,
                "read_ok": 1,
                "num_faces": num_faces,
                "selected_score": selected_score,
                "selected_bbox": selected_bbox,
                "embedding_norm": embedding_norm,
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "read_ok",
                "num_faces",
                "selected_score",
                "selected_bbox",
                "embedding_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(image_paths)
    success_rate = success / total
    summary = [
        f"image_dir: {args.image_dir}",
        f"model: {args.model}",
        f"provider: {args.provider}",
        f"det_size: {args.det_size}",
        f"det_thresh: {args.det_thresh}",
        f"total_images: {total}",
        f"read_fail: {read_fail}",
        f"detected_images: {success}",
        f"detection_rate: {success_rate:.4f}",
        f"multi_face_images: {multi_face}",
        f"csv_path: {csv_path}",
    ]
    summary_text = "\n".join(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(summary_text)


if __name__ == "__main__":
    main()
