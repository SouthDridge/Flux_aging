import argparse
import os
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample FFHQ images with age labels and valid InsightFace IDs.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--insightface_filter_csv",
        type=str,
        default=None,
        help="Reuse a previous InsightFace filter CSV and skip face detection.",
    )
    parser.add_argument("--require_insightface", action="store_true")
    parser.add_argument("--insightface_model", type=str, default="buffalo_l")
    parser.add_argument("--insightface_provider", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--insightface_det_size", type=int, default=640)
    parser.add_argument("--insightface_det_thresh", type=float, default=0.2)
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=None,
        help="Optional cap on candidate rows to scan before sampling.",
    )
    return parser.parse_args()


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("*", "", regex=False)
        .str.replace("\\", "", regex=False)
    )
    return df


def normalize_file_name(value):
    name = str(value).strip()
    if name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
        return name
    return f"{int(float(name)):05d}.png"


def load_valid_rows_from_filter(filter_csv, label_df):
    filter_df = clean_columns(pd.read_csv(filter_csv))
    if "file_name" not in filter_df.columns:
        raise ValueError(
            f"InsightFace filter CSV must contain file_name, got columns: {filter_df.columns.tolist()}"
        )

    filter_df["file_name"] = filter_df["file_name"].apply(normalize_file_name)
    if "valid_identity" in filter_df.columns:
        valid_mask = pd.to_numeric(
            filter_df["valid_identity"], errors="coerce"
        ).fillna(0).astype(int) > 0
    elif "num_faces" in filter_df.columns:
        valid_mask = pd.to_numeric(
            filter_df["num_faces"], errors="coerce"
        ).fillna(0) > 0
    else:
        raise ValueError(
            "InsightFace filter CSV must contain valid_identity or num_faces."
        )

    valid_names = set(filter_df.loc[valid_mask, "file_name"])
    valid_rows = label_df[label_df["file_name"].isin(valid_names)].copy()
    return valid_rows.reset_index(drop=True), filter_df


def build_face_app(args):
    if not args.require_insightface:
        return None
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "Install InsightFace and ONNXRuntime first: `pip install insightface onnxruntime-gpu`."
        ) from exc

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


def has_valid_face(face_app, image_path):
    if face_app is None:
        return True, 0, ""

    image = cv2.imread(str(image_path))
    if image is None:
        return False, 0, ""

    faces = face_app.get(image)
    if len(faces) == 0:
        return False, 0, ""

    selected = max(
        faces,
        key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
    )
    score = float(getattr(selected, "det_score", 0.0))
    return True, len(faces), score


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    dst_root = Path(args.dst_root)
    dst_image_dir = dst_root / "images"
    dst_csv = dst_root / "ffhq_aging_labels.csv"
    detection_csv = dst_root / "insightface_filter_results.csv"

    dst_image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df = clean_columns(df)
    print(f"CSV rows: {len(df)}")
    print(f"CSV columns: {df.columns.tolist()}")

    if "file_name" in df.columns:
        df["file_name"] = df["file_name"].apply(normalize_file_name)
    elif "image_number" in df.columns:
        df["file_name"] = df["image_number"].apply(normalize_file_name)
    else:
        raise ValueError(f"CSV must contain file_name or image_number, got columns: {df.columns.tolist()}")

    if args.insightface_filter_csv:
        df_valid, filter_df = load_valid_rows_from_filter(args.insightface_filter_csv, df)
        if args.scan_limit is not None:
            df_valid = df_valid.head(args.scan_limit).reset_index(drop=True)
        print(f"Reused InsightFace filter CSV: {args.insightface_filter_csv}")
        print(f"Filter CSV rows: {len(filter_df)}")
        print(f"Valid rows with age labels: {len(df_valid)}")
    else:
        local_images = {
            name
            for name in os.listdir(image_dir)
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        }
        print(f"Local images: {len(local_images)}")

        df_available = df[df["file_name"].isin(local_images)].reset_index(drop=True)
        if args.scan_limit is not None:
            df_available = df_available.head(args.scan_limit).reset_index(drop=True)
        print(f"Rows with both label and image: {len(df_available)}")

        face_app = build_face_app(args)
        valid_rows = []
        detection_rows = []
        read_or_detect_fail = 0
        multi_face = 0

        iterator = tqdm(df_available.iterrows(), total=len(df_available), desc="Filtering InsightFace")
        for _, row in iterator:
            image_path = image_dir / row["file_name"]
            valid, num_faces, score = has_valid_face(face_app, image_path)
            if valid:
                valid_rows.append(row)
            else:
                read_or_detect_fail += 1
            if num_faces > 1:
                multi_face += 1
            detection_rows.append(
                {
                    "file_name": row["file_name"],
                    "valid_identity": int(valid),
                    "num_faces": num_faces,
                    "selected_score": score,
                }
            )

        pd.DataFrame(detection_rows).to_csv(detection_csv, index=False)
        df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"InsightFace valid rows: {len(df_valid)}")
        print(f"Invalid/read-fail rows: {read_or_detect_fail}")
        print(f"Multi-face rows: {multi_face}")

    if len(df_valid) < args.num_samples:
        raise ValueError(
            f"Valid samples ({len(df_valid)}) are fewer than requested num_samples ({args.num_samples}). "
            f"Try lower threshold, larger source set, or smaller num_samples."
        )

    df_sampled = df_valid.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled rows: {len(df_sampled)}")

    success_count = 0
    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Copying images"):
        img_name = row["file_name"]
        source_path = image_dir / img_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Image listed in CSV does not exist: {source_path}")
        shutil.copy2(source_path, dst_image_dir / img_name)
        success_count += 1

    cols = ["file_name"] + [col for col in df_sampled.columns if col != "file_name"]
    df_sampled = df_sampled[cols]
    df_sampled.to_csv(dst_csv, index=False)

    print("\n" + "=" * 40)
    print(f"Copied images: {success_count}")
    print(f"CSV rows: {len(df_sampled)}")
    print(f"Output image dir: {dst_image_dir}")
    print(f"Output CSV: {dst_csv}")
    if args.insightface_filter_csv:
        print(f"Reused detection filter CSV: {args.insightface_filter_csv}")
    else:
        print(f"Detection filter CSV: {detection_csv}")
    print("=" * 40)


if __name__ == "__main__":
    main()
