import argparse
import math
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AGE_GROUPS = (
    ("20-29", 20, 29, 25.0),
    ("30-39", 30, 39, 35.0),
    ("40-49", 40, 49, 45.0),
    ("50-69", 50, 69, 60.0),
    ("70-80", 70, 80, 75.0),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select age-grouped FFHQ images from a previously generated "
            "InsightFace-valid CSV without running face detection again."
        )
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--age_csv", type=str, required=True)
    parser.add_argument("--insightface_filter_csv", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling_mode",
        choices=["balanced", "random"],
        default="balanced",
        help=(
            "balanced starts with an equal quota per age group and redistributes "
            "shortages; random follows the available FFHQ distribution."
        ),
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="hardlink avoids duplicating image data when source and destination share a filesystem.",
    )
    parser.add_argument(
        "--age_groups",
        type=str,
        default=None,
        help=(
            "Comma-separated age groups to sample, for example "
            "20-29,30-39,40-49. Defaults to all configured groups."
        ),
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
    if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
        return Path(name).name
    return f"{int(float(name)):05d}.png"


def add_file_name(df, csv_name):
    df = df.copy()
    if "file_name" in df.columns:
        source = "file_name"
    elif "image_number" in df.columns:
        source = "image_number"
    else:
        raise ValueError(
            f"{csv_name} must contain file_name or image_number; "
            f"got columns: {df.columns.tolist()}"
        )
    df["file_name"] = df[source].apply(normalize_file_name)
    return df


def insightface_valid_mask(df):
    numeric_columns = (
        "valid_identity",
        "identity_valid",
        "insightface_valid",
        "is_valid",
        "valid",
    )
    for column in numeric_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0) > 0

    if "num_faces" in df.columns:
        return pd.to_numeric(df["num_faces"], errors="coerce").fillna(0) > 0

    if "status" in df.columns:
        valid_statuses = {"valid", "ok", "success", "detected", "true", "1"}
        return df["status"].astype(str).str.strip().str.lower().isin(valid_statuses)

    print(
        "WARNING: InsightFace CSV has no validity column. "
        "All rows in that CSV will be treated as previously validated."
    )
    return pd.Series(True, index=df.index)


def normalize_age_group(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    text = text.replace(" ", "").replace("_", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    aliases = {
        "20-29": "20-29",
        "30-39": "30-39",
        "40-49": "40-49",
        "50-69": "50-69",
        "70-120": "70-80",
        "70+": "70-80",
        "70plus": "70-80",
        "70-80": "70-80",
    }
    if text in aliases:
        return aliases[text]

    try:
        age = float(text)
    except ValueError:
        return None

    if 20 <= age < 30:
        return "20-29"
    if 30 <= age < 40:
        return "30-39"
    if 40 <= age < 50:
        return "40-49"
    if 50 <= age < 70:
        return "50-69"
    if age >= 70:
        return "70-80"
    return None


def select_age_groups(value):
    if value is None:
        return AGE_GROUPS

    requested = [normalize_age_group(item) for item in value.split(",")]
    if not requested or any(group is None for group in requested):
        valid = ", ".join(group[0] for group in AGE_GROUPS)
        raise ValueError(f"--age_groups must contain only: {valid}")

    requested_set = set(requested)
    if len(requested_set) != len(requested):
        raise ValueError("--age_groups must not contain duplicate groups")

    return tuple(group for group in AGE_GROUPS if group[0] in requested_set)


def add_eval_age_columns(df):
    if "age_group" in df.columns:
        raw_age = df["age_group"]
    elif "age" in df.columns:
        raw_age = df["age"]
    elif "label_age" in df.columns:
        raw_age = df["label_age"]
    else:
        raise ValueError(
            "Age CSV must contain age_group, age, or label_age; "
            f"got columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["source_age_group"] = raw_age.astype(str)
    df["eval_age_group"] = raw_age.apply(normalize_age_group)

    bounds = {
        name: (low, high, midpoint)
        for name, low, high, midpoint in AGE_GROUPS
    }
    df["eval_age_low"] = df["eval_age_group"].map(
        lambda group: bounds[group][0] if group in bounds else None
    )
    df["eval_age_high"] = df["eval_age_group"].map(
        lambda group: bounds[group][1] if group in bounds else None
    )
    df["eval_age_midpoint"] = df["eval_age_group"].map(
        lambda group: bounds[group][2] if group in bounds else None
    )
    return df


def balanced_allocations(group_counts, total, age_groups):
    group_names = [name for name, _, _, _ in age_groups]
    base, remainder = divmod(total, len(group_names))
    targets = {
        group: base + int(index < remainder)
        for index, group in enumerate(group_names)
    }
    allocations = {
        group: min(targets[group], int(group_counts.get(group, 0)))
        for group in group_names
    }

    remaining = total - sum(allocations.values())
    while remaining > 0:
        available = [
            group
            for group in group_names
            if allocations[group] < int(group_counts.get(group, 0))
        ]
        if not available:
            break

        share = max(1, math.ceil(remaining / len(available)))
        for group in available:
            capacity = int(group_counts[group]) - allocations[group]
            take = min(share, capacity, remaining)
            allocations[group] += take
            remaining -= take
            if remaining == 0:
                break

    if remaining:
        raise ValueError(
            f"Only {total - remaining} eligible images are available, "
            f"but {total} were requested."
        )
    return allocations


def sample_rows(df, num_samples, sampling_mode, seed, age_groups):
    if len(df) < num_samples:
        raise ValueError(
            f"Eligible images ({len(df)}) are fewer than requested samples ({num_samples})."
        )

    if sampling_mode == "random":
        sampled = df.sample(n=num_samples, random_state=seed).copy()
        allocations = sampled["eval_age_group"].value_counts().to_dict()
        return sampled.reset_index(drop=True), allocations

    counts = df["eval_age_group"].value_counts().to_dict()
    allocations = balanced_allocations(counts, num_samples, age_groups)
    sampled_parts = []
    for index, (group, _, _, _) in enumerate(age_groups):
        count = allocations[group]
        if count == 0:
            continue
        group_df = df[df["eval_age_group"] == group]
        sampled_parts.append(
            group_df.sample(n=count, random_state=seed + index).copy()
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled, allocations


def ensure_new_output_root(dst_root):
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {dst_root}. "
            "Use a new directory to avoid mixing selections from different runs."
        )
    dst_root.mkdir(parents=True, exist_ok=True)


def transfer_file(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def ordered_columns(df):
    preferred = [
        "file_name",
        "relative_path",
        "source_age_group",
        "eval_age_group",
        "eval_age_low",
        "eval_age_high",
        "eval_age_midpoint",
    ]
    return preferred + [column for column in df.columns if column not in preferred]


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    dst_root = Path(args.dst_root)
    selected_age_groups = select_age_groups(args.age_groups)

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    age_df = add_file_name(
        clean_columns(pd.read_csv(args.age_csv)),
        "Age CSV",
    )
    age_df = add_eval_age_columns(age_df)

    filter_df = add_file_name(
        clean_columns(pd.read_csv(args.insightface_filter_csv)),
        "InsightFace CSV",
    )
    valid_mask = insightface_valid_mask(filter_df)
    valid_names = set(filter_df.loc[valid_mask, "file_name"])

    local_images = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    eligible = age_df[
        age_df["file_name"].isin(valid_names)
        & age_df["file_name"].isin(local_images)
        & age_df["eval_age_group"].notna()
        & age_df["eval_age_group"].isin(
            [group[0] for group in selected_age_groups]
        )
    ].copy()
    eligible = eligible.drop_duplicates(subset=["file_name"]).reset_index(drop=True)

    group_names = [name for name, _, _, _ in selected_age_groups]
    available_counts = (
        eligible["eval_age_group"]
        .value_counts()
        .reindex(group_names, fill_value=0)
    )

    print(f"Age CSV rows: {len(age_df)}")
    print(f"InsightFace CSV rows: {len(filter_df)}")
    print(f"InsightFace-valid names: {len(valid_names)}")
    print(f"Local images: {len(local_images)}")
    print(f"Selected age groups: {', '.join(group_names)}")
    print(f"Eligible ID-valid images in selected groups: {len(eligible)}")
    print("Available by age group:")
    for group, count in available_counts.items():
        print(f"  {group}: {count}")

    sampled, allocations = sample_rows(
        eligible,
        num_samples=args.num_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        age_groups=selected_age_groups,
    )
    sampled["relative_path"] = sampled.apply(
        lambda row: f"{row['eval_age_group']}/{row['file_name']}", axis=1
    )

    ensure_new_output_root(dst_root)

    for row in tqdm(
        sampled.itertuples(index=False),
        total=len(sampled),
        desc=f"Transferring images ({args.copy_mode})",
    ):
        source = image_dir / row.file_name
        destination = dst_root / row.relative_path
        transfer_file(source, destination, args.copy_mode)

    sampled = sampled[ordered_columns(sampled)]
    combined_csv = dst_root / "age_group_selection.csv"
    sampled.to_csv(combined_csv, index=False)

    summary_rows = []
    for group, low, high, midpoint in selected_age_groups:
        group_df = sampled[sampled["eval_age_group"] == group].copy()
        group_csv = dst_root / group / "labels.csv"
        group_df.to_csv(group_csv, index=False)
        summary_rows.append(
            {
                "age_group": group,
                "age_low": low,
                "age_high": high,
                "age_midpoint": midpoint,
                "available_count": int(available_counts[group]),
                "selected_count": int(len(group_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = dst_root / "selection_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 52)
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"Transfer mode: {args.copy_mode}")
    print(f"Selected images: {len(sampled)}")
    for group, _, _, _ in selected_age_groups:
        print(f"  {group}: {allocations.get(group, 0)}")
    print(f"Output root: {dst_root}")
    print(f"Combined CSV: {combined_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("=" * 52)


if __name__ == "__main__":
    main()


