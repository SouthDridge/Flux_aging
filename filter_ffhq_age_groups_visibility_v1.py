"""Second-stage filtering of an existing age-grouped FFHQ dataset.

The source dataset is never modified. The output keeps every input column,
adds audit fields, and can create symlinks/hardlinks/copies for accepted images.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm


OCCLUSION_COLUMNS = (
    "left_eye_occluded",
    "right_eye_occluded",
    "mouth_occluded",
    "forehead_occluded",
    "face_occluded",
)
SUNGLASS_LABELS = {"dark", "sunglasses", "sunglass", "sun glasses", "tinted", "tinted glasses"}
CLEAR_GLASSES_LABELS = {"normal", "reading", "clear", "eyeglasses", "glasses"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--occlusion_threshold", type=float, default=0.5)
    p.add_argument("--reject_clear_glasses", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reject_missing_eye_metadata", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--transfer_mode", choices=("symlink", "hardlink", "copy", "manifest_only"), default="symlink")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_label(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def occluded(value, threshold, percent_scale=False):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        label = normalize_label(value)
        if label in {"true", "yes", "y", "occluded"}:
            return True
        if label in {"false", "no", "n", "clear", "visible"}:
            return False
        try:
            value = float(label)
        except ValueError:
            return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric == -1:
        return None
    if numeric == 1:
        return True
    if percent_scale:
        numeric /= 100.0
    return numeric >= threshold


def rejection_reasons(row, args, available_occlusion_columns, percent_scale_columns=None):
    percent_scale_columns = set(percent_scale_columns or [])
    reasons = []
    glasses = normalize_label(row.get("glasses"))
    compact = glasses.replace(" ", "")
    if glasses in SUNGLASS_LABELS or "sunglass" in compact or glasses == "dark":
        reasons.append("sunglasses")
    elif args.reject_clear_glasses and glasses in CLEAR_GLASSES_LABELS:
        reasons.append("clear_glasses")

    for column in available_occlusion_columns:
        state = occluded(row.get(column), args.occlusion_threshold, column in percent_scale_columns)
        if state is True:
            reasons.append(column)
        elif state is None and args.reject_missing_eye_metadata and column in {"left_eye_occluded", "right_eye_occluded"}:
            reasons.append(f"missing_{column}")
    return reasons


def prepare_output(path, overwrite):
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output is not a directory: {path}")
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise ValueError(f"Output is nonempty: {path}; use a new path or --overwrite")
        if entries and overwrite:
            marker = path / "visibility_filter_summary.json"
            if not marker.is_file():
                raise ValueError("Refusing to overwrite a directory not created by this filter")
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    path.mkdir(parents=True, exist_ok=True)


def transfer(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = (args.manifest or input_dir / "age_group_selection.csv").resolve()
    if not input_dir.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"Missing input directory or manifest: {input_dir}, {manifest}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must not be the input directory or inside it")
    if not 0 <= args.occlusion_threshold <= 100:
        raise ValueError("--occlusion_threshold must be between 0 and 100")

    frame = pd.read_csv(manifest)
    required = {"file_name", "relative_path", "eval_age_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if frame.empty or frame["relative_path"].duplicated().any():
        raise ValueError("Manifest is empty or contains duplicate relative paths")
    available_occlusion = [name for name in OCCLUSION_COLUMNS if name in frame.columns]
    missing_eye_columns = sorted({"left_eye_occluded", "right_eye_occluded"} - set(frame.columns))
    if args.reject_missing_eye_metadata and missing_eye_columns:
        raise ValueError(f"Strict mode requires eye-occlusion columns: {missing_eye_columns}")
    if "glasses" not in frame.columns:
        raise ValueError("Manifest has no glasses column; cannot reliably exclude sunglasses")

    percent_scale_columns = []
    for column in available_occlusion:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric[numeric.notna() & numeric.ge(0)]
        if len(valid) and valid.max() > 1:
            percent_scale_columns.append(column)

    source_paths, path_ok, reasons = [], [], []
    for _, row in frame.iterrows():
        candidate = (input_dir / str(row["relative_path"])).resolve()
        if input_dir not in candidate.parents:
            raise ValueError(f"Path escapes input directory: {row['relative_path']}")
        exists = candidate.is_file()
        row_reasons = rejection_reasons(row, args, available_occlusion, percent_scale_columns)
        if not exists:
            row_reasons.append("missing_image")
        source_paths.append(str(candidate))
        path_ok.append(exists)
        reasons.append(";".join(row_reasons))
    audit = frame.copy()
    audit["visibility_source_path"] = source_paths
    audit["visibility_image_exists"] = path_ok
    audit["visibility_rejection_reasons"] = reasons
    audit["visibility_keep"] = audit["visibility_rejection_reasons"].eq("")
    accepted = audit[audit["visibility_keep"]].copy().reset_index(drop=True)
    rejected = audit[~audit["visibility_keep"]].copy().reset_index(drop=True)

    prepare_output(output_dir, args.overwrite)
    if args.transfer_mode != "manifest_only":
        for row in tqdm(accepted.itertuples(index=False), total=len(accepted), desc=f"Creating {args.transfer_mode}s"):
            transfer(Path(row.visibility_source_path), output_dir / row.relative_path, args.transfer_mode)
    accepted.to_csv(output_dir / "age_group_selection.csv", index=False)
    rejected.to_csv(output_dir / "visibility_rejected.csv", index=False)
    audit.to_csv(output_dir / "visibility_audit_all.csv", index=False)

    group_summary = (audit.groupby(["eval_age_group", "visibility_keep"], dropna=False).size()
                     .unstack(fill_value=0).rename(columns={False: "rejected", True: "accepted"}).reset_index())
    for column in ("accepted", "rejected"):
        if column not in group_summary:
            group_summary[column] = 0
    group_summary["total"] = group_summary["accepted"] + group_summary["rejected"]
    group_summary["acceptance_rate"] = group_summary["accepted"] / group_summary["total"]
    group_summary.to_csv(output_dir / "visibility_summary_by_age_group.csv", index=False)

    reason_counts = {}
    for value in rejected["visibility_rejection_reasons"]:
        for reason in str(value).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "version": "ffhq_age_groups_visibility_v1",
        "input_dir": str(input_dir), "input_manifest": str(manifest), "output_dir": str(output_dir),
        "input_rows": int(len(audit)), "accepted_rows": int(len(accepted)), "rejected_rows": int(len(rejected)),
        "acceptance_rate": float(len(accepted) / len(audit)), "transfer_mode": args.transfer_mode,
        "occlusion_threshold": args.occlusion_threshold, "reject_clear_glasses": args.reject_clear_glasses,
        "reject_missing_eye_metadata": args.reject_missing_eye_metadata,
        "available_occlusion_columns": available_occlusion, "reason_counts": reason_counts,
        "percent_scale_occlusion_columns": percent_scale_columns,
        "limitations": ["Metadata filtering cannot detect unlabelled hands, food, microphones or other occluders.",
                        "Normal/clear glasses are retained unless --reject_clear_glasses is used."],
    }
    (output_dir / "visibility_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(group_summary.to_string(index=False))
    print(f"filtered_manifest={output_dir / 'age_group_selection.csv'}")


if __name__ == "__main__":
    main()
