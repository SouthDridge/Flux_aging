"""Create one source-image row per source_id from a teacher CSV manifest."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def first(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def age_value(row):
    value = first(row, ("source_age", "source_age_midpoint", "eval_age_midpoint"), "")
    if value:
        return value
    group = first(row, ("source_age_group", "eval_age_group", "age_group"), "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", str(group))
    if match:
        return str((float(match.group(1)) + float(match.group(2))) / 2.0)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        print("input_columns:", reader.fieldnames)
        for row in reader:
            image = first(row, ("source_image", "source_path", "input_image"), "")
            source_id = first(row, ("source_id", "source", "image_number"), "")
            if not source_id:
                source_id = Path(image).stem
            if not image or source_id in seen:
                continue
            seen.add(source_id)
            rows.append({
                "source_id": source_id,
                "source_image": image,
                "source_age": age_value(row),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "source_image", "source_age"])
        writer.writeheader()
        writer.writerows(rows)
    print("unique_sources:", len(rows))
    print("output:", output)


if __name__ == "__main__":
    main()
