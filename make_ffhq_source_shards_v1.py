"""Split one filtered FFHQ age group into deterministic source shards."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def first(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--age_group", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader
                if row.get("eval_age_group") == args.age_group]

    if not rows:
        raise ValueError(f"No rows found for eval_age_group={args.age_group}")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")

    random.Random(args.seed).shuffle(rows)

    extra_fields = ["source_id", "source_image", "source_age"]
    output_fields = fieldnames + [x for x in extra_fields if x not in fieldnames]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    base, remainder = divmod(total, args.num_shards)
    offset = 0
    counts = []
    for shard_index in range(args.num_shards):
        size = base + (1 if shard_index < remainder else 0)
        shard_rows = rows[offset:offset + size]
        offset += size
        output = output_dir / f"{args.age_group.replace('-', '_')}_part_{shard_index + 1:02d}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for row in shard_rows:
                relative_path = first(row, ("relative_path", "path"))
                source_image = first(row, ("source_image", "image_path", "absolute_path"))
                if not source_image and relative_path:
                    source_image = str(Path(args.dataset_root) / relative_path)
                # Prefer the filename so IDs such as 07116 keep their leading
                # zero and continue to match generated image paths.
                source_id = first(row, ("source_id", "file_name", "image_number"))
                if not source_id:
                    source_id = Path(relative_path or source_image).stem
                source_age = first(row, ("source_age", "eval_age_midpoint", "age_group_midpoint"))
                enriched = dict(row)
                enriched.update({
                    "source_id": Path(str(source_id)).stem,
                    "source_image": source_image,
                    "source_age": source_age,
                })
                writer.writerow(enriched)
        counts.append((output.name, len(shard_rows)))

    print(f"age_group={args.age_group} total={total} shards={args.num_shards}")
    for name, count in counts:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
