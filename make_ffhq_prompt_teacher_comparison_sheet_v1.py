"""Make a deterministic visual comparison sheet for FFHQ prompt outputs."""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_sources", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--path_rewrite", action="append", default=[])
    return p.parse_args()


def rewrites(values):
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --path_rewrite: {value}")
        result.append(tuple(value.split("=", 1)))
    return result


def resolve(value, rules):
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return candidate
    raw = str(value)
    for source, target in rules:
        if raw.startswith(source):
            candidate = Path(target + raw[len(source):]).expanduser()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Missing image: {value}")


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_id", "source_image", "target_age", "condition", "output_image"}
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")
    return rows


def fit_image(path, size):
    with Image.open(path) as image:
        return ImageOps.fit(
            ImageOps.exif_transpose(image).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )


def choose_groups(rows, count, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_id"], []).append(row)
    ages = sorted({float(row["target_age"]) for row in rows})
    if not ages or not all(math.isfinite(age) for age in ages):
        raise ValueError("Invalid target ages")
    eligible = []
    for source_id, group in grouped.items():
        conditions = {row["condition"] for row in group}
        present = {(row["condition"], float(row["target_age"])) for row in group}
        expected = {(condition, age) for condition in ("base", "detail") for age in ages}
        if (conditions == {"base", "detail"} and present == expected
                and len(group) == len(expected)
                and len({row["source_image"] for row in group}) == 1):
            eligible.append(source_id)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} complete source groups; need {count}")
    selected = random.Random(seed).sample(sorted(eligible), count)
    return selected, ages, grouped


def main():
    args = parse_args()
    if args.num_sources < 1 or args.tile_size < 64:
        raise ValueError("Invalid --num_sources or --tile_size")
    root = Path(args.eval_dir)
    output = Path(args.output_dir)
    manifest_path = root / "generation_manifest.csv"
    rows = load_rows(manifest_path)
    rules = rewrites(args.path_rewrite)
    selected, ages, grouped = choose_groups(rows, args.num_sources, args.seed)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "selected_sources.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "source_image"])
        writer.writeheader()
        writer.writerows({"source_id": source_id, "source_image": grouped[source_id][0]["source_image"]}
                         for source_id in selected)

    cell = args.tile_size
    label_height = max(32, cell // 8)
    left_label = max(130, cell // 2)
    header_height = label_height
    width = left_label + (len(ages) + 1) * cell
    height = header_height + len(selected) * 2 * (cell + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, size=max(12, min(20, cell // 14)))
            break
        except OSError:
            continue

    def put_text(x, y, text):
        draw.text((x, y), text, fill="black", font=font)

    put_text(8, 8, "source")
    for col, age in enumerate([None] + ages):
        label = "original" if age is None else f"target {age:g}"
        put_text(left_label + col * cell + 6, 8, label)

    for index, source_id in enumerate(selected):
        group = grouped[source_id]
        source_path = resolve(group[0]["source_image"], rules)
        row_base = header_height + index * 2 * (cell + label_height)
        source_tile = fit_image(source_path, cell)
        for condition_index, condition in enumerate(("base", "detail")):
            row_top = row_base + condition_index * (cell + label_height)
            put_text(8, row_top + 8, f"{index + 1:02d} / {condition}")
            put_text(8, row_top + label_height + 8, source_id[:8])
            sheet.paste(source_tile, (left_label, row_top + label_height))
            for col, age in enumerate(ages, start=1):
                matches = [row for row in group if row["condition"] == condition and float(row["target_age"]) == age]
                if len(matches) != 1:
                    raise ValueError(f"Expected one image for {source_id}, {condition}, {age}")
                image = fit_image(resolve(matches[0]["output_image"], rules), cell)
                sheet.paste(image, (left_label + col * cell, row_top + label_height))
        group_sheet = Image.new("RGB", (width, header_height + 2 * (cell + label_height)), "white")
        group_sheet.paste(sheet.crop((0, 0, width, header_height)), (0, 0))
        group_sheet.paste(sheet.crop((0, row_base, width, row_base + 2 * (cell + label_height))), (0, header_height))
        group_sheet.save(output / f"group_{index + 1:02d}.jpg", quality=95)
    path = output / f"prompt_teacher_random{len(selected)}_comparison.jpg"
    sheet.save(path, quality=95)
    print(f"selected_sources={','.join(selected)}")
    print(f"comparison_sheet={path}")


if __name__ == "__main__":
    main()
