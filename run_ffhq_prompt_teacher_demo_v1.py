"""Matched, source-only FFHQ prompt experiment; no adapter is loaded."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys


BASE_PROMPT = "a realistic close-up portrait photograph of a person at age {age}"
IDENTITY_PROMPT = (
    "preserving recognizable identity, eye shape, nose shape, mouth shape, "
    "facial proportions, expression, hairstyle, pose, lighting, clothing and background, "
    "without retouching or beautification"
)


def detail_prompt_for_age(age: float) -> str:
    if age < 30:
        appearance = (
            "natural young-adult facial appearance and healthy skin texture, "
            "without wrinkles, exaggerated smooth skin or childish features"
        )
    elif age < 40:
        appearance = (
            "natural adult facial appearance and realistic skin texture, "
            "without adding pronounced wrinkles or artificial aging"
        )
    elif age < 50:
        appearance = (
            "subtle age-consistent skin texture, fine forehead lines and mild periocular "
            "and perioral texture, with a natural adult appearance"
        )
    elif age < 60:
        appearance = (
            "moderate but natural age-consistent skin texture, visible forehead, periocular "
            "and perioral lines, with mild age-related soft-tissue change"
        )
    else:
        appearance = (
            "more pronounced but natural age-consistent skin texture, forehead, periocular "
            "and perioral lines, and mild age-related soft-tissue change, without caricature"
        )
    return f"{BASE_PROMPT.format(age=f'{age:g}')}, {appearance}, {IDENTITY_PROMPT}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--source_group", default="30-39")
    p.add_argument("--max_sources", type=int, default=30)
    p.add_argument("--min_age_confidence", type=float, default=0.6)
    p.add_argument("--target_ages", default="25,35,45,55,65")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", default="output/ffhq_35_prompt_teacher_demo_v1")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--prepare_only", action="store_true")
    return p.parse_args()


def select_sources(args):
    root = Path(args.data_dir).resolve()
    manifest = Path(args.manifest) if args.manifest else root / "age_group_selection.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"relative_path", "eval_age_group", "eval_age_midpoint", "age_group_confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest fields: {sorted(missing)}")
        rows = sorted(reader, key=lambda r: r["relative_path"])
    selected, seen = [], set()
    report = {"group_rows": 0, "low_or_invalid_confidence": 0, "missing_images": 0}
    for row in rows:
        if row["eval_age_group"] != args.source_group:
            continue
        report["group_rows"] += 1
        try:
            confidence = float(row["age_group_confidence"])
        except ValueError:
            confidence = float("nan")
        if not args.min_age_confidence <= confidence <= 1.0:
            report["low_or_invalid_confidence"] += 1
            continue
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image outside data_dir: {path}")
        if not path.is_file():
            report["missing_images"] += 1
            continue
        if path in seen:
            continue
        seen.add(path)
        row = dict(row, source_image=str(path))
        row["source_id"] = hashlib.sha256(row["relative_path"].encode()).hexdigest()[:16]
        selected.append(row)
    report["eligible_unique_images"] = len(selected)
    if len(selected) < args.max_sources:
        raise ValueError(f"Need {args.max_sources} eligible sources; report={report}")
    return random.Random(args.seed).sample(selected, args.max_sources), report


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(args, sources, ages, output):
    jobs = []
    for index, source in enumerate(sources):
        for age in ages:
            for condition, template in (("base", BASE_PROMPT), ("detail", None)):
                name = f"{age:g}"
                jobs.append({
                    "source_id": source["source_id"],
                    "source_image": source["source_image"],
                    "source_group": source["eval_age_group"],
                    "source_age_midpoint": source["eval_age_midpoint"],
                    "target_age": age,
                    "condition": condition,
                    "seed": args.seed + index,
                    "prompt": (
                        detail_prompt_for_age(age)
                        if condition == "detail"
                        else template.format(age=name)
                    ),
                    "output_image": str(output / condition / f"age_{name}" / (source["source_id"] + ".png")),
                })
    return jobs


def make_contact_sheet(source, jobs, output):
    from PIL import Image, ImageDraw, ImageOps

    ages = list(dict.fromkeys(j["target_age"] for j in jobs))
    tile, header = 192, 28
    sheet = Image.new("RGB", ((len(ages) + 1) * tile, 2 * (tile + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, condition in enumerate(("base", "detail")):
        paths = [source["source_image"]] + [j["output_image"] for j in jobs if j["condition"] == condition]
        labels = ["source (30-39 group)" if source["eval_age_group"] == "30-39" else "source"]
        labels += [f"{condition}: {age:g}" for age in ages]
        for col, (path, label) in enumerate(zip(paths, labels)):
            with Image.open(path) as im:
                im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (tile, tile))
                sheet.paste(im, (col * tile, row * (tile + header) + header))
            draw.text((col * tile + 4, row * (tile + header) + 6), label, fill="black")
    folder = output / "contact_sheets"
    folder.mkdir(exist_ok=True)
    sheet.save(folder / (source["source_id"] + ".jpg"))


@contextmanager
def progress_output():
    # Keep progress visible while the shell redirects verbose model logs.
    descriptor = os.environ.get("FFHQ_PROGRESS_FD")
    if descriptor is None:
        yield sys.stderr
    else:
        with os.fdopen(os.dup(int(descriptor)), "w", buffering=1) as stream:
            yield stream


def run(progress_file):
    args = parse_args()
    ages = [float(x) for x in args.target_ages.split(",")]
    if not ages or len(set(ages)) != len(ages) or not all(18 <= x <= 100 for x in ages):
        raise ValueError("target_ages must be unique adult ages between 18 and 100")
    if args.max_sources < 1 or not 0 <= args.min_age_confidence <= 1:
        raise ValueError("Invalid sample size or age confidence threshold")
    if args.resolution < 64 or args.resolution % 16 or not 0 < args.strength <= 1 or args.steps * args.strength < 1:
        raise ValueError("Use resolution divisible by 16 and at least one effective denoising step")
    output = Path(args.output_dir).resolve()
    print("Checking source manifest...", file=progress_file, flush=True)
    sources, report = select_sources(args)
    jobs = build_jobs(args, sources, ages, output)
    config = {k: v for k, v in vars(args).items() if k != "prepare_only"}
    plan = {"config": config, "selection": report, "jobs": jobs,
            "pipeline": "diffusers.FluxImg2ImgPipeline", "adapter_loaded": False,
            "note": "Age midpoint is not a measured source age; unique files do not prove unique identities."}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "run_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing output has a different plan. Choose a new output_dir.")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output without run_plan.json. Choose a new output_dir.")
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_csv(output / "selected_sources.csv", sources)
    write_csv(output / "generation_manifest.csv", jobs)
    print(f"sources={len(sources)} planned_images={len(jobs)} selection={report}", flush=True)
    print(f"Sources: {len(sources)} | Planned images: {len(jobs)}", file=progress_file, flush=True)
    if args.prepare_only:
        print(f"Prepared only; no model loaded. output={output}", flush=True)
        return

    print("Loading FLUX; image progress starts after model loading.", file=progress_file, flush=True)
    import diffusers
    import torch
    from diffusers import FluxImg2ImgPipeline
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --prepare_only to validate the manifest on CPU")
    pipe = FluxImg2ImgPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    (output / "runtime.json").write_text(json.dumps({"torch": torch.__version__, "diffusers": diffusers.__version__}, indent=2))
    progress = tqdm(total=len(jobs), desc="FFHQ images", unit="img", ascii=True,
                    file=progress_file, disable=False, dynamic_ncols=True)
    try:
        generate_images(args, sources, jobs, output, pipe, progress)
    finally:
        progress.close()
    print(f"DONE output={output}", flush=True)


def generate_images(args, sources, jobs, output, pipe, progress):
    import torch
    from PIL import Image, ImageOps

    completed = 0
    reused = 0
    for source_index, source in enumerate(sources, start=1):
        source_jobs = [j for j in jobs if j["source_id"] == source["source_id"]]
        with Image.open(source["source_image"]) as image:
            image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (args.resolution, args.resolution), method=Image.Resampling.LANCZOS)
        for job in source_jobs:
            progress.set_postfix_str(
                f"source={source_index}/{len(sources)} {job['condition']} age={job['target_age']:g} reused={reused}")
            destination = Path(job["output_image"])
            valid = False
            if destination.exists():
                try:
                    with Image.open(destination) as existing:
                        existing.load()
                        valid = existing.size == (args.resolution, args.resolution)
                except OSError:
                    pass
            if not valid:
                # Reset the generator for every condition and age to couple stochastic inputs.
                with torch.inference_mode():
                    result = pipe(prompt=job["prompt"], image=image, height=args.resolution,
                                  width=args.resolution, strength=args.strength,
                                  num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                                  max_sequence_length=512,
                                  generator=torch.Generator(device="cpu").manual_seed(job["seed"])).images[0]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.png")
                result.save(temporary)
                temporary.replace(destination)
            else:
                reused += 1
            completed += 1
            progress.update(1)
            print(f"completed_images={completed}/{len(jobs)} reused={reused}", flush=True)
        make_contact_sheet(source, source_jobs, output)
    progress.set_postfix_str(f"generated={completed - reused} reused={reused}")


def main():
    with progress_output() as stream:
        run(stream)


if __name__ == "__main__":
    main()
