"""Validate shard caches and build a source-disjoint train/validation index.

The velocity tensors stay in their original shard files.  The generated JSON
index only stores metadata and row numbers, so it is safe for large datasets.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm


REQUIRED = ("base_velocity", "detail_velocity", "teacher_delta", "source_age", "target_age")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_age_delta", type=float, default=1.0)
    p.add_argument("--check_delta", action="store_true")
    p.add_argument(
        "--skip_incompatible",
        action="store_true",
        help="Skip caches whose token shape is not [N, 1024, 64].",
    )
    return p.parse_args()


def source_key(record, row):
    path = str(record.get("source_image", ""))
    if path:
        return path
    return str(record.get("source_id", row))


def main():
    a = args()
    root = Path(a.cache_root)
    paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise SystemExit(f"no .pt cache files under {root}")

    rows = []
    report = []
    print(f"found {len(paths)} cache files under {root}", flush=True)
    for path in tqdm(paths, desc="validating velocity caches", unit="shard", file=sys.stdout):
        print(f"\nloading {path.name}", flush=True)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        missing = [k for k in REQUIRED if k not in obj]
        if missing:
            print(f"SKIP {path}: missing {missing}")
            continue
        base = obj["base_velocity"]
        detail = obj["detail_velocity"]
        delta = obj["teacher_delta"]
        n = int(base.shape[0])
        if base.ndim != 3 or base.shape[1:] != (1024, 64):
            message = f"{path}: unexpected velocity shape {tuple(base.shape)}"
            if a.skip_incompatible:
                print(f"SKIP incompatible cache: {message}")
                continue
            raise ValueError(message)
        if tuple(detail.shape) != tuple(base.shape) or tuple(delta.shape) != tuple(base.shape):
            raise ValueError(f"{path}: velocity shapes do not match")
        source_age = obj["source_age"].reshape(-1).tolist()
        target_age = obj["target_age"].reshape(-1).tolist()
        records = obj.get("records", [])
        if len(source_age) != n or len(target_age) != n or len(records) != n:
            raise ValueError(f"{path}: tensor/record count mismatch")
        if not all(torch.isfinite(x).all().item() for x in (base, detail, delta)):
            raise ValueError(f"{path}: non-finite velocity value")
        if a.check_delta:
            err = (delta.float() - (detail.float() - base.float())).abs().max().item()
            if err > 1e-4:
                raise ValueError(f"{path}: teacher_delta mismatch, max error={err}")

        kept = 0
        for i, record in enumerate(records):
            sa, ta = float(source_age[i]), float(target_age[i])
            if ta - sa < a.min_age_delta:
                continue
            rows.append({
                "cache": str(path.resolve()), "row": i,
                "source_key": source_key(record, i),
                "source_age": sa, "target_age": ta,
                "timestep": float(obj.get("timestep", [0.0] * n)[i]),
            })
            kept += 1
        report.append({"cache": str(path), "records": n, "kept": kept})
        print(f"validated {path.name}: records={n} kept_positive_age={kept}", flush=True)
        del obj

    if not rows:
        raise SystemExit("no positive-age records found")
    sources = sorted({r["source_key"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(sources)
    n_val = max(1, round(len(sources) * a.val_fraction))
    val_sources = set(sources[:n_val])
    for r in rows:
        r["split"] = "val" if r["source_key"] in val_sources else "train"

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1, "seed": a.seed, "min_age_delta": a.min_age_delta,
        "num_records": len(rows), "num_sources": len(sources),
        "num_train": sum(r["split"] == "train" for r in rows),
        "num_val": sum(r["split"] == "val" for r in rows),
        "shards": report, "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("num_records", "num_sources", "num_train", "num_val")}, indent=2))
    print("age_pairs_by_source_age:", Counter((r["source_age"], r["target_age"]) for r in rows))


if __name__ == "__main__":
    main()
