"""Evaluate a trained velocity adapter on the held-out cache split."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from train_age_velocity_adapter_v1 import AdaptiveAgeVelocityAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = AdaptiveAgeVelocityAdapter(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    index = json.loads(Path(a.index).read_text(encoding="utf-8"))
    rows = [r for r in index["rows"] if r["split"] == a.split]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cache"]].append(row)
    cache_map = {}
    sum_mse = sum_cos = sum_pred_rms = sum_target_rms = 0.0
    count = 0

    with torch.no_grad():
        for path, shard_rows in sorted(grouped.items()):
            obj = cache_map.get(path)
            if obj is None:
                print(f"loading {Path(path).name}", flush=True)
                obj = torch.load(path, map_location="cpu", weights_only=False)
                cache_map.clear()
                cache_map[path] = obj
            for start in range(0, len(shard_rows), a.batch_size):
                chosen = shard_rows[start:start + a.batch_size]
                idx = torch.tensor([r["row"] for r in chosen], dtype=torch.long)
                base = obj["base_velocity"].index_select(0, idx).to(device)
                detail = obj["detail_velocity"].index_select(0, idx).to(device)
                target = obj["teacher_delta"].index_select(0, idx).to(device).float()
                source_age = torch.tensor([r["source_age"] for r in chosen], device=device)
                target_age = torch.tensor([r["target_age"] for r in chosen], device=device)
                timestep = torch.tensor([r["timestep"] for r in chosen], device=device)
                pred = model(
                    base, source_age, target_age, timestep,
                    detail if config.get("use_detail_input", False) else None,
                ).float()
                pred_flat, target_flat = pred.flatten(1), target.flatten(1)
                cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-6)
                pred_rms = pred.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
                target_rms = target.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
                n = len(chosen)
                sum_mse += F.mse_loss(pred, target, reduction="sum").item()
                sum_cos += cosine.sum().item()
                sum_pred_rms += pred_rms.sum().item()
                sum_target_rms += target_rms.sum().item()
                count += n

    result = {
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "split": a.split,
        "samples": count,
        "mse": sum_mse / (count * 1024 * 64),
        "cosine_similarity": sum_cos / count,
        "pred_rms": sum_pred_rms / count,
        "target_rms": sum_target_rms / count,
        "pred_to_target_rms": (sum_pred_rms / max(sum_target_rms, 1e-12)),
    }
    print(json.dumps(result, indent=2))
    out = Path(a.checkpoint).with_name(Path(a.checkpoint).stem + f"_{a.split}_metrics.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
