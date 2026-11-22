"""Train an adaptive FLUX aging-direction adapter from shard velocity caches.

The adapter predicts the detail-minus-base velocity residual.  Shards are
loaded one at a time; the script never concatenates the full dataset.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAgeVelocityAdapter(nn.Module):
    def __init__(self, token_dim=64, hidden_dim=256, cond_dim=128, use_detail_input=False):
        super().__init__()
        self.use_detail_input = bool(use_detail_input)
        self.config = {
            "token_dim": token_dim, "hidden_dim": hidden_dim, "cond_dim": cond_dim,
            "use_detail_input": self.use_detail_input,
        }
        self.cond = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim), nn.LayerNorm(cond_dim),
        )
        self.input_fusion = (
            nn.Sequential(nn.LayerNorm(token_dim * 2), nn.Linear(token_dim * 2, token_dim), nn.SiLU())
            if self.use_detail_input else None
        )
        self.film = nn.Linear(cond_dim, token_dim * 2)
        self.body = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        # Start as a no-op. The residual head learns only when supervised.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.log_gain = nn.Parameter(torch.tensor(-1.0))

    def forward(self, base, source_age, target_age, timestep, detail=None):
        if self.use_detail_input:
            if detail is None:
                raise ValueError("detail velocity is required when use_detail_input=True")
            base = self.input_fusion(torch.cat([base.float(), detail.float()], dim=-1))
        age_delta = target_age - source_age
        c = torch.stack([source_age / 80.0, target_age / 80.0,
                         age_delta / 60.0, timestep], dim=-1)
        c = self.cond(c)
        gamma, beta = self.film(c).chunk(2, dim=-1)
        x = base.float() * (1.0 + 0.25 * torch.tanh(gamma[:, None])) + beta[:, None]
        pred = self.body(x)
        return pred * self.log_gain.exp().clamp(max=10.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument(
        "--shard_steps", type=int, default=500,
        help="Keep one cache shard in memory for this many optimizer steps.",
    )
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=0.1)
    p.add_argument("--lambda_rms", type=float, default=0.1)
    p.add_argument(
        "--lambda_magnitude", type=float, default=1.0,
        help="Relative RMS matching weight; prevents near-zero residual collapse.",
    )
    p.add_argument(
        "--resume_from", default=None,
        help="Resume model/optimizer/step from a checkpoint saved by this script.",
    )
    p.add_argument(
        "--use_detail_input", action="store_true",
        help="Diagnostic/oracle mode: provide detail_velocity as an additional input.",
    )
    return p.parse_args()


def batch_from_rows(rows, cache_map, device):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["cache"], []).append(r)
    path, selected = random.choice(list(grouped.items()))
    obj = cache_map.get(path)
    if obj is None:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        cache_map.clear()
        cache_map[path] = obj
    picks = random.choices(selected, k=min(len(selected), len(selected)))
    # The caller samples the desired batch size after grouping.
    return obj, selected, device


def make_batch(train_rows, cache_map, batch_size, device, path=None):
    if path is None:
        path = random.choice(list({r["cache"] for r in train_rows}))
    candidates = [r for r in train_rows if r["cache"] == path]
    chosen = random.choices(candidates, k=batch_size)
    obj = cache_map.get(path)
    if obj is None:
        print(f"loading shard {Path(path).name}", flush=True)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        cache_map.clear()
        cache_map[path] = obj
    idx = torch.tensor([r["row"] for r in chosen], dtype=torch.long)
    return (
        obj["base_velocity"].index_select(0, idx).to(device),
        obj["detail_velocity"].index_select(0, idx).to(device),
        obj["teacher_delta"].index_select(0, idx).to(device),
        torch.tensor([r["source_age"] for r in chosen], device=device),
        torch.tensor([r["target_age"] for r in chosen], device=device),
        torch.tensor([r["timestep"] for r in chosen], device=device),
    )


def main():
    a = parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    index = json.loads(Path(a.index).read_text(encoding="utf-8"))
    train = [r for r in index["rows"] if r["split"] == "train"]
    val = [r for r in index["rows"] if r["split"] == "val"]
    model = AdaptiveAgeVelocityAdapter(
        hidden_dim=a.hidden_dim, use_detail_input=a.use_detail_input
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    start_step = 0
    if a.resume_from:
        resume = torch.load(a.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        if "optimizer" in resume:
            opt.load_state_dict(resume["optimizer"])
            for state in opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        start_step = int(resume.get("step", 0))
        print(f"RESUME checkpoint={a.resume_from} step={start_step}", flush=True)
    cache_map = {}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_paths = sorted({r["cache"] for r in train})
    if not train_paths:
        raise ValueError("index contains no training rows")

    def save_checkpoint(step, name):
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config,
                "index": str(Path(a.index).resolve()),
                "step": step,
                "optimizer": opt.state_dict(),
            },
            out / name,
        )

    current_path = None
    if start_step >= a.steps:
        print(f"checkpoint already reached target step={start_step}", flush=True)
        return
    for step in range(start_step + 1, a.steps + 1):
        model.train()
        if current_path is None or (step - 1) % max(a.shard_steps, 1) == 0:
            current_path = train_paths[((step - 1) // max(a.shard_steps, 1)) % len(train_paths)]
            print(f"step={step} shard={Path(current_path).name}", flush=True)
            if cache_map and current_path not in cache_map:
                cache_map.clear()
                gc.collect()
        base, detail, target, sa, ta, ts = make_batch(
            train, cache_map, a.batch_size, device, current_path
        )
        pred = model(base, sa, ta, ts, detail if a.use_detail_input else None)
        target = target.float()
        mse = F.mse_loss(pred, target)
        pred_flat = pred.float().flatten(1)
        target_flat = target.flatten(1)
        pred_norm = pred_flat.norm(dim=1)
        target_norm = target_flat.norm(dim=1).clamp_min(1e-6)
        # The residual head is deliberately initialized to zero. Avoid taking
        # a cosine derivative at exactly zero; MSE warms the head up first.
        active = pred_norm > 1e-4
        if active.any():
            cosine_similarity = (
                (pred_flat[active] * target_flat[active]).sum(dim=1)
                / (pred_norm[active].clamp_min(1e-4) * target_norm[active])
            ).clamp(-1.0, 1.0)
            cosine = 1.0 - cosine_similarity.mean()
        else:
            cosine = pred_flat.sum() * 0.0
        # Clamp before sqrt because the residual head starts at exactly zero;
        # sqrt(0) can produce an infinite backward derivative.
        pred_rms = pred.float().square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
        target_rms = target.float().square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
        rms = (pred_rms - target_rms).square().mean()
        relative_magnitude = (
            (pred_rms - target_rms) / target_rms.clamp_min(1e-3)
        ).square().mean()
        loss = (
            a.lambda_mse * mse
            + a.lambda_cos * cosine
            + a.lambda_rms * rms
            + a.lambda_magnitude * relative_magnitude
        )
        opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite training state at step={step}: "
                f"loss={loss.item()} grad_norm={grad_norm.item()}"
            )
        opt.step()
        if step == 1 or step % a.log_every == 0:
            print(f"step={step} loss={loss.item():.6f} mse={mse.item():.6f} cosine={cosine.item():.6f} rms={rms.item():.6f} magnitude={relative_magnitude.item():.6f} gain={model.log_gain.exp().item():.4f} grad_norm={grad_norm.item():.6f}", flush=True)
        if a.checkpoint_every > 0 and step % a.checkpoint_every == 0:
            save_checkpoint(step, f"adaptive_age_velocity_adapter_step{step}.pt")
    save_checkpoint(a.steps, "adaptive_age_velocity_adapter.pt")
    print(f"DONE checkpoint={out / 'adaptive_age_velocity_adapter.pt'} train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()
