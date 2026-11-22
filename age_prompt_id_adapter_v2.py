"""Landmark-conditioned prompt identity adapter.

This module operates on the same-state velocity cache.  It uses landmark
geometry as an identity condition and predicts a small correction to the
aging velocity.  The training-side identity objective is a protected-region
velocity proxy; it is not an image-level ArcFace loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class PromptIDAdapterV2Config:
    velocity_dim: int = 64
    hidden_dim: int = 128
    condition_dim: int = 64
    identity_dim: int = 512
    landmark_points: int = 478
    max_correction: float = 0.1


def normalize_landmarks(landmark_xy: torch.Tensor) -> torch.Tensor:
    """Normalize [B, P, 2] landmarks to translation/scale-stable coordinates."""
    x = landmark_xy.float()
    center = x.mean(dim=1, keepdim=True)
    x = x - center
    scale = x.square().sum(dim=-1).sqrt().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return x / scale.unsqueeze(-1)


class PromptIdentityAdapterV2(nn.Module):
    def __init__(self, config: PromptIDAdapterV2Config):
        super().__init__()
        self.config = config
        landmark_dim = config.landmark_points * 2

        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.condition_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim + config.identity_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(config.velocity_dim * 2 + 1, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(config.hidden_dim, config.velocity_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        base: torch.Tensor,
        age: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        timestep: torch.Tensor,
        strength: torch.Tensor,
        landmark_xy: torch.Tensor,
        identity_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if landmark_xy.ndim == 2:
            landmark_xy = landmark_xy.reshape(landmark_xy.shape[0], -1, 2)
        landmark = normalize_landmarks(landmark_xy).flatten(1)
        landmark_condition = self.landmark_encoder(landmark)

        if identity_embedding is None:
            identity_embedding = base.new_zeros((base.shape[0], self.config.identity_dim))
        identity_embedding = identity_embedding.float()
        if identity_embedding.shape[-1] != self.config.identity_dim:
            raise ValueError(
                f"identity embedding dim {identity_embedding.shape[-1]} "
                f"!= configured {self.config.identity_dim}"
            )
        identity_embedding = identity_embedding / identity_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        age_span = ((target_age - source_age) / 100.0).unsqueeze(-1)
        scalar_condition = torch.cat(
            [source_age.unsqueeze(-1) / 100.0, target_age.unsqueeze(-1) / 100.0, age_span],
            dim=-1,
        )
        condition = self.condition_encoder(
            torch.cat([landmark_condition, identity_embedding, scalar_condition], dim=-1)
        ).unsqueeze(1)
        condition = condition.expand(-1, base.shape[1], -1)

        token_scaler = timestep.reshape(-1, 1, 1).expand(-1, base.shape[1], 1)
        token_input = torch.cat([base, age - base, token_scaler], dim=-1)
        token = self.token_encoder(token_input)
        hidden = self.fuse(torch.cat([token, condition], dim=-1))
        return torch.tanh(self.output(hidden)) * self.config.max_correction


def apply_correction(
    age_velocity: torch.Tensor,
    correction: torch.Tensor,
    max_injection_ratio: float = 0.35,
):
    """Cap correction relative to the current aging velocity per sample."""
    age_rms = age_velocity.square().mean(dim=tuple(range(1, age_velocity.ndim)), keepdim=True).sqrt()
    correction_rms = correction.square().mean(dim=tuple(range(1, correction.ndim)), keepdim=True).sqrt()
    allowed = age_rms * max_injection_ratio
    cap = torch.minimum(torch.ones_like(correction_rms), allowed / correction_rms.clamp_min(1e-8))
    return age_velocity + correction * cap, cap


def config_dict(config: PromptIDAdapterV2Config):
    return asdict(config)
