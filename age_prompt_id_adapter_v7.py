"""Structural, age-orthogonal identity correction for the prompt-ID route.

v7 keeps the v6 adapter backbone but changes where and in which direction its
correction can act:

* correction is gated toward eyes, nose, mouth, and face contour landmarks;
* the component parallel to the current aging velocity is removed per token.

The second constraint prevents the identity branch from improving identity by
simply cancelling the aging direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from age_prompt_id_adapter_v2 import PromptIdentityAdapterV2
from age_prompt_id_adapter_v6 import (
    PromptIDAdapterV6Config,
    landmark_face_gate,
)


@dataclass
class PromptIDAdapterV7Config(PromptIDAdapterV6Config):
    structure_sigma: float = 0.16
    structure_floor: float = 0.05
    orthogonal_eps: float = 1e-6


# MediaPipe Face Mesh indices for stable identity-bearing structures.
STRUCTURE_GROUPS = {
    "eyes": (33, 133, 159, 145, 263, 362, 386, 374, 70, 300),
    "nose": (1, 2, 4, 5, 19, 94, 168, 195, 278, 331, 344),
    "mouth": (0, 13, 14, 17, 61, 291, 78, 308),
    "contour": (10, 152, 234, 454, 127, 356, 132, 361),
}


def _normalize_points(landmark_xy: torch.Tensor, image_size: float) -> torch.Tensor:
    points = landmark_xy.float()
    if float(points.detach().amax()) > 1.5:
        points = points / max(float(image_size), 1.0)
    return points.clamp(0.0, 1.0)


def landmark_structure_gate(
    landmark_xy: torch.Tensor,
    token_count: int,
    image_size: float = 512.0,
    sigma: float = 0.16,
    floor: float = 0.05,
    face_pad: float = 0.12,
    face_temperature: float = 0.08,
) -> torch.Tensor:
    """Build a soft [B, token_count, 1] gate around identity structures."""
    if landmark_xy.ndim == 2:
        landmark_xy = landmark_xy.reshape(landmark_xy.shape[0], -1, 2)
    if landmark_xy.ndim != 3 or landmark_xy.shape[-1] != 2:
        raise ValueError(f"expected [B, P, 2] landmarks, got {tuple(landmark_xy.shape)}")
    side = int(token_count ** 0.5)
    if side * side != token_count:
        raise ValueError(f"expected square token grid, got {token_count}")
    points = _normalize_points(landmark_xy, image_size)
    if points.shape[1] < 478:
        raise ValueError(f"expected 478 landmarks, got {points.shape[1]}")

    axis = torch.linspace(
        0.5 / side,
        1.0 - 0.5 / side,
        side,
        device=points.device,
        dtype=points.dtype,
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).unsqueeze(0)
    union = points.new_zeros((points.shape[0], token_count))

    # Scale the regions relative to the face extent so the gate works across
    # different portrait crops without requiring another mask cache.
    face_extent = (points.amax(dim=1) - points.amin(dim=1)).mean(dim=-1).clamp_min(0.05)
    for indices in STRUCTURE_GROUPS.values():
        center = points[:, list(indices)].mean(dim=1, keepdim=True)
        local_sigma = (face_extent[:, None] * float(sigma)).clamp_min(0.025)
        distance = (grid - center).square().sum(dim=-1)
        union = torch.maximum(union, torch.exp(-distance / (2.0 * local_sigma.square())))

    face = landmark_face_gate(
        landmark_xy,
        token_count,
        pad=face_pad,
        temperature=face_temperature,
        image_size=image_size,
    ).squeeze(-1)
    gate = face * (float(floor) + (1.0 - float(floor)) * union)
    return gate.unsqueeze(-1).to(dtype=landmark_xy.dtype)


class PromptIdentityAdapterV7(PromptIdentityAdapterV2):
    """v2 MLP with structural gating and age-direction orthogonalization."""

    requires_spatial_gate = True
    requires_structure_gate = True

    def __init__(self, config: PromptIDAdapterV7Config):
        super().__init__(config)
        self.config = config

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
        spatial_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        correction = super().forward(
            base,
            age,
            source_age,
            target_age,
            timestep,
            strength,
            landmark_xy,
            identity_embedding,
        )
        age_norm_sq = age.float().square().sum(dim=-1, keepdim=True)
        projection = (correction.float() * age.float()).sum(dim=-1, keepdim=True)
        correction = correction - (
            projection / age_norm_sq.clamp_min(float(self.config.orthogonal_eps))
        ).to(dtype=correction.dtype) * age
        if spatial_gate is None:
            spatial_gate = landmark_structure_gate(
                landmark_xy,
                base.shape[1],
                image_size=self.config.landmark_image_size,
                sigma=self.config.structure_sigma,
                floor=self.config.structure_floor,
                face_pad=self.config.face_gate_pad,
                face_temperature=self.config.face_gate_temperature,
            )
        return correction * spatial_gate.to(device=correction.device, dtype=correction.dtype)

