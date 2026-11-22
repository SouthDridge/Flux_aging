"""Spatially gated prompt-identity adapter.

This is the first v6 prototype.  It keeps the v2/v5 velocity adapter intact,
but gates its correction with a soft face-region mask derived from the 478
MediaPipe landmarks.  The gate is applied per packed FLUX token, so identity
correction is not injected uniformly into the background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from age_prompt_id_adapter_v2 import (
    PromptIDAdapterV2Config,
    PromptIdentityAdapterV2,
)


@dataclass
class PromptIDAdapterV6Config(PromptIDAdapterV2Config):
    face_gate_pad: float = 0.12
    face_gate_temperature: float = 0.08
    landmark_image_size: float = 512.0


def landmark_face_gate(
    landmark_xy: torch.Tensor,
    token_count: int,
    pad: float = 0.12,
    temperature: float = 0.08,
    image_size: float = 512.0,
) -> torch.Tensor:
    """Return a soft [B, token_count, 1] gate from normalized face landmarks.

    The cache stores MediaPipe coordinates in normalized image space.  A soft
    ellipse around their bounding box avoids a hard correction boundary while
    excluding most background tokens.  FLUX's packed image tokens are assumed
    to be a square row-major grid, matching the existing velocity cache.
    """
    if landmark_xy.ndim == 2:
        landmark_xy = landmark_xy.reshape(landmark_xy.shape[0], -1, 2)
    if landmark_xy.ndim != 3 or landmark_xy.shape[-1] != 2:
        raise ValueError(f"expected [B, P, 2] landmarks, got {tuple(landmark_xy.shape)}")
    side = int(token_count ** 0.5)
    if side * side != token_count:
        raise ValueError(f"expected square token grid, got {token_count}")

    points = landmark_xy.float()
    # The existing landmark cache stores pixel coordinates from the source
    # image.  Accept both that format and normalized MediaPipe coordinates.
    if float(points.detach().amax()) > 1.5:
        points = points / max(float(image_size), 1.0)
    points = points.clamp(0.0, 1.0)
    lo = points.amin(dim=1)
    hi = points.amax(dim=1)
    center = (lo + hi) * 0.5
    radius = ((hi - lo) * 0.5 * (1.0 + float(pad))).clamp_min(1e-4)

    axis = torch.linspace(
        0.5 / side,
        1.0 - 0.5 / side,
        side,
        device=points.device,
        dtype=points.dtype,
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = (xx.reshape(1, -1) - center[:, None, 0]) / radius[:, None, 0]
    dy = (yy.reshape(1, -1) - center[:, None, 1]) / radius[:, None, 1]
    distance = dx.square() + dy.square()
    softness = max(float(temperature), 1e-3)
    gate = torch.sigmoid((1.0 - distance) / softness)
    return gate.unsqueeze(-1).to(dtype=landmark_xy.dtype)


class PromptIdentityAdapterV6(PromptIdentityAdapterV2):
    """v2 adapter with a spatial face gate on its velocity correction."""

    requires_spatial_gate = True

    def __init__(self, config: PromptIDAdapterV6Config):
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
        if spatial_gate is None:
            spatial_gate = landmark_face_gate(
                landmark_xy,
                base.shape[1],
                pad=self.config.face_gate_pad,
                temperature=self.config.face_gate_temperature,
                image_size=self.config.landmark_image_size,
            )
        if spatial_gate.shape[:2] != correction.shape[:2]:
            raise ValueError(
                f"spatial gate shape {tuple(spatial_gate.shape)} does not match "
                f"correction shape {tuple(correction.shape)}"
            )
        return correction * spatial_gate.to(device=correction.device, dtype=correction.dtype)

