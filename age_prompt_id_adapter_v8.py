"""Identity residual adapter with FiLM and a learnable spatial gate.

The aging velocity remains the main route.  This module adds a separate
identity-conditioned residual route instead of expanding one global condition
over every token and relying only on a fixed landmark gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from age_prompt_id_adapter_v2 import normalize_landmarks
from age_prompt_id_adapter_v7 import landmark_structure_gate


@dataclass
class PromptIDAdapterV8Config:
    velocity_dim: int = 64
    hidden_dim: int = 128
    condition_dim: int = 64
    identity_dim: int = 512
    landmark_points: int = 478
    max_correction: float = 0.1
    face_gate_pad: float = 0.12
    face_gate_temperature: float = 0.08
    landmark_image_size: float = 512.0
    structure_sigma: float = 0.16
    structure_floor: float = 0.05
    orthogonal_eps: float = 1e-6
    identity_gain_init: float = 1.0
    learned_gate_floor: float = 0.5


class PromptIdentityAdapterV8(nn.Module):
    """Separate identity residual branch with FiLM and learned routing."""

    requires_spatial_gate = True
    requires_structure_gate = True

    def __init__(self, config: PromptIDAdapterV8Config):
        super().__init__()
        self.config = config
        landmark_dim = config.landmark_points * 2
        hidden = config.hidden_dim

        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.condition_dim),
        )
        self.identity_encoder = nn.Sequential(
            nn.Linear(config.identity_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(config.condition_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(config.velocity_dim * 2 + 1, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        # Zero initialization preserves the original aging route at startup.
        self.film = nn.Linear(hidden, hidden * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.identity_fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.correction_head = nn.Linear(hidden, config.velocity_dim)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

        self.learned_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.learned_gate[-1].weight)
        nn.init.zeros_(self.learned_gate[-1].bias)

        initial_gain = max(float(config.identity_gain_init), 1e-4)
        self.log_identity_gain = nn.Parameter(torch.tensor(initial_gain).log())

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
        if landmark_xy.ndim == 2:
            landmark_xy = landmark_xy.reshape(landmark_xy.shape[0], -1, 2)
        if landmark_xy.ndim != 3 or landmark_xy.shape[-1] != 2:
            raise ValueError(f"expected [B, P, 2] landmarks, got {tuple(landmark_xy.shape)}")
        if identity_embedding is None:
            identity_embedding = base.new_zeros((base.shape[0], self.config.identity_dim))
        identity_embedding = identity_embedding.float()
        identity_embedding = identity_embedding / identity_embedding.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        landmark = normalize_landmarks(landmark_xy).flatten(1)
        landmark_condition = self.landmark_encoder(landmark)
        scalar_condition = torch.stack(
            (
                source_age.reshape(-1) / 100.0,
                target_age.reshape(-1) / 100.0,
                (target_age.reshape(-1) - source_age.reshape(-1)) / 100.0,
            ),
            dim=-1,
        )
        route = self.route_encoder(torch.cat((landmark_condition, scalar_condition), dim=-1))
        identity_condition = self.identity_encoder(identity_embedding)

        token_input = torch.cat(
            (base, age - base, timestep.reshape(-1, 1, 1).expand(-1, base.shape[1], 1)),
            dim=-1,
        )
        token = self.token_encoder(token_input)
        route = route.unsqueeze(1).expand(-1, base.shape[1], -1)
        identity_tokens = identity_condition.unsqueeze(1).expand(-1, base.shape[1], -1)

        # FiLM lets the identity condition modulate each token without changing
        # the aging velocity branch itself.
        gamma, beta = self.film(identity_condition).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        modulated = (token + route) * (1.0 + gamma) + beta
        fused = self.identity_fuse(torch.cat((modulated, identity_tokens, route), dim=-1))

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
        spatial_gate = spatial_gate.to(device=base.device, dtype=fused.dtype)
        learned_gate = torch.sigmoid(
            self.learned_gate(torch.cat((fused, identity_tokens, route), dim=-1))
        )
        final_gate = spatial_gate * (
            float(self.config.learned_gate_floor) + learned_gate
        )
        gain = self.log_identity_gain.exp().clamp(0.05, 8.0)
        correction = torch.tanh(self.correction_head(fused))
        return correction * float(self.config.max_correction) * gain * final_gate

