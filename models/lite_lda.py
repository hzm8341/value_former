"""Lite-LDA / Physical-grounded World Critic skeleton.

Source: R&D plan V3.2, Phase 6. Kept intentionally minimal: per Gate 6, this
model is only worth expanding once real AOP paired-ranking is validated on
hardware (Phase 4's Gate 4). Main prediction: (z_t, a_chunk) -> z_future,
with physical grounding heads mapping both z_t and z_future into an
interpretable physical-state space so "future latent" claims can be checked
against real geometry/contact/progress rather than only latent loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from labels.contact_labels import ContactState

N_CONTACT_STATES = len(ContactState)
PHYSICAL_STATE_DIM = 5  # [d_tip, e_xy, e_rot, d_insert, q_contact], matches industrial_progress.py observables


@dataclass(frozen=True)
class LiteLDAConfig:
    z_dim: int = 256
    action_dim: int = 8
    hidden_dim: int = 256
    physical_state_dim: int = PHYSICAL_STATE_DIM
    n_contact_states: int = N_CONTACT_STATES


class LiteLDA(nn.Module):
    def __init__(self, config: LiteLDAConfig = LiteLDAConfig()):
        super().__init__()
        self.config = config

        self.action_chunk_encoder = nn.GRU(config.action_dim, config.hidden_dim, batch_first=True)
        self.future_predictor = nn.Sequential(
            nn.Linear(config.z_dim + config.hidden_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.z_dim),
        )

        self.grounding_head = nn.Linear(config.z_dim, config.physical_state_dim)  # shared: z_t and z_future
        self.delta_head = nn.Linear(2 * config.z_dim, config.physical_state_dim)
        self.contact_future_head = nn.Linear(config.z_dim, config.n_contact_states)
        self.progress_future_head = nn.Linear(config.z_dim, 1)

    def forward(self, z_t: torch.Tensor, a_chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        """z_t: (B, z_dim). a_chunk: (B, chunk_len, action_dim)."""
        _, h_n = self.action_chunk_encoder(a_chunk)
        action_repr = h_n[-1]
        z_future = self.future_predictor(torch.cat([z_t, action_repr], dim=-1))

        physical_state_t = self.grounding_head(z_t)
        physical_state_future = self.grounding_head(z_future)
        delta_physical_state = self.delta_head(torch.cat([z_t, z_future], dim=-1))

        return {
            "z_future": z_future,
            "physical_state_t": physical_state_t,
            "physical_state_future": physical_state_future,
            "delta_physical_state": delta_physical_state,
            "delta_physical_state_consistency": physical_state_future - physical_state_t,
            "contact_future_logits": self.contact_future_head(z_future),
            "progress_future": torch.sigmoid(self.progress_future_head(z_future)).squeeze(-1),
        }
