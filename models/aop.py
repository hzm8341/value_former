"""Action Outcome Predictor V0: Q_H(s,a). Source: R&D plan V3.2, Section 1.3
and 4.1-4.4, Phase 4.

Rather than a single hard-to-interpret scalar Q, AOP V0 emits multi-head
short-horizon outcomes over a fixed horizon H, which are then combined into
a ranking score. Consumes the ValueFormer state representation (state_repr,
the causal-transformer last-token output) plus a CanonicalAction -- it does
not re-encode raw observations itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from labels.contact_labels import ContactState

N_CONTACT_STATES = len(ContactState)


@dataclass(frozen=True)
class AOPConfig:
    state_repr_dim: int = 256
    action_dim: int = 8
    hidden_dim: int = 256
    n_contact_states: int = N_CONTACT_STATES


@dataclass(frozen=True)
class RankingWeights:
    w_progress: float = 1.0
    w_jam: float = 1.0
    w_slip: float = 0.5
    w_success: float = 1.0


class ActionOutcomePredictor(nn.Module):
    def __init__(self, config: AOPConfig = AOPConfig()):
        super().__init__()
        self.config = config
        in_dim = config.state_repr_dim + config.action_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
        )
        self.delta_progress_head = nn.Linear(config.hidden_dim, 1)
        self.p_jam_head = nn.Linear(config.hidden_dim, 1)
        self.p_slip_head = nn.Linear(config.hidden_dim, 1)
        self.p_success_h_head = nn.Linear(config.hidden_dim, 1)
        self.delta_ee_pose_head = nn.Linear(config.hidden_dim, 6)
        self.delta_insertion_depth_head = nn.Linear(config.hidden_dim, 1)
        self.contact_transition_head = nn.Linear(config.hidden_dim, config.n_contact_states)

    def forward(self, state_repr: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(torch.cat([state_repr, action], dim=-1))
        return {
            "delta_progress": self.delta_progress_head(h).squeeze(-1),
            "p_jam": torch.sigmoid(self.p_jam_head(h)).squeeze(-1),
            "p_slip": torch.sigmoid(self.p_slip_head(h)).squeeze(-1),
            "p_success_h": torch.sigmoid(self.p_success_h_head(h)).squeeze(-1),
            "delta_ee_pose": self.delta_ee_pose_head(h),
            "delta_insertion_depth": self.delta_insertion_depth_head(h).squeeze(-1),
            "contact_transition_logits": self.contact_transition_head(h),
            "trunk_repr": h,
        }

    @staticmethod
    def ranking_score(outputs: dict[str, torch.Tensor], weights: RankingWeights = RankingWeights()) -> torch.Tensor:
        """Score = w1*ΔProgress - w2*P(Jam) - w3*P(Slip) + w4*P(Success_H), Section 1.3."""
        return (
            weights.w_progress * outputs["delta_progress"]
            - weights.w_jam * outputs["p_jam"]
            - weights.w_slip * outputs["p_slip"]
            + weights.w_success * outputs["p_success_h"]
        )
