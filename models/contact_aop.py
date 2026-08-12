"""Contact AOP: AOP V0 extended with a force/tactile history branch.

Source: R&D plan V3.2, Phase 5. Runs at the fast contact-rate branch,
separate from the low-frequency AOP/World Critic (Section 1.4's multi-rate
boundary). Outputs augment the base AOP heads with contact-side and
P(recoverable within H_r).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from labels.contact_labels import ContactState
from models.aop import AOPConfig, ActionOutcomePredictor

N_CONTACT_STATES = len(ContactState)
N_CONTACT_SIDES = 5  # left / right / front / back / centered


@dataclass(frozen=True)
class ContactAOPConfig:
    base: AOPConfig = AOPConfig()
    force_dim: int = 6
    tactile_dim: int = 16
    history_encoder_hidden: int = 64


class ContactActionOutcomePredictor(nn.Module):
    def __init__(self, config: ContactAOPConfig = ContactAOPConfig()):
        super().__init__()
        self.config = config
        self.base_aop = ActionOutcomePredictor(config.base)

        self.history_encoder = nn.GRU(
            input_size=config.force_dim + config.tactile_dim,
            hidden_size=config.history_encoder_hidden,
            batch_first=True,
        )
        extra_in = config.history_encoder_hidden
        self.contact_side_head = nn.Linear(config.base.hidden_dim + extra_in, N_CONTACT_SIDES)
        self.p_recoverable_head = nn.Linear(config.base.hidden_dim + extra_in, 1)
        self.history_proj = nn.Linear(extra_in, config.base.hidden_dim)

    def forward(
        self,
        state_repr: torch.Tensor,
        action: torch.Tensor,
        force_tactile_history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``force_tactile_history``: (B, history_len, force_dim+tactile_dim)."""
        base_out = self.base_aop(state_repr, action)
        _, h_n = self.history_encoder(force_tactile_history)
        history_repr = h_n[-1]  # (B, hidden)

        combined = torch.cat([base_out["trunk_repr"], history_repr], dim=-1)
        base_out["contact_side_logits"] = self.contact_side_head(combined)
        base_out["p_recoverable"] = torch.sigmoid(self.p_recoverable_head(combined)).squeeze(-1)
        return base_out
