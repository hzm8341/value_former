"""Training objective. Source: ValueFormer v1 Section IV-E, Eq. (6)-(7).

L = L_mc + beta * L_bin
L_mc  = soft-target weighted BCE on the continuous MC label (Eq. 6)
L_bin = BCE with a batch-wise pos_weight alpha_+ on the binary mistake label (Eq. 7)
beta = 1.0
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mc_loss(v_mc_logit: torch.Tensor, v_mc_target: torch.Tensor, w_fail: float = 1.0) -> torch.Tensor:
    """Eq. (6): w_i = 1 + (w_fail - 1) * 1{v_i == 0} optionally up-weights
    flat-zero frames; w_fail=1.0 is a no-op (uniform weighting).
    """
    weight = 1.0 + (w_fail - 1.0) * (v_mc_target == 0).float()
    return F.binary_cross_entropy_with_logits(v_mc_logit, v_mc_target, weight=weight, reduction="mean")


def batch_pos_weight(v_bin_target: torch.Tensor, clip_range: tuple[float, float] = (0.1, 10.0)) -> torch.Tensor:
    """Eq. (7): alpha_+ = clip(n0/n1, lo, hi), n0 = mistake-frame count
    (v_bin=0), n1 = good-frame count (v_bin=1) in the current batch.
    """
    n1 = (v_bin_target == 1).float().sum()
    n0 = (v_bin_target == 0).float().sum()
    n1 = torch.clamp(n1, min=1.0)
    alpha_plus = torch.clamp(n0 / n1, min=clip_range[0], max=clip_range[1])
    return alpha_plus


def bin_loss(v_bin_logit: torch.Tensor, v_bin_target: torch.Tensor, clip_range: tuple[float, float] = (0.1, 10.0)) -> torch.Tensor:
    alpha_plus = batch_pos_weight(v_bin_target, clip_range)
    return F.binary_cross_entropy_with_logits(v_bin_logit, v_bin_target, pos_weight=alpha_plus, reduction="mean")


def combined_loss(
    v_mc_logit: torch.Tensor,
    v_mc_target: torch.Tensor,
    v_bin_logit: torch.Tensor,
    v_bin_target: torch.Tensor,
    beta: float = 1.0,
    w_fail: float = 1.0,
    alpha_clip: tuple[float, float] = (0.1, 10.0),
) -> dict[str, torch.Tensor]:
    l_mc = mc_loss(v_mc_logit, v_mc_target, w_fail=w_fail)
    l_bin = bin_loss(v_bin_logit, v_bin_target, clip_range=alpha_clip)
    total = l_mc + beta * l_bin
    return {"loss": total, "l_mc": l_mc, "l_bin": l_bin}
