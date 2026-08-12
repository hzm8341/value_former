"""Deterministic Candidate Safety Envelope. Source: R&D plan V3.2, Section 4.3.

Runs before any candidate reaches AOP. AOP is a learned ranker and must never
be the sole source of safety -- this filter is deterministic and rule-based,
and an unsafe candidate is rejected here regardless of how the ranker would
have scored it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from candidate_generator.generator import Candidate


@dataclass(frozen=True)
class SafetyLimits:
    workspace_xyz_min: tuple[float, float, float]
    workspace_xyz_max: tuple[float, float, float]
    max_cartesian_step_m: float
    max_rotation_step_rad: float
    forbidden_force_directions: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    forbidden_direction_cos_threshold: float = 0.9


@dataclass(frozen=True)
class SafetyDecision:
    candidate_id: str
    accepted: bool
    reasons: tuple[str, ...]


class SafetyEnvelope:
    def __init__(self, limits: SafetyLimits):
        self.limits = limits

    def _check_one(
        self,
        candidate: Candidate,
        current_ee_pos: np.ndarray,
        jam_active: bool,
        overload_active: bool,
    ) -> SafetyDecision:
        reasons: list[str] = []
        a = candidate.action

        if jam_active or overload_active:
            if candidate.origin != "withdraw":
                reasons.append("jam_or_overload_active_only_withdraw_allowed")

        next_pos = current_ee_pos + np.array([a.dx, a.dy, a.dz])
        lo = np.array(self.limits.workspace_xyz_min)
        hi = np.array(self.limits.workspace_xyz_max)
        if np.any(next_pos < lo) or np.any(next_pos > hi):
            reasons.append("workspace_limit_violation")

        cart_step = float(np.linalg.norm([a.dx, a.dy, a.dz]))
        if cart_step > self.limits.max_cartesian_step_m:
            reasons.append("max_cartesian_step_exceeded")

        rot_step = max(abs(a.drx), abs(a.dry), abs(a.drz))
        if rot_step > self.limits.max_rotation_step_rad:
            reasons.append("max_rotation_step_exceeded")

        if self.limits.forbidden_force_directions and cart_step > 1e-9:
            direction = np.array([a.dx, a.dy, a.dz]) / cart_step
            for forbidden in self.limits.forbidden_force_directions:
                forbidden = np.array(forbidden)
                denom = np.linalg.norm(forbidden)
                if denom < 1e-9:
                    continue
                cos_sim = float(np.dot(direction, forbidden / denom))
                if cos_sim > self.limits.forbidden_direction_cos_threshold:
                    reasons.append("forbidden_force_direction")
                    break

        return SafetyDecision(candidate.candidate_id, accepted=len(reasons) == 0, reasons=tuple(reasons))

    def filter(
        self,
        candidates: list[Candidate],
        current_ee_pos: np.ndarray,
        jam_active: bool = False,
        overload_active: bool = False,
    ) -> tuple[list[Candidate], list[SafetyDecision]]:
        decisions = [self._check_one(c, current_ee_pos, jam_active, overload_active) for c in candidates]
        accepted = [c for c, d in zip(candidates, decisions) if d.accepted]
        return accepted, decisions
