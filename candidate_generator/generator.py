"""Candidate Action Generator. Source: R&D plan V3.2, Section 4.2.

Expands the nominal VLA action into a small set of physically meaningful
alternatives for AOP to rank. Generation and evaluation are kept as separate
problems: this module does not judge candidates, it only proposes them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from policy_adapters.canonical_action import CanonicalAction


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    action: CanonicalAction
    origin: str  # nominal | geometry_correction | contact_correction | withdraw | perturbation


class CandidateActionGenerator:
    def __init__(
        self,
        n_perturbations: int = 4,
        max_cartesian_step_m: float = 0.01,
        max_rotation_step_rad: float = 0.1,
        withdraw_distance_m: float = 0.02,
        seed: int = 0,
    ):
        self.n_perturbations = n_perturbations
        self.max_cartesian_step_m = max_cartesian_step_m
        self.max_rotation_step_rad = max_rotation_step_rad
        self.withdraw_distance_m = withdraw_distance_m
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        nominal: CanonicalAction,
        e_xy: tuple[float, float] | None = None,
        e_rot: float | None = None,
        contact_side: str | None = None,
    ) -> list[Candidate]:
        candidates = [Candidate("cand_0_nominal", nominal, "nominal")]

        if e_xy is not None:
            ex, ey = e_xy
            geo = replace(nominal, dx=nominal.dx - ex, dy=nominal.dy - ey)
            candidates.append(Candidate("cand_1_geometry", geo, "geometry_correction"))

        if contact_side is not None:
            side_shift = {
                "left": (0.0, self.max_cartesian_step_m),
                "right": (0.0, -self.max_cartesian_step_m),
                "front": (self.max_cartesian_step_m, 0.0),
                "back": (-self.max_cartesian_step_m, 0.0),
            }.get(contact_side, (0.0, 0.0))
            contact = replace(nominal, dx=nominal.dx + side_shift[0], dy=nominal.dy + side_shift[1])
            candidates.append(Candidate("cand_2_contact_side", contact, "contact_correction"))

        withdraw = replace(nominal, dz=nominal.dz + self.withdraw_distance_m, gripper_state=nominal.gripper_state)
        candidates.append(Candidate("cand_3_withdraw", withdraw, "withdraw"))

        for i in range(self.n_perturbations):
            noise_xyz = self._rng.uniform(-self.max_cartesian_step_m, self.max_cartesian_step_m, size=3)
            noise_rot = self._rng.uniform(-self.max_rotation_step_rad, self.max_rotation_step_rad, size=3)
            perturbed = replace(
                nominal,
                dx=nominal.dx + noise_xyz[0], dy=nominal.dy + noise_xyz[1], dz=nominal.dz + noise_xyz[2],
                drx=nominal.drx + noise_rot[0], dry=nominal.dry + noise_rot[1], drz=nominal.drz + noise_rot[2],
            )
            candidates.append(Candidate(f"cand_{4 + i}_perturbation", perturbed, "perturbation"))

        return candidates
