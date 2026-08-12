"""Canonical Action Representation. Source: R&D plan V3.2, Section 4.1.

AOP does not consume the native action format of any specific policy family
(ACT / pi0 / DP / rule controller) directly. Every candidate is resampled
into this EE-centric physical representation before scoring, and chunk
duration/mask is kept explicit so the model cannot use chunk length as a
proxy for policy identity (relevant for the Phase 3B cross-policy gate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANONICAL_DIM = 8  # [dx, dy, dz, drx, dry, drz, gripper_state, dt]


@dataclass(frozen=True)
class CanonicalAction:
    dx: float
    dy: float
    dz: float
    drx: float
    dry: float
    drz: float
    gripper_state: float          # in [0, 1], 0=open, 1=closed
    dt: float                        # seconds this action step spans
    target_force: float | None = None
    impedance_gain: float | None = None
    primitive_id: str | None = None

    def to_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz, self.drx, self.dry, self.drz, self.gripper_state, self.dt], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "CanonicalAction":
        if arr.shape[-1] != CANONICAL_DIM:
            raise ValueError(f"expected last dim {CANONICAL_DIM}, got {arr.shape[-1]}")
        dx, dy, dz, drx, dry, drz, gripper, dt = arr.tolist()
        return CanonicalAction(dx, dy, dz, drx, dry, drz, gripper, dt)


def resample_chunk(
    raw_actions: np.ndarray,
    raw_dt: float,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resamples a (T_raw, 6+) delta-action chunk sampled at raw_dt to a
    uniform target_dt grid via linear interpolation on the cumulative pose
    delta, and returns an explicit validity mask so callers never confuse
    chunk length with policy identity (Section 4.1's "explicit mask/duration").

    Returns (resampled (T_new, 6), mask (T_new,) all-True for this chunk).
    """
    if raw_actions.ndim != 2:
        raise ValueError("raw_actions must be (T_raw, D)")
    t_raw = np.arange(len(raw_actions)) * raw_dt
    total_duration = t_raw[-1] if len(t_raw) > 1 else raw_dt
    n_new = max(int(round(total_duration / target_dt)), 1)
    t_new = np.arange(n_new) * target_dt

    cumulative = np.cumsum(raw_actions, axis=0)
    resampled_cumulative = np.stack(
        [np.interp(t_new, t_raw, cumulative[:, d]) for d in range(raw_actions.shape[1])], axis=-1
    )
    resampled_deltas = np.diff(resampled_cumulative, axis=0, prepend=np.zeros((1, raw_actions.shape[1])))
    mask = np.ones(n_new, dtype=bool)
    return resampled_deltas.astype(np.float32), mask
