"""Industrial Track: non-monotonic Physical Progress potential.

Source: R&D plan V3.2, Section 2.4. Progress is defined as "current physical
state relative to the success goal", not "how far along the episode timeline
we are" -- it must be able to fall during jam/slip and rise again during
recovery. w_* and sigma_* are pilot-calibration parameters, not fixed
constants; they live in configs/industrial_progress/*.yaml and are versioned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProgressNormalization:
    sigma_d: float
    sigma_xy: float
    sigma_rot: float
    d_goal: float


@dataclass(frozen=True)
class ProgressWeights:
    w_approach: float
    w_align: float
    w_depth: float
    w_contact: float
    w_jam: float
    w_slip: float
    w_overload: float


@dataclass(frozen=True)
class RawObservables:
    """Per-frame raw physical measurements, each shape (T,)."""

    d_tip: np.ndarray
    e_xy: np.ndarray
    e_rot: np.ndarray
    d_insert: np.ndarray
    q_contact: np.ndarray
    p_jam: np.ndarray
    p_slip: np.ndarray
    p_overload: np.ndarray

    def __post_init__(self):
        lengths = {len(v) for v in self.__dict__.values()}
        if len(lengths) != 1:
            raise ValueError(f"all observable arrays must share length, got {lengths}")


def normalized_subscores(obs: RawObservables, norm: ProgressNormalization) -> dict[str, np.ndarray]:
    g_approach = np.exp(-((obs.d_tip / norm.sigma_d) ** 2))
    g_align = np.exp(-((obs.e_xy / norm.sigma_xy) ** 2)) * np.exp(-((obs.e_rot / norm.sigma_rot) ** 2))
    g_depth = np.clip(obs.d_insert / norm.d_goal, 0.0, 1.0)
    g_contact = np.clip(obs.q_contact, 0.0, 1.0)
    return {"g_approach": g_approach, "g_align": g_align, "g_depth": g_depth, "g_contact": g_contact}


def raw_potential(obs: RawObservables, weights: ProgressWeights, norm: ProgressNormalization) -> np.ndarray:
    g = normalized_subscores(obs, norm)
    p_raw = (
        weights.w_approach * g["g_approach"]
        + weights.w_align * g["g_align"]
        + weights.w_depth * g["g_depth"]
        + weights.w_contact * g["g_contact"]
        - weights.w_jam * obs.p_jam
        - weights.w_slip * obs.p_slip
        - weights.w_overload * obs.p_overload
    )
    return np.clip(p_raw, 0.0, 1.0)


def temporal_filter(
    p_raw: np.ndarray,
    jam_or_overload_onset: np.ndarray,
    ema_alpha: float,
) -> np.ndarray:
    """Short EMA on normal frames; immediate (unsmoothed) application on
    jam/overload onset frames; no cumulative-max monotonic clamp anywhere, so
    recovery can raise P(t) back up after a dip.
    """
    p_smooth = np.empty_like(p_raw)
    p_smooth[0] = p_raw[0]
    for t in range(1, len(p_raw)):
        if jam_or_overload_onset[t]:
            p_smooth[t] = p_raw[t]
        else:
            p_smooth[t] = ema_alpha * p_raw[t] + (1.0 - ema_alpha) * p_smooth[t - 1]
    return p_smooth


def apply_terminal_overrides(
    p: np.ndarray,
    fully_seated_mask: np.ndarray | None = None,
    hard_abort_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = p.copy()
    if fully_seated_mask is not None:
        out = np.where(fully_seated_mask, 1.0, out)
    if hard_abort_mask is not None:
        out = np.where(hard_abort_mask, 0.0, out)
    return out


def physical_progress_sequence(
    obs: RawObservables,
    weights: ProgressWeights,
    norm: ProgressNormalization,
    ema_alpha: float = 0.3,
    jam_threshold: float = 0.5,
    overload_threshold: float = 0.5,
    fully_seated_mask: np.ndarray | None = None,
    hard_abort_mask: np.ndarray | None = None,
) -> np.ndarray:
    """End-to-end Physical Progress label per Section 2.4: raw potential ->
    temporal filtering (immediate on jam/overload onset) -> terminal override.
    """
    p_raw = raw_potential(obs, weights, norm)
    onset = (obs.p_jam >= jam_threshold) | (obs.p_overload >= overload_threshold)
    p_smooth = temporal_filter(p_raw, onset, ema_alpha)
    return apply_terminal_overrides(p_smooth, fully_seated_mask, hard_abort_mask)
