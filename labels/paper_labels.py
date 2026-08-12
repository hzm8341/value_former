"""Paper Reproduction Track labels.

Implements ValueFormer v1 (arXiv:2608.02958v1) Section IV-A Eq. (2)-(4):
stage-aware, success-then-decay Monte Carlo value labels for V_mc, and
segment-based mistake-interval binary labels for V_bin. Also implements the
four alternative fail-episode label shapes from Section V (outcome-scaled,
cliff, C-linear, C-late) used for the label-shape ablation.

This module must stay a faithful reproduction of the paper's label
definitions. Non-monotonic, physically-grounded industrial progress lives in
labels/industrial_progress.py instead -- do not mix the two here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

FAIL_SHAPES = ("mc_smooth", "outcome_scaled", "cliff", "c_linear", "c_late")


def success_mc_labels(n: int, gamma: float) -> np.ndarray:
    """Eq. (2): v_k^succ = gamma^(N-1-k), k = 0..N-1."""
    k = np.arange(n)
    return gamma ** (n - 1 - k)


def _completed_stage_fraction(s_fail: int, n_stages: int) -> float:
    if not (1 <= s_fail <= n_stages):
        raise ValueError(f"s_fail must be in [1, {n_stages}], got {s_fail}")
    return s_fail / n_stages


def failed_mc_labels(
    n: int,
    s_fail: int,
    n_stages: int,
    gamma: float,
    shape: str = "mc_smooth",
) -> np.ndarray:
    """Per-frame MC value label for a failed episode under the given label shape.

    ``shape="mc_smooth"`` is Eq. (3), the scheme used for all headline results.
    The other four shapes reproduce the Section V ablation and are kept only
    for that comparison, never as the default training target.
    """
    if shape not in FAIL_SHAPES:
        raise ValueError(f"unknown fail label shape {shape!r}, expected one of {FAIL_SHAPES}")

    v_succ = success_mc_labels(n, gamma)
    c = _completed_stage_fraction(s_fail, n_stages)
    k_fail = int(np.floor(c * n))
    k_fail = min(max(k_fail, 0), n - 1)
    k = np.arange(n)

    if shape == "mc_smooth":
        v_kfail = gamma ** (n - 1 - k_fail)
        pre = gamma ** (n - 1 - k)
        post = v_kfail * gamma ** (k - k_fail)
        return np.where(k <= k_fail, pre, post)

    if shape == "outcome_scaled":
        v = c * v_succ
        v = np.where(k < k_fail, v, 0.0)
        return v

    if shape == "cliff":
        return np.where(k < k_fail, v_succ, 0.0)

    if shape == "c_linear":
        k_fail_safe = max(k_fail, 1)
        alpha = np.clip(k / k_fail_safe, 0.0, 1.0)
        v = (1.0 - alpha) * v_succ + alpha * c
        return np.where(k < k_fail, v, 0.0)

    if shape == "c_late":
        rho = 0.25
        k_start = int(np.floor((1.0 - rho) * k_fail))
        k_start = min(max(k_start, 0), k_fail)
        v_start = v_succ[k_start] if n else 0.0
        span = max(k_fail - k_start, 1)
        ramp_alpha = np.clip((k - k_start) / span, 0.0, 1.0)
        ramped = (1.0 - ramp_alpha) * v_start + ramp_alpha * c
        v = np.where(k < k_start, v_succ, ramped)
        return np.where(k < k_fail, v, 0.0)

    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class MistakeInterval:
    t_start: float
    t_end: float


def binary_mistake_labels(
    sample_times: Sequence[float],
    mistake_intervals: Sequence[MistakeInterval],
) -> np.ndarray:
    """Eq. (4): v_bin(k) = 0 if sample k falls inside any annotated mistake
    interval [t_start, t_end), else 1.
    """
    t = np.asarray(sample_times, dtype=np.float64)
    v_bin = np.ones_like(t)
    for interval in mistake_intervals:
        inside = (t >= interval.t_start) & (t < interval.t_end)
        v_bin = np.where(inside, 0.0, v_bin)
    return v_bin


def episode_mc_labels(
    n: int,
    success: bool,
    gamma: float,
    s_fail: int | None = None,
    n_stages: int | None = None,
    shape: str = "mc_smooth",
) -> np.ndarray:
    """Convenience wrapper dispatching to success/failed label functions."""
    if success:
        return success_mc_labels(n, gamma)
    if s_fail is None or n_stages is None:
        raise ValueError("s_fail and n_stages are required for a failed episode")
    return failed_mc_labels(n, s_fail, n_stages, gamma, shape=shape)
