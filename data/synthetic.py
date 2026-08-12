"""Synthetic LeRobot-v2-like episode generator.

No real robot dataset is available in this environment. This module
generates synthetic sandwich-assembly-style episodes that reproduce the four
canonical rollout signatures from ValueFormer v1 Fig. 10 (clean success,
success-with-retry, early-collapse, stuck-scratching) plus matching
industrial physical observables, purely so the label engine, dataset,
model, and training loop can be exercised end-to-end offline. It is not a
substitute for real robot data and must never be used to report a paper or
product result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from models.dinov3_encoder import DEFAULT_VIEWS

ROLLOUT_SIGNATURES = ("clean_success", "success_retry", "early_collapse", "stuck_scratching")
SOURCE_GROUPS = ("expert-success", "expert-fail", "rollout-success", "rollout-fail")


@dataclass
class SyntheticEpisode:
    episode_id: str
    source_group: str
    success: bool
    rollout_signature: str
    n_stages: int
    s_fail: int | None
    sample_times: np.ndarray                       # (T,) seconds, 0.5s stride
    images: dict[str, torch.Tensor]                  # view -> (T,3,H,W)
    joint_state: torch.Tensor                          # (T,14)
    time_feat: torch.Tensor                              # (T,2) [t_sec, t_frac]
    mistake_intervals: list[tuple[float, float]]
    physical: dict[str, np.ndarray] = field(default_factory=dict)  # industrial observables


def _duration_for_signature(rng: np.random.Generator, signature: str) -> float:
    if signature == "clean_success":
        return rng.uniform(50, 75)
    if signature == "success_retry":
        return rng.uniform(60, 85)
    if signature == "early_collapse":
        return rng.uniform(20, 40)
    if signature == "stuck_scratching":
        return rng.uniform(30, 50)
    raise ValueError(signature)


def _physical_trace(rng: np.random.Generator, t_norm: np.ndarray, signature: str) -> dict[str, np.ndarray]:
    """t_norm in [0,1] episode-relative time. Builds a physically plausible
    approach/align/depth/contact trace consistent with the signature, for the
    Industrial Track labels (independent of the Paper Track labels below).
    """
    n = len(t_norm)
    d_tip = np.clip(0.25 * (1 - t_norm) + rng.normal(0, 0.01, n), 0.0, None)
    e_xy = np.abs(rng.normal(0.002, 0.001, n))
    e_rot = np.abs(rng.normal(0.02, 0.01, n))
    d_insert = np.clip(0.03 * t_norm + rng.normal(0, 0.001, n), 0.0, 0.03)
    q_contact = np.clip(t_norm + rng.normal(0, 0.05, n), 0.0, 1.0)
    p_jam = np.zeros(n)
    p_slip = np.zeros(n)
    p_overload = np.zeros(n)

    if signature == "success_retry":
        lo, hi = int(0.4 * n), int(0.55 * n)
        p_slip[lo:hi] = rng.uniform(0.6, 0.9, hi - lo)
        d_insert[lo:hi] *= 0.5
    elif signature == "early_collapse":
        cut = int(0.5 * n)
        p_jam[cut:] = rng.uniform(0.6, 0.95, n - cut)
        d_insert[cut:] = d_insert[cut] * 0.6
        q_contact[cut:] *= 0.5
    elif signature == "stuck_scratching":
        cut = int(0.3 * n)
        d_insert[cut:] = d_insert[cut]
        q_contact[cut:] = np.clip(q_contact[cut] + rng.normal(0, 0.05, n - cut), 0, 1)
        p_slip[cut:] = rng.uniform(0.2, 0.5, n - cut)

    return {
        "d_tip": d_tip, "e_xy": e_xy, "e_rot": e_rot, "d_insert": d_insert,
        "q_contact": q_contact, "p_jam": p_jam, "p_slip": p_slip, "p_overload": p_overload,
    }


def generate_synthetic_episode(
    rng: np.random.Generator,
    episode_id: str,
    source_group: str,
    rollout_signature: str,
    n_stages: int = 6,
    stride_sec: float = 0.5,
    image_size: int = 32,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> SyntheticEpisode:
    if rollout_signature not in ROLLOUT_SIGNATURES:
        raise ValueError(rollout_signature)
    success = rollout_signature == "clean_success" or (
        rollout_signature == "success_retry"
    )

    duration = _duration_for_signature(rng, rollout_signature)
    n = max(int(duration / stride_sec), 4)
    sample_times = np.arange(n) * stride_sec
    t_norm = sample_times / sample_times[-1]

    s_fail = None
    mistake_intervals: list[tuple[float, float]] = []
    if rollout_signature == "success_retry":
        lo, hi = sample_times[int(0.4 * n)], sample_times[int(0.55 * n) - 1]
        mistake_intervals = [(float(lo), float(hi))]
    elif rollout_signature == "early_collapse":
        s_fail = rng.integers(2, 4)  # stopped partway through the ordered stages
        mistake_intervals = [(float(sample_times[int(0.5 * n)]), float(sample_times[-1]))]
    elif rollout_signature == "stuck_scratching":
        s_fail = 1
        mistake_intervals = [(float(sample_times[int(0.3 * n)]), float(sample_times[-1]))]

    images = {
        v: torch.rand(n, 3, image_size, image_size) * 0.2
        + torch.linspace(0, 1, n).view(n, 1, 1, 1) * 0.1
        for v in views
    }
    joint_state = torch.from_numpy(rng.normal(0, 0.1, size=(n, 14)).astype(np.float32))
    time_feat = torch.from_numpy(
        np.stack([sample_times, t_norm], axis=-1).astype(np.float32)
    )

    physical = _physical_trace(rng, t_norm, rollout_signature)

    return SyntheticEpisode(
        episode_id=episode_id,
        source_group=source_group,
        success=success,
        rollout_signature=rollout_signature,
        n_stages=n_stages,
        s_fail=int(s_fail) if s_fail is not None else None,
        sample_times=sample_times,
        images=images,
        joint_state=joint_state,
        time_feat=time_feat,
        mistake_intervals=mistake_intervals,
        physical=physical,
    )


def generate_synthetic_dataset(
    n_expert_success: int = 20,
    n_rollout_success: int = 8,
    n_rollout_fail: int = 8,
    seed: int = 0,
    **episode_kwargs,
) -> list[SyntheticEpisode]:
    """expert-fail is intentionally left empty by default, matching the
    paper's observation that expert teleoperators rarely fail (Section V-A).
    """
    rng = np.random.default_rng(seed)
    episodes: list[SyntheticEpisode] = []

    for i in range(n_expert_success):
        episodes.append(
            generate_synthetic_episode(rng, f"expert_success_{i:04d}", "expert-success", "clean_success", **episode_kwargs)
        )
    for i in range(n_rollout_success):
        sig = "clean_success" if i % 2 == 0 else "success_retry"
        episodes.append(
            generate_synthetic_episode(rng, f"rollout_success_{i:04d}", "rollout-success", sig, **episode_kwargs)
        )
    for i in range(n_rollout_fail):
        sig = "early_collapse" if i % 2 == 0 else "stuck_scratching"
        episodes.append(
            generate_synthetic_episode(rng, f"rollout_fail_{i:04d}", "rollout-fail", sig, **episode_kwargs)
        )
    return episodes
