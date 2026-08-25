"""Windowed dataset, feature caching, episode-level split, and the 4-group
balanced sampler used by the Paper Reproduction Track.

Source: ValueFormer v1 Section IV-D/V-B and Fig. 2 (Stage 1: sample every 15
frames -> DINOv3 CLS cache; Stage 2: sliding seq_len=16 window, 4-group
balanced sampler over {expert-success, expert-fail, rollout-success,
rollout-fail}, train/val split by episode with no frame leakage).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from data.synthetic import SOURCE_GROUPS, SyntheticEpisode
from labels.paper_labels import binary_mistake_labels, episode_mc_labels, MistakeInterval
from models.dinov3_encoder import DEFAULT_VIEWS, FrozenDinoV3Encoder


@torch.no_grad()
def encode_episode(
    episode: SyntheticEpisode,
    encoder: FrozenDinoV3Encoder,
    batch_size: int = 32,
) -> torch.Tensor:
    """Runs the frozen encoder over every frame of an episode. Returns (T, vision_dim)."""
    n = len(episode.sample_times)
    device = next(encoder.parameters()).device
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        views = {v: episode.images[v][start:end].to(device) for v in DEFAULT_VIEWS}
        chunks.append(encoder(views).cpu())
    return torch.cat(chunks, dim=0)


def episode_level_split(
    episodes: list[SyntheticEpisode],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[SyntheticEpisode], list[SyntheticEpisode]]:
    """Splits by whole episode, independently within each source group, so
    train/val never share frames from the same episode (Section 3.2 rule).
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    by_group: dict[str, list[SyntheticEpisode]] = {}
    for ep in episodes:
        by_group.setdefault(ep.source_group, []).append(ep)
    for group_eps in by_group.values():
        idx = rng.permutation(len(group_eps))
        n_val = max(1, int(round(val_fraction * len(group_eps)))) if len(group_eps) > 1 else 0
        val_idx = set(idx[:n_val].tolist())
        for i, ep in enumerate(group_eps):
            (val if i in val_idx else train).append(ep)
    return train, val


class ValueFormerWindowDataset(Dataset):
    """One sample = a causal window of frames ending at frame k, labeled with
    the Paper Track Eq.(2)-(4) targets for frame k.
    """

    def __init__(
        self,
        episodes: list[SyntheticEpisode],
        encoder: FrozenDinoV3Encoder,
        seq_len: int = 16,
        gamma: float = 0.99,
        n_stages: int = 6,
        fail_shape: str = "mc_smooth",
    ):
        self.seq_len = seq_len
        self.episodes = episodes
        self._vision_cache: list[torch.Tensor] = []
        self._vmc_cache: list[np.ndarray] = []
        self._vbin_cache: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []

        for ep_idx, ep in enumerate(episodes):
            vision = encode_episode(ep, encoder)
            self._vision_cache.append(vision)

            v_mc = episode_mc_labels(
                n=len(ep.sample_times),
                success=ep.success,
                gamma=gamma,
                s_fail=ep.s_fail,
                n_stages=n_stages,
                shape=fail_shape,
            )
            self._vmc_cache.append(v_mc)

            intervals = [MistakeInterval(t0, t1) for t0, t1 in ep.mistake_intervals]
            v_bin = binary_mistake_labels(ep.sample_times, intervals)
            self._vbin_cache.append(v_bin)

            for frame_idx in range(len(ep.sample_times)):
                self._index.append((ep_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def group_ids(self) -> list[str]:
        return [self.episodes[ep_idx].source_group for ep_idx, _ in self._index]

    def _window(self, arr: torch.Tensor, frame_idx: int) -> torch.Tensor:
        start = frame_idx - self.seq_len + 1
        if start >= 0:
            return arr[start : frame_idx + 1]
        pad = arr[0:1].repeat(-start, *([1] * (arr.dim() - 1)))
        return torch.cat([pad, arr[: frame_idx + 1]], dim=0)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_idx, frame_idx = self._index[i]
        ep = self.episodes[ep_idx]

        vision_window = self._window(self._vision_cache[ep_idx], frame_idx)
        state_window = self._window(ep.joint_state, frame_idx)
        time_window = self._window(ep.time_feat, frame_idx)

        return {
            "vision": vision_window,
            "state": state_window,
            "time_feat": time_window,
            "v_mc": torch.tensor(self._vmc_cache[ep_idx][frame_idx], dtype=torch.float32),
            "v_bin": torch.tensor(self._vbin_cache[ep_idx][frame_idx], dtype=torch.float32),
            "success": torch.tensor(ep.success, dtype=torch.bool),
            "group": ep.source_group,
        }


def build_balanced_sampler(
    group_ids: list[str],
    groups: tuple[str, ...] = SOURCE_GROUPS,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Equalizes the expected weight of each present data-source group,
    regardless of its absolute episode/frame count; groups with zero samples
    (e.g. expert-fail, typically empty) are skipped, matching Section V-A.
    """
    counts = Counter(group_ids)
    present = [g for g in groups if counts.get(g, 0) > 0]
    if not present:
        raise ValueError("no samples to build a sampler from")
    per_group_weight = {g: 1.0 / (counts[g] * len(present)) for g in present}
    weights = [per_group_weight[g] for g in group_ids]
    n = num_samples if num_samples is not None else len(group_ids)
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)
