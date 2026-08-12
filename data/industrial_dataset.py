"""Industrial Track windowed dataset: non-monotonic Physical Progress
regression target + contact-mistake binary target, built independently from
the Paper Track's stage-aware MC labels (Section 2.1: the two tracks must
never share labels or conclusions).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import encode_episode
from data.synthetic import SyntheticEpisode
from labels.contact_labels import build_mistake_segments
from labels.industrial_progress import (
    ProgressNormalization,
    ProgressWeights,
    RawObservables,
    physical_progress_sequence,
)
from labels.paper_labels import MistakeInterval, binary_mistake_labels
from models.dinov3_encoder import FrozenDinoV3Encoder

MISTAKE_ONSET_THRESHOLD = 0.5


def _raw_observables_from_episode(ep: SyntheticEpisode) -> RawObservables:
    p = ep.physical
    return RawObservables(
        d_tip=p["d_tip"], e_xy=p["e_xy"], e_rot=p["e_rot"], d_insert=p["d_insert"],
        q_contact=p["q_contact"], p_jam=p["p_jam"], p_slip=p["p_slip"], p_overload=p["p_overload"],
    )


def industrial_progress_labels(ep: SyntheticEpisode, weights: ProgressWeights, norm: ProgressNormalization, ema_alpha: float) -> np.ndarray:
    obs = _raw_observables_from_episode(ep)
    fully_seated = np.zeros(len(ep.sample_times), dtype=bool)
    fully_seated[-1] = ep.success
    return physical_progress_sequence(obs, weights, norm, ema_alpha=ema_alpha, fully_seated_mask=fully_seated)


def industrial_mistake_labels(ep: SyntheticEpisode) -> np.ndarray:
    p = ep.physical
    in_mistake = (p["p_jam"] >= MISTAKE_ONSET_THRESHOLD) | (p["p_slip"] >= MISTAKE_ONSET_THRESHOLD) | (p["p_overload"] >= MISTAKE_ONSET_THRESHOLD)
    recovered = np.zeros(len(in_mistake), dtype=bool)
    segments = build_mistake_segments(ep.sample_times.tolist(), in_mistake.tolist(), recovered.tolist())
    intervals = [MistakeInterval(seg.t_start, seg.t_end) for seg in segments]
    return binary_mistake_labels(ep.sample_times, intervals)


class IndustrialWindowDataset(Dataset):
    def __init__(
        self,
        episodes: list[SyntheticEpisode],
        encoder: FrozenDinoV3Encoder,
        seq_len: int,
        weights: ProgressWeights,
        norm: ProgressNormalization,
        ema_alpha: float = 0.3,
    ):
        self.seq_len = seq_len
        self.episodes = episodes
        self._vision_cache: list[torch.Tensor] = []
        self._progress_cache: list[np.ndarray] = []
        self._mistake_cache: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []

        for ep_idx, ep in enumerate(episodes):
            self._vision_cache.append(encode_episode(ep, encoder))
            self._progress_cache.append(industrial_progress_labels(ep, weights, norm, ema_alpha))
            self._mistake_cache.append(industrial_mistake_labels(ep))
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
        return {
            "vision": self._window(self._vision_cache[ep_idx], frame_idx),
            "state": self._window(ep.joint_state, frame_idx),
            "time_feat": self._window(ep.time_feat, frame_idx),
            "progress": torch.tensor(self._progress_cache[ep_idx][frame_idx], dtype=torch.float32),
            "mistake_ok": torch.tensor(self._mistake_cache[ep_idx][frame_idx], dtype=torch.float32),
            "success": torch.tensor(ep.success, dtype=torch.bool),
            "group": ep.source_group,
        }
