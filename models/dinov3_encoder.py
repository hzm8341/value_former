"""Frozen visual frontend: six-view DINOv3 ViT-L/16 CLS encoding.

Source: ValueFormer v1 Section IV-D, Fig. 6-7. Three full camera views
(cam_high, cam_left_wrist, cam_right_wrist) plus three ROI crops are each
encoded independently by the same frozen backbone; CLS tokens are
concatenated to a 6*1024 = 6144-d per-frame descriptor.

DINOv3 weights are not available offline in this environment. We first try
`torch.hub` (real backbone, matches the paper exactly when network/weights
are available); if that fails we fall back to a deterministic, seeded,
frozen random-projection encoder with the same output shape so the rest of
the pipeline (labels, model, training) is fully testable offline. The
fallback is clearly named and must not be presented as the paper's DINOv3
features.
"""
from __future__ import annotations

import hashlib
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
    "cam_high_roi",
    "cam_left_wrist_roi",
    "cam_right_wrist_roi",
)


def _seed_from_name(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


class _FrozenRandomProjectionView(nn.Module):
    """Deterministic frozen stand-in for one view's DINOv3 CLS embedding.

    A small fixed (seeded) conv stack + global pool + linear head maps an
    arbitrary-resolution RGB image to a fixed out_dim vector. All parameters
    are frozen at construction time, matching the "frozen encoder" contract
    the rest of the system relies on.
    """

    def __init__(self, view_name: str, out_dim: int = 1024):
        super().__init__()
        gen = torch.Generator().manual_seed(_seed_from_name(view_name))
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(32, out_dim)
        with torch.no_grad():
            for p in (self.conv.weight, self.conv.bias, self.proj.weight, self.proj.bias):
                p.copy_(torch.empty_like(p).normal_(generator=gen) * 0.05)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.proj(h)


class FrozenDinoV3Encoder(nn.Module):
    """Six-view frozen encoder producing a (B, n_views * out_dim) descriptor."""

    def __init__(
        self,
        view_names: tuple[str, ...] = DEFAULT_VIEWS,
        out_dim: int = 1024,
        use_real_dinov3: bool = True,
    ):
        super().__init__()
        self.view_names = view_names
        self.out_dim = out_dim
        self.backend = "random_projection_fallback"

        self.per_view = nn.ModuleDict()
        real_backbone = None
        if use_real_dinov3:
            real_backbone = self._try_load_dinov3()

        if real_backbone is not None:
            self.backend = "dinov3_vitl16"
            self.shared_backbone = real_backbone
            for p in self.shared_backbone.parameters():
                p.requires_grad_(False)
        else:
            for name in view_names:
                self.per_view[name] = _FrozenRandomProjectionView(name, out_dim)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _try_load_dinov3():
        try:
            model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=True)
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001 - any failure -> offline fallback
            logger.warning("DINOv3 hub load failed (%s); using frozen random-projection fallback.", exc)
            return None

    @torch.no_grad()
    def _encode_view(self, name: str, image: torch.Tensor) -> torch.Tensor:
        if self.backend == "dinov3_vitl16":
            out = self.shared_backbone(image)
            return out[:, 0] if out.dim() == 3 else out  # CLS token
        return self.per_view[name](image)

    @torch.no_grad()
    def forward(self, views: dict[str, torch.Tensor]) -> torch.Tensor:
        """``views[name]``: (B, 3, H, W) float tensor. Returns (B, n_views*out_dim)."""
        missing = [v for v in self.view_names if v not in views]
        if missing:
            raise ValueError(f"missing views: {missing}")
        embeds = [self._encode_view(name, views[name]) for name in self.view_names]
        return torch.cat(embeds, dim=-1)
