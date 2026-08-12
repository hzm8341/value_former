"""ValueFormer: causal Transformer dual-head state critic.

Source: ValueFormer v1 Section IV-D, Table I, Fig. 7. A short causal window
of per-frame vision/state/time features is summed (not concatenated) into a
d_model embedding, passed through a 2-layer causal Transformer encoder, and
the last-token representation feeds two heads: a smooth V_mc (advantage
critic) and a sharp per-frame V_bin (online mistake detector).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ValueFormerConfig:
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    dropout_body: float = 0.2
    dropout_aux: float = 0.4
    vision_dim_per_view: int = 1024
    n_views: int = 6
    state_dim: int = 14
    time_feat_dim: int = 2
    seq_len: int = 16
    vmc_hidden: tuple[int, ...] = (512, 256)
    vbin_hidden: tuple[int, ...] = (128,)

    @property
    def vision_dim(self) -> int:
        return self.vision_dim_per_view * self.n_views


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float, layernorm: bool):
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ValueFormer(nn.Module):
    def __init__(self, config: ValueFormerConfig = ValueFormerConfig()):
        super().__init__()
        self.config = config
        d = config.d_model

        self.vision_proj = nn.Linear(config.vision_dim, d)
        self.state_proj = nn.Linear(config.state_dim, d)
        self.time_proj = nn.Linear(config.time_feat_dim, d)
        self.pos_embed = nn.Embedding(config.seq_len, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_mult * d,
            dropout=config.dropout_body,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(d)

        self.vmc_head = _MLPHead(d, config.vmc_hidden, dropout=config.dropout_body, layernorm=True)
        self.vbin_head = _MLPHead(d, config.vbin_hidden, dropout=config.dropout_aux, layernorm=False)

        self.register_buffer(
            "_causal_mask_cache",
            nn.Transformer.generate_square_subsequent_mask(config.seq_len),
            persistent=False,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len == self.config.seq_len:
            return self._causal_mask_cache.to(device)
        return nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    def forward(
        self,
        vision_embed: torch.Tensor,
        state: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """All inputs shape (B, S, *). Returns logits and sigmoid probabilities
        for V_mc and V_bin, taken from the last (most recent) token position.
        """
        b, s, _ = vision_embed.shape
        x = self.vision_proj(vision_embed) + self.state_proj(state) + self.time_proj(time_feat)
        positions = torch.arange(s, device=x.device).clamp(max=self.config.seq_len - 1)
        x = x + self.pos_embed(positions).unsqueeze(0)

        mask = self._causal_mask(s, x.device)
        h = self.body(x, mask=mask, is_causal=True)
        h = self.final_norm(h)

        last = h[:, -1, :]
        vmc_logit = self.vmc_head(last)
        vbin_logit = self.vbin_head(last)
        return {
            "v_mc_logit": vmc_logit,
            "v_mc": torch.sigmoid(vmc_logit),
            "v_bin_logit": vbin_logit,
            "v_bin": torch.sigmoid(vbin_logit),
            "state_repr": last,
        }

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
