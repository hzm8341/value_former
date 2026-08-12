"""Industrial Track critic training: same ValueFormer architecture, but the
V_mc slot regresses the non-monotonic Physical Progress potential (MSE, per
plan Section 2.1's "engineering loss ablation" allowance) and the V_bin slot
is supervised by physical mistake onset/recovery intervals instead of the
paper's stage-aware labels. Trained and reported completely separately from
train_paper.py -- results from the two must never be conflated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_balanced_sampler
from data.industrial_dataset import IndustrialWindowDataset
from data.synthetic import SyntheticEpisode
from evaluation.evaluator import StateCriticEvaluator
from labels.industrial_progress import ProgressNormalization, ProgressWeights
from models.dinov3_encoder import FrozenDinoV3Encoder
from models.valueformer import ValueFormer, ValueFormerConfig
from training.losses import bin_loss
from training.train_paper import _lr_lambda


@dataclass
class IndustrialTrainResult:
    best_state_dict: dict
    best_epoch: int
    best_val_mse: float
    history: list[dict]
    wall_clock_sec: float


def train_industrial_track(
    train_episodes: list[SyntheticEpisode],
    val_episodes: list[SyntheticEpisode],
    config: dict,
    industrial_config: dict,
    encoder: FrozenDinoV3Encoder | None = None,
    device: str = "cpu",
    max_epochs_override: int | None = None,
    log_fn=print,
) -> IndustrialTrainResult:
    data_cfg, model_cfg, optim_cfg = config["data"], config["model"], config["optim"]
    weights = ProgressWeights(**industrial_config["weights"])
    norm = ProgressNormalization(**industrial_config["normalization"])
    ema_alpha = industrial_config["temporal"]["ema_alpha"]

    encoder = encoder or FrozenDinoV3Encoder(use_real_dinov3=False)
    encoder.to(device)

    seq_len = industrial_config["model"]["seq_len"]
    train_ds = IndustrialWindowDataset(train_episodes, encoder, seq_len, weights, norm, ema_alpha)
    val_ds = IndustrialWindowDataset(val_episodes, encoder, seq_len, weights, norm, ema_alpha)

    batch_size = min(optim_cfg["batch_size"], max(len(train_ds), 1))
    sampler = build_balanced_sampler(train_ds.group_ids(), num_samples=len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 1), shuffle=False)

    vf_config = ValueFormerConfig(
        d_model=industrial_config["model"]["d_model"], n_layers=industrial_config["model"]["n_layers"],
        n_heads=industrial_config["model"]["n_heads"], ffn_mult=model_cfg["ffn_mult"],
        dropout_body=model_cfg["dropout_body"], dropout_aux=model_cfg["dropout_aux"],
        vision_dim_per_view=model_cfg["vision_dim_per_view"], n_views=model_cfg["n_views"],
        state_dim=data_cfg["state_dim"], time_feat_dim=data_cfg["time_feat_dim"], seq_len=seq_len,
        vmc_hidden=tuple(model_cfg["vmc_hidden"]), vbin_hidden=tuple(model_cfg["vbin_hidden"]),
    )
    model = ValueFormer(vf_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    max_epochs = max_epochs_override or optim_cfg["max_epochs"]
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = optim_cfg["warmup_epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_lambda(s, warmup_steps, total_steps))

    evaluator = StateCriticEvaluator()
    history, best_val_mse, best_state_dict, best_epoch, patience = [], float("inf"), None, -1, 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            out = model(batch["vision"].to(device), batch["state"].to(device), batch["time_feat"].to(device))
            progress_loss = F.mse_loss(out["v_mc"], batch["progress"].to(device))
            mistake_loss = bin_loss(out["v_bin_logit"], batch["mistake_ok"].to(device))
            loss = progress_loss + mistake_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

        report = _validate_industrial(model, val_loader, evaluator, device)
        report["epoch"] = epoch
        history.append(report)
        log_fn(f"epoch {epoch:3d} | progress_mse={report['val_mse']:.6f} sep={report['separation']:.4f}")

        if report["val_mse"] < best_val_mse:
            best_val_mse, best_epoch = report["val_mse"], epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= optim_cfg["early_stop_patience"]:
                break

    return IndustrialTrainResult(
        best_state_dict=best_state_dict or model.state_dict(), best_epoch=best_epoch,
        best_val_mse=best_val_mse, history=history, wall_clock_sec=time.time() - t0,
    )


@torch.no_grad()
def _validate_industrial(model, val_loader, evaluator: StateCriticEvaluator, device: str) -> dict:
    model.eval()
    preds, targets, success = [], [], []
    for batch in val_loader:
        out = model(batch["vision"].to(device), batch["state"].to(device), batch["time_feat"].to(device))
        preds.append(out["v_mc"].cpu().numpy())
        targets.append(batch["progress"].numpy())
        success.append(batch["success"].numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    target = np.concatenate(targets) if targets else np.array([])
    succ = np.concatenate(success) if success else np.array([], dtype=bool)
    return evaluator.evaluate(pred, target, succ)
