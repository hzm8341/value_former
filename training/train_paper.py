"""Paper Reproduction Track training loop.

Source: ValueFormer v1 Section IV-E / V-A: AdamW lr=1e-4, weight_decay=0.05,
5-epoch linear warmup then cosine decay to 0, batch_size=256, 4-group
balanced sampler, up to 100 epochs with early stopping on validation MSE
(patience=15). Training uses the BCE objective of Eq.(6)-(7) exclusively;
validation MSE is a measurement/early-stopping signal only, never a training
signal (Section 1.1 of the R&D plan).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ValueFormerWindowDataset, build_balanced_sampler
from data.synthetic import SyntheticEpisode
from evaluation.evaluator import StateCriticEvaluator
from models.dinov3_encoder import FrozenDinoV3Encoder
from models.valueformer import ValueFormer, ValueFormerConfig
from training.losses import combined_loss


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainResult:
    best_state_dict: dict
    best_epoch: int
    best_val_mse: float
    history: list[dict]
    wall_clock_sec: float


def train_paper_track(
    train_episodes: list[SyntheticEpisode],
    val_episodes: list[SyntheticEpisode],
    config: dict,
    encoder: FrozenDinoV3Encoder | None = None,
    device: str = "cpu",
    max_epochs_override: int | None = None,
    log_fn=print,
) -> TrainResult:
    data_cfg, model_cfg, label_cfg = config["data"], config["model"], config["labels"]
    loss_cfg, optim_cfg = config["loss"], config["optim"]

    encoder = encoder or FrozenDinoV3Encoder(use_real_dinov3=False)
    encoder.to(device)

    train_ds = ValueFormerWindowDataset(
        train_episodes, encoder,
        seq_len=data_cfg["seq_len"], gamma=label_cfg["gamma"],
        n_stages=data_cfg["n_stages"], fail_shape=label_cfg["fail_shape"],
    )
    val_ds = ValueFormerWindowDataset(
        val_episodes, encoder,
        seq_len=data_cfg["seq_len"], gamma=label_cfg["gamma"],
        n_stages=data_cfg["n_stages"], fail_shape=label_cfg["fail_shape"],
    )

    batch_size = min(optim_cfg["batch_size"], max(len(train_ds), 1))
    sampler = build_balanced_sampler(train_ds.group_ids(), num_samples=len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 1), shuffle=False)

    vf_config = ValueFormerConfig(
        d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"], n_heads=model_cfg["n_heads"],
        ffn_mult=model_cfg["ffn_mult"], dropout_body=model_cfg["dropout_body"], dropout_aux=model_cfg["dropout_aux"],
        vision_dim_per_view=model_cfg["vision_dim_per_view"], n_views=model_cfg["n_views"],
        state_dim=data_cfg["state_dim"], time_feat_dim=data_cfg["time_feat_dim"], seq_len=data_cfg["seq_len"],
        vmc_hidden=tuple(model_cfg["vmc_hidden"]), vbin_hidden=tuple(model_cfg["vbin_hidden"]),
    )
    model = ValueFormer(vf_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    max_epochs = max_epochs_override or optim_cfg["max_epochs"]
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = optim_cfg["warmup_epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps)
    )

    evaluator = StateCriticEvaluator()
    history: list[dict] = []
    best_val_mse = float("inf")
    best_state_dict = None
    best_epoch = -1
    patience_counter = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            vision = batch["vision"].to(device)
            state = batch["state"].to(device)
            time_feat = batch["time_feat"].to(device)
            v_mc_target = batch["v_mc"].to(device)
            v_bin_target = batch["v_bin"].to(device)

            out = model(vision, state, time_feat)
            losses = combined_loss(
                out["v_mc_logit"], v_mc_target, out["v_bin_logit"], v_bin_target,
                beta=loss_cfg["beta"], w_fail=loss_cfg["w_fail"], alpha_clip=tuple(loss_cfg["alpha_clip"]),
            )
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

        report = _validate(model, val_loader, evaluator, device)
        report["epoch"] = epoch
        history.append(report)
        log_fn(f"epoch {epoch:3d} | val_mse={report['val_mse']:.6f} val_mae={report['val_mae']:.4f} "
               f"sep={report['separation']:.4f}")

        if report["val_mse"] < best_val_mse:
            best_val_mse = report["val_mse"]
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= optim_cfg["early_stop_patience"]:
                log_fn(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val_mse={best_val_mse:.6f})")
                break

    wall_clock = time.time() - t0
    return TrainResult(
        best_state_dict=best_state_dict or model.state_dict(),
        best_epoch=best_epoch,
        best_val_mse=best_val_mse,
        history=history,
        wall_clock_sec=wall_clock,
    )


@torch.no_grad()
def _validate(model: ValueFormer, val_loader: DataLoader, evaluator: StateCriticEvaluator, device: str) -> dict:
    model.eval()
    preds_mc, targets_mc, preds_bin, targets_bin, success_mask = [], [], [], [], []
    for batch in val_loader:
        out = model(batch["vision"].to(device), batch["state"].to(device), batch["time_feat"].to(device))
        preds_mc.append(out["v_mc"].cpu().numpy())
        targets_mc.append(batch["v_mc"].numpy())
        preds_bin.append(out["v_bin"].cpu().numpy())
        targets_bin.append(batch["v_bin"].numpy())
        success_mask.append(batch["success"].numpy())

    v_mc_pred = np.concatenate(preds_mc) if preds_mc else np.array([])
    v_mc_target = np.concatenate(targets_mc) if targets_mc else np.array([])
    v_bin_pred = np.concatenate(preds_bin) if preds_bin else np.array([])
    v_bin_target = np.concatenate(targets_bin) if targets_bin else np.array([])
    success = np.concatenate(success_mask) if success_mask else np.array([], dtype=bool)

    return evaluator.evaluate(v_mc_pred, v_mc_target, success, v_bin_pred, v_bin_target)
