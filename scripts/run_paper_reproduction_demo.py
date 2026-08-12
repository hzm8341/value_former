"""End-to-end smoke test across the whole repository, on synthetic data.

Runs: synthetic dataset generation -> Paper Track training (Eq. 2-4 labels,
Eq. 6-7 loss, ValueFormer architecture) -> Industrial Track training
(non-monotonic Physical Progress) -> AOP V0 forward + candidate generation +
safety filtering + ranking on a toy state.

This script proves the pipeline is wired correctly end to end. It does NOT
reproduce the paper's reported numbers (Table II) -- that requires the real
1,427-episode LeRobot v2 sandwich-assembly dataset and real DINOv3 features,
neither of which is available in this environment. See README.md.

Usage: python scripts/run_paper_reproduction_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from candidate_generator.generator import CandidateActionGenerator
from data.dataset import episode_level_split
from data.synthetic import generate_synthetic_dataset
from models.aop import ActionOutcomePredictor, AOPConfig, RankingWeights
from models.dinov3_encoder import FrozenDinoV3Encoder
from models.valueformer import ValueFormer, ValueFormerConfig
from policy_adapters.canonical_action import CanonicalAction
from safety.envelope import SafetyEnvelope, SafetyLimits
from training.train_industrial import train_industrial_track
from training.train_paper import train_paper_track


def _valueformer_config_from_yaml(paper_cfg: dict) -> ValueFormerConfig:
    mc, dc = paper_cfg["model"], paper_cfg["data"]
    return ValueFormerConfig(
        d_model=mc["d_model"], n_layers=mc["n_layers"], n_heads=mc["n_heads"], ffn_mult=mc["ffn_mult"],
        dropout_body=mc["dropout_body"], dropout_aux=mc["dropout_aux"], vision_dim_per_view=mc["vision_dim_per_view"],
        n_views=mc["n_views"], state_dim=dc["state_dim"], time_feat_dim=dc["time_feat_dim"], seq_len=dc["seq_len"],
        vmc_hidden=tuple(mc["vmc_hidden"]), vbin_hidden=tuple(mc["vbin_hidden"]),
    )

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main():
    section("1. Synthetic dataset")
    episodes = generate_synthetic_dataset(n_expert_success=16, n_rollout_success=8, n_rollout_fail=8, seed=0)
    train_eps, val_eps = episode_level_split(episodes, val_fraction=0.25, seed=0)
    print(f"episodes: {len(episodes)} total, {len(train_eps)} train / {len(val_eps)} val")

    encoder = FrozenDinoV3Encoder(use_real_dinov3=False)
    print(f"visual frontend backend: {encoder.backend} (offline fallback; not real DINOv3)")

    section("2. Paper Reproduction Track: ValueFormer training (Eq. 2-4 labels, Eq. 6-7 loss)")
    paper_cfg = load_yaml(REPO_ROOT / "configs" / "paper_valueformer" / "base.yaml")
    paper_cfg["optim"]["batch_size"] = 32
    paper_result = train_paper_track(train_eps, val_eps, paper_cfg, encoder=encoder, max_epochs_override=8)
    print(f"\nbest epoch {paper_result.best_epoch}, best val MSE {paper_result.best_val_mse:.6f}, "
          f"wall clock {paper_result.wall_clock_sec:.1f}s")
    probe_model = ValueFormer(_valueformer_config_from_yaml(paper_cfg))
    print(f"trainable params: {probe_model.num_trainable_params():,} (paper Table I: 3,459,586)")

    section("3. Industrial Track: non-monotonic Physical Progress critic")
    industrial_cfg = load_yaml(REPO_ROOT / "configs" / "industrial_progress" / "base.yaml")
    industrial_result = train_industrial_track(
        train_eps, val_eps, paper_cfg, industrial_cfg, encoder=encoder, max_epochs_override=8
    )
    print(f"\nbest epoch {industrial_result.best_epoch}, best val progress-MSE {industrial_result.best_val_mse:.6f}")

    section("4. AOP V0 + Candidate Generator + Safety Envelope (toy state)")
    nominal = CanonicalAction(dx=0.006, dy=0.001, dz=-0.004, drx=0, dry=0, drz=0, gripper_state=1.0, dt=0.1)
    candidates = CandidateActionGenerator(n_perturbations=3, seed=0).generate(
        nominal, e_xy=(0.002, -0.001), e_rot=0.01, contact_side="left"
    )
    limits = SafetyLimits((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5), max_cartesian_step_m=0.02, max_rotation_step_rad=0.2)
    accepted, decisions = SafetyEnvelope(limits).filter(candidates, current_ee_pos=np.array([0.1, 0.1, 0.1]))
    print(f"candidates generated: {len(candidates)}, accepted by safety envelope: {len(accepted)}")

    aop = ActionOutcomePredictor(AOPConfig())
    state_repr = torch.randn(len(accepted), 256)  # stand-in for ValueFormer's state_repr output
    actions = torch.stack([torch.from_numpy(c.action.to_array()) for c in accepted])
    with torch.no_grad():
        out = aop(state_repr, actions)
        scores = ActionOutcomePredictor.ranking_score(out, RankingWeights())
    ranked = sorted(zip(accepted, scores.tolist()), key=lambda x: x[1], reverse=True)
    print("AOP ranking (untrained weights, illustrative only):")
    for cand, score in ranked:
        print(f"  {cand.candidate_id:24s} origin={cand.origin:20s} score={score:+.4f}")

    section("Done")
    print("Full pipeline (labels -> dataset -> ValueFormer -> AOP -> safety -> ranking) ran without error.")


if __name__ == "__main__":
    main()
