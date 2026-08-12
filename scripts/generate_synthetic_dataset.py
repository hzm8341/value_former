"""Generates a synthetic episode set and writes the episode/session-level
manifest + split assignment required by R&D plan V3.2 Section 3.2 / 10
(no frame leakage, split by whole episode).

Usage: python scripts/generate_synthetic_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import episode_level_split
from data.synthetic import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-expert-success", type=int, default=20)
    parser.add_argument("--n-rollout-success", type=int, default=8)
    parser.add_argument("--n-rollout-fail", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = parser.parse_args()

    episodes = generate_synthetic_dataset(
        n_expert_success=args.n_expert_success,
        n_rollout_success=args.n_rollout_success,
        n_rollout_fail=args.n_rollout_fail,
        seed=args.seed,
    )
    train_eps, val_eps = episode_level_split(episodes, val_fraction=args.val_fraction, seed=args.seed)

    manifest = [
        {
            "episode_id": ep.episode_id,
            "source_group": ep.source_group,
            "success": ep.success,
            "rollout_signature": ep.rollout_signature,
            "s_fail": ep.s_fail,
            "n_frames": len(ep.sample_times),
            "duration_sec": float(ep.sample_times[-1]),
        }
        for ep in episodes
    ]
    manifest_path = args.out_dir / "manifests" / f"synthetic_seed{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    split = {
        "train": [ep.episode_id for ep in train_eps],
        "val": [ep.episode_id for ep in val_eps],
    }
    split_path = args.out_dir / "splits" / "paper_track" / f"synthetic_seed{args.seed}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))

    print(f"wrote {len(manifest)} episodes to {manifest_path}")
    print(f"wrote split ({len(split['train'])} train / {len(split['val'])} val) to {split_path}")


if __name__ == "__main__":
    main()
