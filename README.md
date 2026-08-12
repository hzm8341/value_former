# ValueFormer

Implementation of **ValueFormer: A Causal Transformer Value Function with
Stage-Aware Labels for Semi-Autonomous Vision-Language-Action Policies**
(arXiv:2608.02958v1), plus the scaffolding for the industrial AOP / Contact
AOP / Lite-LDA migration path described in `ValueFormer复现与AOP迁移研发计划_V3_2.html`
(R&D plan V3.2, kept in this repo as the design source of truth).

## What this repo is

Two tracks, kept strictly separate per the R&D plan's core rule:

- **Paper Reproduction Track** (`labels/paper_labels.py`, `training/train_paper.py`)
  — a faithful implementation of the paper: stage-aware success-then-decay
  Monte Carlo labels (Eq. 2-4), the dual-head causal Transformer (Table I /
  Fig. 7), and the Eq. (6)-(7) BCE training objective.
- **Industrial Track** (`labels/industrial_progress.py`,
  `labels/contact_labels.py`, `training/train_industrial.py`, and the AOP /
  Contact AOP / Lite-LDA / candidate-generator / safety modules) — V0
  scaffolding for the non-monotonic Physical Progress critic and the later
  action-conditioned phases (Phase 4-6 of the plan). These are **not**
  claims of a finished, gate-passed product pipeline; they are runnable,
  tested interfaces that match the plan's Section 10 repository skeleton, so
  the phases can be filled in with real robot data without re-deriving the
  architecture.

## What is NOT included

- **Real DINOv3 weights / real robot data.** This environment has neither
  network access to fetch DINOv3 nor the 1,427-episode LeRobot v2
  sandwich-assembly dataset the paper reports on. `models/dinov3_encoder.py`
  tries `torch.hub` first and transparently falls back to a deterministic,
  seeded, frozen random-projection encoder with the same output shape
  (`backend == "random_projection_fallback"`) so the rest of the pipeline is
  fully testable offline. **Do not read numbers produced against the
  fallback encoder or synthetic data as a reproduction of Table II** — they
  only prove the pipeline (labels → dataset → model → loss → training loop →
  eval) is wired correctly.
- **AOP / Contact AOP / Lite-LDA training data and real-hardware gates.**
  Phases 4-6 of the plan are gated behind real paired-action ranking on
  hardware (Gate 4) and are implemented here as correctly-shaped, unit-tested
  models and a toy ranking demo only. Do not train these for a real product
  decision without the real data and gate evaluation the plan specifies.

## Repository layout

```
configs/            paper_valueformer / industrial_progress / aop / contact_aop YAML snapshots
labels/              paper_labels.py (Eq 2-4), industrial_progress.py (Section 2.4),
                      contact_labels.py (Task Phase / Contact / Mistake), label_schema.json
data/                synthetic.py (offline episode generator), dataset.py / industrial_dataset.py
models/              dinov3_encoder.py, valueformer.py, aop.py, contact_aop.py, lite_lda.py
policy_adapters/     canonical_action.py (Section 4.1)
candidate_generator/ generator.py (Section 4.2)
safety/              envelope.py (Section 4.3, deterministic pre-AOP filter)
training/            losses.py (Eq 6-7), train_paper.py, train_industrial.py
evaluation/          metrics.py, evaluator.py (Table II/III + AOP ranking stats)
scripts/             generate_synthetic_dataset.py, run_paper_reproduction_demo.py
tests/               pytest coverage for every module above
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                      # 38 unit tests, offline, ~2s
python scripts/generate_synthetic_dataset.py     # episode manifest + train/val split
python scripts/run_paper_reproduction_demo.py    # full pipeline smoke test end to end
```

`run_paper_reproduction_demo.py` generates a synthetic dataset reproducing
the four canonical rollout signatures from Fig. 10 (clean success,
success-with-retry, early-collapse, stuck-scratching), trains the Paper
Track ValueFormer and the Industrial Track progress critic for a few epochs
each, and runs a toy AOP ranking over safety-filtered candidates.

## Validated against the paper

- `ValueFormer.num_trainable_params()` returns **3,459,586**, matching
  Table I's reported ≈3.46M trainable parameters exactly, for the same
  six-view dual-head configuration.
- The causal mask is unit-tested directly on the Transformer body: earlier
  positions are provably unaffected by perturbing a later frame
  (`tests/test_model_shapes.py::test_causal_mask_blocks_future_positions`).
- The MC-smooth label (Eq. 3) is tested to reuse the success curve exactly
  up to `k_fail` and to decay smoothly (never cliff to zero) afterward,
  and the four ablation shapes (A/B/C-linear/C-late, Section V) are
  implemented alongside it in `labels/paper_labels.py`.

## Using real data

To move from this scaffold to a real reproduction:

1. Replace `FrozenDinoV3Encoder(use_real_dinov3=True)` and ensure
   `torch.hub` can fetch DINOv3 ViT-L/16 (or point it at local weights).
2. Replace `data/synthetic.py` with a loader over your LeRobot v2 episode
   set, producing the same `SyntheticEpisode`-shaped fields (or adapt
   `data/dataset.py` to your own episode type).
3. Keep `configs/paper_valueformer/base.yaml` unedited for the paper
   snapshot; put any hardware/task-specific changes in
   `configs/industrial_progress/*.yaml` per the plan's Section 1.2 rule
   ("never overwrite the paper config").
4. Follow the R&D plan's Gate sequence (Section 05/11) before promoting
   any Industrial Track model into the AOP / Contact AOP / Lite-LDA phases.

## Source documents

- `ValueFormer.pdf` — arXiv:2608.02958v1
- `ValueFormer复现与AOP迁移研发计划_V3_2.html` — internal R&D plan V3.2
