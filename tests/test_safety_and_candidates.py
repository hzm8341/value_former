import numpy as np

from candidate_generator.generator import CandidateActionGenerator
from policy_adapters.canonical_action import CanonicalAction, resample_chunk
from safety.envelope import SafetyEnvelope, SafetyLimits

NOMINAL = CanonicalAction(dx=0.005, dy=0.0, dz=-0.003, drx=0, dry=0, drz=0, gripper_state=1.0, dt=0.1)


def test_candidate_generator_produces_expected_origins():
    gen = CandidateActionGenerator(n_perturbations=3, seed=0)
    cands = gen.generate(NOMINAL, e_xy=(0.002, -0.001), e_rot=0.01, contact_side="left")
    origins = {c.origin for c in cands}
    assert origins == {"nominal", "geometry_correction", "contact_correction", "withdraw", "perturbation"}
    assert sum(c.origin == "perturbation" for c in cands) == 3


def test_safety_envelope_rejects_workspace_violation():
    gen = CandidateActionGenerator(seed=0)
    big_step = CanonicalAction(dx=0.4, dy=0.0, dz=0.0, drx=0, dry=0, drz=0, gripper_state=1.0, dt=0.1)
    cands = gen.generate(big_step)
    limits = SafetyLimits((-0.1, -0.1, 0.0), (0.1, 0.1, 0.1), max_cartesian_step_m=0.5, max_rotation_step_rad=0.5)
    env = SafetyEnvelope(limits)
    accepted, decisions = env.filter(cands, current_ee_pos=np.array([0.0, 0.0, 0.05]))
    nominal_decision = next(d for d in decisions if d.candidate_id == "cand_0_nominal")
    assert not nominal_decision.accepted
    assert "workspace_limit_violation" in nominal_decision.reasons


def test_safety_envelope_jam_active_only_allows_withdraw():
    gen = CandidateActionGenerator(seed=0)
    cands = gen.generate(NOMINAL)
    limits = SafetyLimits((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5), max_cartesian_step_m=0.02, max_rotation_step_rad=0.2)
    env = SafetyEnvelope(limits)
    accepted, _ = env.filter(cands, current_ee_pos=np.array([0.1, 0.1, 0.1]), jam_active=True)
    assert len(accepted) == 1
    assert accepted[0].origin == "withdraw"


def test_resample_chunk_preserves_total_displacement():
    raw = np.zeros((30, 6), dtype=np.float32)
    raw[:, 0] = 0.001  # constant per-step dx at 30 fps
    resampled, mask = resample_chunk(raw, raw_dt=1 / 30, target_dt=0.1)
    assert mask.all()
    total_raw = raw[:, 0].sum()
    total_resampled = resampled[:, 0].sum()
    # Resampling only covers [0, t_raw[-1]]; a coarser target grid can miss a
    # fractional trailing step, so allow slack proportional to one target step.
    assert abs(total_raw - total_resampled) < raw[:, 0].mean() * 3
