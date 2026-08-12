import numpy as np

from labels.contact_labels import MistakeState, TaskPhase, build_mistake_segments, reaction_delay
from labels.industrial_progress import ProgressNormalization, ProgressWeights, RawObservables, physical_progress_sequence


def _weights_norm():
    w = ProgressWeights(0.15, 0.15, 0.4, 0.3, 0.6, 0.4, 0.6)
    n = ProgressNormalization(0.15, 0.01, 0.15, 0.03)
    return w, n


def test_progress_rises_as_robot_approaches_and_inserts():
    t = np.linspace(0, 1, 30)
    obs = RawObservables(
        d_tip=0.2 * (1 - t), e_xy=np.full(30, 0.002), e_rot=np.full(30, 0.02),
        d_insert=0.03 * t, q_contact=t,
        p_jam=np.zeros(30), p_slip=np.zeros(30), p_overload=np.zeros(30),
    )
    w, n = _weights_norm()
    p = physical_progress_sequence(obs, w, n)
    assert p[-1] > p[0]


def test_jam_causes_immediate_non_monotonic_dip_and_recovery():
    t = np.linspace(0, 1, 30)
    p_jam = np.zeros(30)
    p_jam[15:18] = 0.9
    obs = RawObservables(
        d_tip=0.2 * (1 - t), e_xy=np.full(30, 0.002), e_rot=np.full(30, 0.02),
        d_insert=0.03 * t, q_contact=t, p_jam=p_jam, p_slip=np.zeros(30), p_overload=np.zeros(30),
    )
    w, n = _weights_norm()
    p = physical_progress_sequence(obs, w, n)
    assert p[16] < p[14]  # dips during jam
    assert p[-1] > p[16]  # recovers afterward, not clamped to a cumulative max


def test_terminal_override_forces_success_to_one():
    t = np.linspace(0, 1, 10)
    obs = RawObservables(
        d_tip=np.zeros(10), e_xy=np.zeros(10), e_rot=np.zeros(10),
        d_insert=np.full(10, 0.03), q_contact=np.ones(10),
        p_jam=np.zeros(10), p_slip=np.zeros(10), p_overload=np.zeros(10),
    )
    w, n = _weights_norm()
    fully_seated = np.zeros(10, dtype=bool)
    fully_seated[-1] = True
    p = physical_progress_sequence(obs, w, n, fully_seated_mask=fully_seated)
    assert p[-1] == 1.0


def test_task_phase_allows_regression_except_terminal():
    from labels.contact_labels import PHASE_ALLOWS_REGRESSION

    assert PHASE_ALLOWS_REGRESSION[TaskPhase.S3_CONTACT_LOCALIZATION_SEARCH] is True
    assert PHASE_ALLOWS_REGRESSION[TaskPhase.S6_FULLY_SEATED] is False


def test_recovered_mistake_segment_is_recoverable_not_terminal():
    t = list(range(8))
    bad = [False, False, True, True, True, False, False, False]
    recovered = [False] * 8
    segs = build_mistake_segments(t, bad, recovered)
    assert len(segs) == 1
    assert segs[0].state == MistakeState.RECOVERABLE_MISTAKE
    assert segs[0].t_start == 2 and segs[0].t_end == 5


def test_unrecovered_trailing_mistake_is_terminal():
    t = list(range(8))
    bad = [False, True, True, True, True, True, True, True]
    recovered = [False] * 8
    segs = build_mistake_segments(t, bad, recovered)
    assert len(segs) == 1
    assert segs[0].state == MistakeState.TERMINAL_FAILURE


def test_reaction_delay():
    assert reaction_delay(t_intervention=10.0, t_physical_failure_onset=8.5) == 1.5
