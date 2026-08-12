import numpy as np
import pytest

from labels.paper_labels import (
    FAIL_SHAPES,
    MistakeInterval,
    binary_mistake_labels,
    failed_mc_labels,
    success_mc_labels,
)


def test_success_labels_reach_one_at_terminal():
    v = success_mc_labels(50, gamma=0.99)
    assert v[-1] == pytest.approx(1.0)
    assert np.all(np.diff(v) > 0)  # strictly rising


def test_mc_smooth_matches_success_before_failure_and_decays_after():
    n, s_fail, n_stages, gamma = 100, 3, 6, 0.99
    v = failed_mc_labels(n, s_fail, n_stages, gamma, shape="mc_smooth")
    v_succ = success_mc_labels(n, gamma)
    k_fail = int(np.floor((s_fail / n_stages) * n))
    assert np.allclose(v[: k_fail + 1], v_succ[: k_fail + 1])
    assert not np.all(np.diff(v[k_fail:]) > 0)  # not rising after failure
    assert v[-1] < v[k_fail]  # decays, does not hit zero cliff
    assert v[-1] > 0.0


def test_cliff_shape_drops_to_zero_at_kfail():
    n, s_fail, n_stages, gamma = 100, 3, 6, 0.99
    v = failed_mc_labels(n, s_fail, n_stages, gamma, shape="cliff")
    k_fail = int(np.floor((s_fail / n_stages) * n))
    assert np.all(v[k_fail:] == 0.0)
    assert v[k_fail - 1] > 0.0


@pytest.mark.parametrize("shape", FAIL_SHAPES)
def test_all_shapes_stay_in_unit_interval(shape):
    v = failed_mc_labels(120, 4, 6, 0.99, shape=shape)
    assert np.all(v >= 0.0) and np.all(v <= 1.0)


def test_binary_mistake_labels_zero_inside_interval_only():
    times = np.arange(10) * 0.5
    v = binary_mistake_labels(times, [MistakeInterval(2.0, 3.5)])
    expected = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float)
    assert np.array_equal(v, expected)


def test_invalid_s_fail_raises():
    with pytest.raises(ValueError):
        failed_mc_labels(10, s_fail=0, n_stages=6, gamma=0.99)
