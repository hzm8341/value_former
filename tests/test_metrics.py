import numpy as np

from evaluation import metrics as m
from evaluation.evaluator import StateCriticEvaluator


def test_mse_mae_zero_for_perfect_prediction():
    pred = np.array([0.1, 0.5, 0.9])
    assert m.mse(pred, pred) == 0.0
    assert m.mae(pred, pred) == 0.0


def test_separation_positive_when_success_scores_higher():
    pred = np.array([0.9, 0.8, 0.1, 0.2])
    success = np.array([True, True, False, False])
    sep = m.separation(pred, success)
    assert sep > 0.5


def test_ap_auroc_perfect_separation():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    target = np.array([1, 1, 0, 0])
    ap, auroc = m.ap_auroc(scores, target)
    assert ap == 1.0
    assert auroc == 1.0


def test_pairwise_ranking_accuracy_and_ci_bounds():
    preferred = np.array([1.0, 0.9, 0.2, 0.8])
    other = np.array([0.5, 0.5, 0.5, 0.5])
    result = m.pairwise_ranking_accuracy(preferred, other, n_bootstrap=500, seed=0)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["ci_lo"] <= result["accuracy"] <= result["ci_hi"]


def test_exact_mcnemar_symmetric_gives_high_pvalue():
    p = m.exact_mcnemar(b=10, c=10)
    assert p > 0.5


def test_state_critic_evaluator_report_keys():
    pred = np.array([0.9, 0.8, 0.1, 0.2])
    target = np.array([1.0, 0.9, 0.0, 0.1])
    success = np.array([True, True, False, False])
    v_bin_pred = np.array([1.0, 1.0, 0.2, 0.9])
    v_bin_target = np.array([1.0, 1.0, 0.0, 1.0])
    report = StateCriticEvaluator().evaluate(pred, target, success, v_bin_pred, v_bin_target)
    for key in ("val_mse", "val_mae", "mean_v_success", "mean_v_fail", "separation", "mistake_ap", "mistake_auroc"):
        assert key in report
