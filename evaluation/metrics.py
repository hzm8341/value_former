"""Evaluation metrics. Source: R&D plan V3.2 Section 06 and ValueFormer v1
Section VI (Table II/III: validation MSE/MAE, mean-V-success/fail,
separation). AOP/statistical metrics (pairwise ranking CI, McNemar) are
included for the later AOP/online-eval phases per plan Section 06.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def mean_by_success(pred: np.ndarray, success_mask: np.ndarray) -> tuple[float, float]:
    """Returns (mean_V_success, mean_V_fail); nan if a group is empty."""
    succ = pred[success_mask]
    fail = pred[~success_mask]
    mean_succ = float(succ.mean()) if len(succ) else float("nan")
    mean_fail = float(fail.mean()) if len(fail) else float("nan")
    return mean_succ, mean_fail


def separation(pred: np.ndarray, success_mask: np.ndarray) -> float:
    mean_succ, mean_fail = mean_by_success(pred, success_mask)
    return mean_succ - mean_fail


def ap_auroc(scores: np.ndarray, binary_target: np.ndarray) -> tuple[float, float]:
    """binary_target: 1 = positive class to detect (e.g. mistake=1-v_bin)."""
    if len(np.unique(binary_target)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(binary_target, scores)), float(roc_auc_score(binary_target, scores))


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prob - target) ** 2))


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(prob)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if not mask.any():
            continue
        conf = prob[mask].mean()
        acc = target[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def pairwise_ranking_accuracy(
    preferred_score: np.ndarray,
    other_score: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Fraction of pairs where the candidate judged better by ground truth
    also scores higher under the model, with a binomial-bootstrap 95% CI.
    Random baseline is 0.5 (Section 06 / Gate 4).
    """
    correct = (preferred_score > other_score).astype(np.float64)
    n = len(correct)
    acc = float(correct.mean()) if n else float("nan")
    rng = np.random.default_rng(seed)
    boots = [correct[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)] if n else []
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))
    return {"accuracy": acc, "ci_lo": lo, "ci_hi": hi, "n_pairs": n}


def exact_mcnemar(b: int, c: int) -> float:
    """Exact McNemar test p-value for paired binary outcomes; b, c are the
    two discordant-pair counts. Uses the exact binomial two-sided test.
    """
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return float("nan")
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
