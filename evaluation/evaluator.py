"""Table-II/III-style report builder for the Paper Reproduction Track, and a
thin AOP ranking evaluator for later phases.
"""
from __future__ import annotations

import numpy as np

from evaluation import metrics as m


class StateCriticEvaluator:
    def __init__(self, mc_target_name: str = "V_mc"):
        self.mc_target_name = mc_target_name

    def evaluate(
        self,
        v_mc_pred: np.ndarray,
        v_mc_target: np.ndarray,
        success_mask: np.ndarray,
        v_bin_pred: np.ndarray | None = None,
        v_bin_target: np.ndarray | None = None,
    ) -> dict[str, float]:
        report = {
            "val_mse": m.mse(v_mc_pred, v_mc_target),
            "val_mae": m.mae(v_mc_pred, v_mc_target),
        }
        mean_succ, mean_fail = m.mean_by_success(v_mc_pred, success_mask)
        report["mean_v_success"] = mean_succ
        report["mean_v_fail"] = mean_fail
        report["separation"] = mean_succ - mean_fail

        if v_bin_pred is not None and v_bin_target is not None:
            mistake_score = 1.0 - v_bin_pred
            mistake_target = (v_bin_target == 0).astype(np.float64)
            ap, auroc = m.ap_auroc(mistake_score, mistake_target)
            report["mistake_ap"] = ap
            report["mistake_auroc"] = auroc
            report["mistake_brier"] = m.brier_score(v_bin_pred, v_bin_target)
            report["mistake_ece"] = m.expected_calibration_error(v_bin_pred, v_bin_target)
        return report


class AOPEvaluator:
    def evaluate(self, preferred_score: np.ndarray, other_score: np.ndarray) -> dict[str, float]:
        return m.pairwise_ranking_accuracy(preferred_score, other_score)
