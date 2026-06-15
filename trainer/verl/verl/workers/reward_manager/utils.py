from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

import re
import numpy as np
import pandas as pd
import math
import statistics


def parse_prediction(text: str) -> float:
    """Parse the first numeric value. Prefer the first value inside \boxed{...}; if no \boxed{} present, search the whole text."""
    if not text:
        return None

    matches = re.findall(r"\\boxed\s*\{([^}]*)\}", text)
    if matches:
        candidate = matches[-1].strip()
    else:
        return None
        
    # strip common noise
    candidate = candidate.replace(",", "").replace("%", "").strip()

    # match integer/float with optional exponent
    num_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    mnum = re.search(num_re, candidate)
    if mnum:
        try:
            return float(mnum.group(1))
        except Exception:
            return None
    return None


def calculate_metrics(preds_clean: List[float], targets_clean: List[float]) -> Dict[str, float]:

    if preds_clean.size == 0:
        return {"mae": 1000.0, "mse": 1000.0, "rmse": 1000.0, "spearman": 0.0, "kendall_tau": 0.0}

    mae = float(np.mean(np.abs(preds_clean - targets_clean)))
    mse = float(np.mean((preds_clean - targets_clean) ** 2))
    rmse = float(np.sqrt(mse))

    if preds_clean.size < 2:
        # Not enough data
        spearman = float("nan")
        kendall_tau = float("nan")
    else:
        # Spearman correlation = Pearson correlation of ranked values (average ranks for ties)
        s_pred = pd.Series(preds_clean)
        s_tgt = pd.Series(targets_clean)
        pred_rank = s_pred.rank(method="average").to_numpy()
        tgt_rank = s_tgt.rank(method="average").to_numpy()

        with np.errstate(invalid="ignore"):
            corrmat = np.corrcoef(pred_rank, tgt_rank)
            spearman = float(corrmat[0, 1]) if corrmat.size >= 4 else float("nan")

        # Kendall tau-b (accounts for ties), O(n^2)
        n = preds_clean.size
        concordant = 0
        discordant = 0
        ties_pred = 0
        ties_tgt = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                dp = np.sign(preds_clean[j] - preds_clean[i])
                dt = np.sign(targets_clean[j] - targets_clean[i])
                if dp == 0 and dt == 0:
                    # tied pair in both -> neither concordant nor discordant, contributes to ties
                    ties_pred += 1
                    ties_tgt += 1
                elif dp == 0:
                    ties_pred += 1
                elif dt == 0:
                    ties_tgt += 1
                elif dp == dt:
                    concordant += 1
                else:
                    discordant += 1

        denom = np.sqrt(
            (concordant + discordant + ties_pred) *
            (concordant + discordant + ties_tgt)
        )
        kendall_tau = float((concordant - discordant) / denom) if denom > 0 else float("nan")
    
    return {"mae": mae, "mse": mse, "rmse": rmse, "spearman": spearman, "kendall_tau": kendall_tau}


def compute_training_metrics(preds_clean: List[float], targets_clean: List[float], uid_arr: List[str]) -> Dict[str, float]:
    """
    Compute rollout-distribution metrics grouped by uid.

    Inputs are assumed to be rollout-level:
      - preds_clean[j]  = p_{i,k} for some i (identified by uid_arr[j])
      - targets_clean[j]= y_i (same for all rollouts that share the same uid)
      - uid_arr[j]      = unique id for input i

    Returns scalar aggregates over inputs (not rollouts).
    """
    if not (len(preds_clean) == len(targets_clean) == len(uid_arr)):
        raise ValueError("preds_clean, targets_clean, uid_arr must have the same length")

    # Group predictions by uid, and keep one target per uid
    preds_by_uid: Dict[str, List[float]] = defaultdict(list)
    target_by_uid: Dict[str, float] = {}

    for p, y, uid in zip(preds_clean, targets_clean, uid_arr):
        if not (isinstance(uid, str) and uid):
            raise ValueError(f"uid must be a non-empty string, got: {uid!r}")
        if not (math.isfinite(p) and math.isfinite(y)):
            # Skip non-finite values (shouldn't happen if "clean")
            continue
        preds_by_uid[uid].append(float(p))
        if uid in target_by_uid:
            # Targets should be identical across rollouts; if not, take the first but warn via strict check.
            if target_by_uid[uid] != float(y):
                raise ValueError(f"Inconsistent targets for uid={uid}: {target_by_uid[uid]} vs {y}")
        else:
            target_by_uid[uid] = float(y)

    # If everything got filtered out
    if not preds_by_uid:
        return {
            "n_inputs": 0.0,
            "mean_abs_mean_error": float("nan"),
            "mean_abs_median_error": float("nan"),
            "mean_best_of_k_abs_error": float("nan"),
            "mean_worst_of_k_abs_error": float("nan"),
            "mean_signed_bias": float("nan"),
            "mean_abs_signed_bias": float("nan"),
            "one_sided_rate": float("nan"),
            "bracketing_rate": float("nan"),
            "avg_k": float("nan"),
        }

    # Per-input metric accumulators
    abs_mean_errors = []
    abs_median_errors = []
    best_of_k_errors = []
    worst_of_k_errors = []

    signed_biases = []
    abs_signed_biases = []

    one_sided_flags = []
    bracket_flags = []

    k_list = []

    for uid, p_list in preds_by_uid.items():
        if not p_list:
            continue
        y = target_by_uid[uid]

        k = len(p_list)
        k_list.append(k)

        p_mean = sum(p_list) / k
        # statistics.median handles both odd/even K
        p_median = statistics.median(p_list)

        # A) Accuracy (distribution summaries -> point metrics)
        abs_mean_errors.append(abs(p_mean - y))
        abs_median_errors.append(abs(p_median - y))

        abs_errors = [abs(p - y) for p in p_list]
        best_of_k_errors.append(min(abs_errors))
        worst_of_k_errors.append(max(abs_errors))

        # B) Bias / centering
        signed_bias = p_mean - y
        signed_biases.append(signed_bias)
        abs_signed_biases.append(abs(signed_bias))

        # One-sidedness: all rollouts strictly above OR strictly below target
        # (use strict inequality as specified)
        min_diff = min(p - y for p in p_list)
        max_diff = max(p - y for p in p_list)
        one_sided = 1.0 if (min_diff > 0.0 or max_diff < 0.0) else 0.0
        one_sided_flags.append(one_sided)

        # C) Coverage / bracketing
        # inclusive: min <= y <= max
        bracket = 1.0 if (min(p_list) <= y <= max(p_list)) else 0.0
        bracket_flags.append(bracket)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    n_inputs = float(len(abs_mean_errors))

    metrics: Dict[str, float] = {
        "n_inputs": n_inputs,
        "avg_k": _mean(k_list),

        # A) accuracy
        "mean_abs_mean_error": _mean(abs_mean_errors),
        "mean_abs_median_error": _mean(abs_median_errors),
        "mean_best_of_k_abs_error": _mean(best_of_k_errors),
        "mean_worst_of_k_abs_error": _mean(worst_of_k_errors),

        # B) bias/centering
        "mean_signed_bias": _mean(signed_biases),
        "mean_abs_signed_bias": _mean(abs_signed_biases),
        "one_sided_rate": _mean(one_sided_flags),  # fraction in [0,1]

        # C) coverage/bracketing
        "bracketing_rate": _mean(bracket_flags),  # fraction in [0,1]
    }

    return metrics
