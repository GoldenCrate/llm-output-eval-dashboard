"""
metrics.py — Pure, dependency-light compute functions for the scorecard.

Kept separate from app.py so the statistics can be unit-tested headlessly
(no Streamlit import required). All functions are pure: given the same inputs
they return the same outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict:
    """Return TP / FP / TN / FN for binary labels (1 = positive / acceptable)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, F1 for binary predictions."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **c,
    }


# --------------------------------------------------------------------------
# A/B test: two-proportion z-test
# --------------------------------------------------------------------------
@dataclass
class ABResult:
    rate_a: float          # proportion for Model A
    rate_b: float          # proportion for Model B
    n_a: int
    n_b: int
    successes_a: int
    successes_b: int
    abs_lift: float        # rate_b - rate_a (absolute, percentage points as a fraction)
    rel_lift: float        # (rate_b - rate_a) / rate_a
    z: float
    p_value: float         # two-sided
    ci_low: float          # 95% CI on (rate_b - rate_a)
    ci_high: float
    significant: bool      # at alpha = 0.05

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> ABResult:
    """
    Two-sided two-proportion z-test comparing Model B against Model A.

    - z statistic / p-value use the POOLED proportion (standard for the test of
      H0: p_a == p_b).
    - The confidence interval on the difference (p_b - p_a) uses the UNPOOLED
      standard error, which is the conventional CI for a difference of
      proportions. (It is normal for the p-value and CI to use slightly
      different SEs; this matches most A/B-testing references.)
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a

    # Pooled SE for the hypothesis test.
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the CI on the difference.
    se_unpooled = np.sqrt(
        p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    rel_lift = diff / p_a if p_a > 0 else float("nan")

    return ABResult(
        rate_a=p_a,
        rate_b=p_b,
        n_a=int(n_a),
        n_b=int(n_b),
        successes_a=int(successes_a),
        successes_b=int(successes_b),
        abs_lift=diff,
        rel_lift=rel_lift,
        z=float(z),
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(p_value < alpha),
    )


# --------------------------------------------------------------------------
# Ranking metric: NDCG@k
# --------------------------------------------------------------------------
def _dcg(relevances: np.ndarray) -> float:
    """Discounted cumulative gain with gain = 2^rel - 1."""
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    gains = (2.0 ** relevances) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k_single(scores, relevances, k: int) -> float:
    """NDCG@k for a single query given model scores and graded relevances."""
    scores = np.asarray(scores, dtype=float)
    relevances = np.asarray(relevances, dtype=float)
    if scores.size == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")          # rank by model score
    ranked_rel = relevances[order][:k]
    ideal_rel = np.sort(relevances)[::-1][:k]            # best possible ordering

    idcg = _dcg(ideal_rel)
    if idcg == 0:
        return float("nan")  # no relevant items for this query -> undefined
    return _dcg(ranked_rel) / idcg


def mean_ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 3,
                   rel_col: str = "true_relevance",
                   group_col: str = "query_id") -> float:
    """Mean NDCG@k over all queries for a given model's score column."""
    vals = []
    for _, g in df.groupby(group_col):
        v = ndcg_at_k_single(g[score_col].values, g[rel_col].values, k)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------
# Segment breakdown
# --------------------------------------------------------------------------
def segment_breakdown(df: pd.DataFrame, segment_col: str,
                      k: int = 3) -> pd.DataFrame:
    """
    Per-segment comparison of A vs B: accuracy, F1, and NDCG@k, plus the
    accuracy delta (B - A). One row per segment value.
    """
    records = []
    for seg_val, g in df.groupby(segment_col):
        ma = classification_metrics(g["y_true"], g["pred_a"])
        mb = classification_metrics(g["y_true"], g["pred_b"])
        ndcg_a = mean_ndcg_at_k(g, "score_a", k=k)
        ndcg_b = mean_ndcg_at_k(g, "score_b", k=k)
        records.append(
            {
                "segment": seg_val,
                "n_items": len(g),
                "acc_a": ma["accuracy"],
                "acc_b": mb["accuracy"],
                "acc_delta": mb["accuracy"] - ma["accuracy"],
                "f1_a": ma["f1"],
                "f1_b": mb["f1"],
                "ndcg_a": ndcg_a,
                "ndcg_b": ndcg_b,
                "ndcg_delta": ndcg_b - ndcg_a,
            }
        )
    out = pd.DataFrame(records).sort_values("acc_delta").reset_index(drop=True)
    return out


def scorecard(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Headline side-by-side metrics table for both models."""
    ma = classification_metrics(df["y_true"], df["pred_a"])
    mb = classification_metrics(df["y_true"], df["pred_b"])
    rows = [
        ("Accuracy", ma["accuracy"], mb["accuracy"]),
        ("Precision", ma["precision"], mb["precision"]),
        ("Recall", ma["recall"], mb["recall"]),
        ("F1", ma["f1"], mb["f1"]),
        (f"NDCG@{k}", mean_ndcg_at_k(df, "score_a", k), mean_ndcg_at_k(df, "score_b", k)),
    ]
    return pd.DataFrame(rows, columns=["metric", "model_a", "model_b"])
