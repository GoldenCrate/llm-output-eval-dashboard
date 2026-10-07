"""
data_gen.py — Synthetic data generator for the LLM Output Evaluation Scorecard.

=============================  SYNTHETIC DATA  ==============================
Everything produced here is FAKE, generated from a fixed random seed. It does
NOT come from any real model, benchmark, customer, or proprietary dataset. It
exists only so the dashboard has something realistic to demonstrate against.
============================================================================

Scenario
--------
We are evaluating an LLM-powered answer system. For a set of user queries, the
system retrieves/generates several candidate responses and must (a) decide
whether each response is "acceptable" to show (a binary quality judgment) and
(b) rank the candidates so the best one surfaces first.

We compare two approaches head-to-head:
  * Model A  — the incumbent / baseline scorer.
  * Model B  — a candidate challenger we are considering shipping.

Each row is one (query, candidate response) pair with:
  item_id         unique id
  query_id        groups candidates that belong to the same user query (for NDCG)
  task_type       segment: kind of task (summarization, code, reasoning, ...)
  difficulty      segment: easy / medium / hard
  true_relevance  graded ground-truth relevance 0..3 (human label)
  y_true          binary ground truth: 1 if response is acceptable (relevance>=2)
  score_a         Model A's predicted relevance score in [0,1]
  score_b         Model B's predicted relevance score in [0,1]
  pred_a          Model A binary accept decision (score_a >= THRESHOLD)
  pred_b          Model B binary accept decision (score_b >= THRESHOLD)

Design intent (so the analysis is interesting, not a toy):
  * Model B is modestly BETTER overall.
  * But B is deliberately a bit WORSE on one segment ("code"), so the
    error-analysis view and the recommendation have real nuance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
THRESHOLD = 0.5  # decision threshold applied to both models' scores
N_QUERIES = 900
OUTPUT_CSV = "eval_data.csv"

TASK_TYPES = ["summarization", "code", "reasoning", "extraction", "chitchat"]
DIFFICULTIES = ["easy", "medium", "hard"]

# Base acceptable-rate of the underlying candidates by difficulty.
_DIFFICULTY_BASE = {"easy": 0.72, "medium": 0.52, "hard": 0.33}

# How well each model's SCORE separates good from bad candidates, per task.
# Higher = the model's score is more informative (better AUC-like separation).
# Model B is stronger everywhere EXCEPT "code", where it regressed.
_SEPARATION_A = {
    "summarization": 1.05,
    "code": 1.35,
    "reasoning": 0.95,
    "extraction": 1.15,
    "chitchat": 1.25,
}
_SEPARATION_B = {
    "summarization": 1.55,
    "code": 1.00,   # <-- B regressed here vs A (1.35)
    "reasoning": 1.45,
    "extraction": 1.60,
    "chitchat": 1.70,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_data(seed: int = SEED) -> pd.DataFrame:
    """Return the synthetic evaluation DataFrame (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    rows = []
    item_id = 0

    for q in range(N_QUERIES):
        task = rng.choice(TASK_TYPES, p=[0.22, 0.18, 0.22, 0.20, 0.18])
        difficulty = rng.choice(DIFFICULTIES, p=[0.4, 0.4, 0.2])
        base = _DIFFICULTY_BASE[difficulty]
        n_candidates = int(rng.integers(3, 7))  # 3..6 candidates per query

        sep_a = _SEPARATION_A[task]
        sep_b = _SEPARATION_B[task]

        for _ in range(n_candidates):
            # Latent true quality of this candidate (continuous).
            quality = rng.normal(loc=base, scale=0.28)
            quality = float(np.clip(quality, 0.0, 1.0))

            # Graded ground-truth relevance 0..3 from latent quality + noise.
            graded = quality + rng.normal(0, 0.12)
            if graded < 0.30:
                true_relevance = 0
            elif graded < 0.50:
                true_relevance = 1
            elif graded < 0.72:
                true_relevance = 2
            else:
                true_relevance = 3
            y_true = 1 if true_relevance >= 2 else 0

            # Each model observes latent quality through its own noisy lens.
            # Separation controls how tightly the score tracks true quality.
            logit_a = sep_a * (quality - 0.5) * 6.0 + rng.normal(0, 1.0)
            logit_b = sep_b * (quality - 0.5) * 6.0 + rng.normal(0, 1.0)
            score_a = float(_sigmoid(np.array(logit_a)))
            score_b = float(_sigmoid(np.array(logit_b)))

            rows.append(
                {
                    "item_id": item_id,
                    "query_id": q,
                    "task_type": task,
                    "difficulty": difficulty,
                    "true_relevance": true_relevance,
                    "y_true": y_true,
                    "score_a": round(score_a, 4),
                    "score_b": round(score_b, 4),
                    "pred_a": int(score_a >= THRESHOLD),
                    "pred_b": int(score_b >= THRESHOLD),
                }
            )
            item_id += 1

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    df = generate_data()
    df.to_csv(OUTPUT_CSV, index=False)
    acc_a = (df["pred_a"] == df["y_true"]).mean()
    acc_b = (df["pred_b"] == df["y_true"]).mean()
    print("Wrote SYNTHETIC dataset ->", OUTPUT_CSV)
    print(f"  rows (query/candidate pairs): {len(df):,}")
    print(f"  unique queries              : {df['query_id'].nunique():,}")
    print(f"  segments (task_type)        : {sorted(df['task_type'].unique())}")
    print(f"  overall accuracy  Model A   : {acc_a:.3f}")
    print(f"  overall accuracy  Model B   : {acc_b:.3f}")


if __name__ == "__main__":
    main()
