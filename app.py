"""
app.py — LLM Output Evaluation Scorecard (Streamlit)

A portfolio demo by Leo Chan (github.com/GoldenCrate).

Compares two candidate LLM scoring/ranking approaches ("Model A" the incumbent
vs "Model B" the challenger) on a SYNTHETIC labeled dataset, across four views:

  1. Scorecard        — headline KPI tiles + side-by-side metric table
  2. A/B Comparison   — two-proportion z-test (lift, p-value, 95% CI)
  3. Error Analysis   — confusion matrices + per-segment breakdown
  4. So What          — plain-English recommendation

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data": the title
states the takeaway, clutter is stripped, series are directly labeled, and
color is restrained to one neutral (baseline) + one accent (challenger).

ALL DATA IS SYNTHETIC — see data_gen.py. Nothing here is real or proprietary.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

import metrics as M
from data_gen import OUTPUT_CSV, THRESHOLD, generate_data

# ---------------------------------------------------------------------------
# Restrained "Storytelling with Data" palette: grey baseline, one accent.
# ---------------------------------------------------------------------------
GREY = "#8E8E93"        # Model A (incumbent / baseline)
ACCENT = "#2563EB"      # Model B (challenger) — the single accent color
INK = "#1F2933"         # primary text
MUTED = "#6B7280"       # secondary text
GOOD = "#15803D"        # status: improvement
BAD = "#B91C1C"         # status: regression
GRID = "#E5E7EB"

NDCG_K = 3

st.set_page_config(
    page_title="LLM Output Evaluation Scorecard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if os.path.exists(OUTPUT_CSV):
        return pd.read_csv(OUTPUT_CSV)
    return generate_data()


def style_ax(ax):
    """Declutter a matplotlib axis, SWD-style."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.title.set_color(INK)
    return ax


def fig_caption(fig):
    fig.text(0.01, -0.02, "Synthetic data — illustrative only.",
             fontsize=7.5, color=MUTED, ha="left")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_metric_bars(sc: pd.DataFrame):
    """Grouped horizontal bars: Model A vs B across all headline metrics."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    metrics_list = sc["metric"].tolist()
    y = np.arange(len(metrics_list))
    h = 0.38

    ax.barh(y + h / 2, sc["model_a"], height=h, color=GREY, label="Model A (incumbent)")
    ax.barh(y - h / 2, sc["model_b"], height=h, color=ACCENT, label="Model B (challenger)")

    for yi, (va, vb) in enumerate(zip(sc["model_a"], sc["model_b"])):
        ax.text(va + 0.01, yi + h / 2, f"{va:.2f}", va="center", fontsize=8.5, color=MUTED)
        ax.text(vb + 0.01, yi - h / 2, f"{vb:.2f}", va="center", fontsize=8.5,
                color=ACCENT, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics_list)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([])
    ax.invert_yaxis()
    style_ax(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Model B leads on every headline metric", loc="left",
                 fontsize=12, fontweight="bold", pad=12)
    fig_caption(fig)
    fig.tight_layout()
    return fig


def chart_segment_delta(seg: pd.DataFrame, segment_name: str):
    """Diverging horizontal bars of accuracy delta (B - A) by segment."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    seg = seg.sort_values("acc_delta")
    colors = [GOOD if d >= 0 else BAD for d in seg["acc_delta"]]
    y = np.arange(len(seg))
    ax.barh(y, seg["acc_delta"] * 100, color=colors)

    for yi, d in zip(y, seg["acc_delta"]):
        off = 0.15 if d >= 0 else -0.15
        ha = "left" if d >= 0 else "right"
        ax.text(d * 100 + off, yi, f"{d*100:+.1f} pp", va="center", ha=ha,
                fontsize=8.5, color=(GOOD if d >= 0 else BAD), fontweight="bold")

    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(seg["segment"])
    ax.set_xlabel("Accuracy change, Model B − Model A (percentage points)",
                  fontsize=9, color=MUTED)
    lim = max(6.0, np.abs(seg["acc_delta"] * 100).max() * 1.35)
    ax.set_xlim(-lim, lim)
    style_ax(ax)
    ax.set_title(f"B helps most segments — but regresses on one ({segment_name})",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    fig_caption(fig)
    fig.tight_layout()
    return fig


def chart_confusion(y_true, y_pred, title, accent):
    """Confusion matrix rendered as a clean labeled 2x2 grid."""
    c = M.confusion_counts(y_true, y_pred)
    grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    vmax = grid.max()
    for (i, j), val in np.ndenumerate(grid):
        # Shade cells by count, but only in the accent hue (restrained).
        shade = 0.12 + 0.6 * (val / vmax if vmax else 0)
        is_error = (i, j) in [(0, 1), (1, 0)]
        face = (BAD if is_error else accent)
        ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=face, alpha=shade,
                               edgecolor="white", linewidth=2))
        label = {(0, 0): "True negative", (0, 1): "False positive",
                 (1, 0): "False negative", (1, 1): "True positive"}[(i, j)]
        ax.text(j + 0.5, 1 - i + 0.62, f"{val:,}", ha="center", va="center",
                fontsize=15, fontweight="bold", color=INK)
        ax.text(j + 0.5, 1 - i + 0.30, label, ha="center", va="center",
                fontsize=8, color=MUTED)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted reject", "Predicted accept"], fontsize=8.5)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actually\nacceptable", "Actually\nnot"], fontsize=8.5)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    fig.tight_layout()
    return fig


def chart_ab(ab: M.ABResult):
    """Point-estimate + 95% CI of the lift (B − A) on accept-rate."""
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    lo, hi, mid = ab.ci_low * 100, ab.ci_high * 100, ab.abs_lift * 100
    ax.plot([lo, hi], [0, 0], color=ACCENT, linewidth=3, solid_capstyle="round")
    ax.plot(mid, 0, "o", color=ACCENT, markersize=11)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")

    ax.text(mid, 0.22, f"{mid:+.2f} pp", ha="center", color=ACCENT,
            fontweight="bold", fontsize=11)
    ax.text(lo, -0.28, f"{lo:+.2f}", ha="center", color=MUTED, fontsize=8.5)
    ax.text(hi, -0.28, f"{hi:+.2f}", ha="center", color=MUTED, fontsize=8.5)
    ax.text(0, -0.5, "no difference", ha="center", color=MUTED, fontsize=8)

    span = max(abs(lo), abs(hi)) * 1.5 + 0.5
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.7, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Accept-rate lift, Model B − Model A (percentage points)",
                  fontsize=9, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    verdict = "statistically significant" if ab.significant else "not significant"
    ax.set_title(f"95% CI excludes zero → difference is {verdict}" if ab.significant
                 else f"95% CI includes zero → difference is {verdict}",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    fig_caption(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App body
# ---------------------------------------------------------------------------
df = load_data()
sc = M.scorecard(df, k=NDCG_K)
seg_task = M.segment_breakdown(df, "task_type", k=NDCG_K)
seg_diff = M.segment_breakdown(df, "difficulty", k=NDCG_K)

# metrics for both models
ma = M.classification_metrics(df["y_true"], df["pred_a"])
mb = M.classification_metrics(df["y_true"], df["pred_b"])

with st.sidebar:
    st.markdown("### LLM Output Eval Scorecard")
    st.caption("Model A (incumbent) vs Model B (challenger)")
    view = st.radio(
        "View",
        ["1 · Scorecard", "2 · A/B Comparison", "3 · Error Analysis", "4 · So What"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"**{len(df):,}** labeled (query, response) pairs across "
        f"**{df['query_id'].nunique():,}** queries.\n\n"
        f"Decision threshold: **{THRESHOLD}**  ·  Ranking metric: **NDCG@{NDCG_K}**"
    )
    st.warning("All data is SYNTHETIC (see data_gen.py). Nothing here is real "
               "or proprietary.", icon="⚠️")
    st.caption("Leo Chan — github.com/GoldenCrate")


# ===========================  VIEW 1: SCORECARD  ===========================
if view.startswith("1"):
    st.title("LLM Output Evaluation Scorecard")
    st.markdown(
        "**Question:** should we replace our incumbent response scorer (**Model A**) "
        "with a challenger (**Model B**)? This scorecard judges both on the same "
        "labeled set — binary *accept/reject* quality plus *ranking* quality."
    )

    ndcg_a = M.mean_ndcg_at_k(df, "score_a", NDCG_K)
    ndcg_b = M.mean_ndcg_at_k(df, "score_b", NDCG_K)

    st.subheader("Headline metrics — Model B vs Model A")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{mb['accuracy']:.3f}",
              f"{(mb['accuracy']-ma['accuracy'])*100:+.1f} pp")
    c2.metric("Precision", f"{mb['precision']:.3f}",
              f"{(mb['precision']-ma['precision'])*100:+.1f} pp")
    c3.metric("Recall", f"{mb['recall']:.3f}",
              f"{(mb['recall']-ma['recall'])*100:+.1f} pp")
    c4.metric("F1", f"{mb['f1']:.3f}",
              f"{(mb['f1']-ma['f1'])*100:+.1f} pp")
    c5.metric(f"NDCG@{NDCG_K}", f"{ndcg_b:.3f}",
              f"{(ndcg_b-ndcg_a)*100:+.1f} pp")
    st.caption("Deltas show Model B minus Model A, in percentage points.")

    st.pyplot(chart_metric_bars(sc), width="stretch")

    with st.expander("See the raw metric table"):
        show = sc.copy()
        show["model_a"] = show["model_a"].round(4)
        show["model_b"] = show["model_b"].round(4)
        show["delta (B−A)"] = (show["model_b"] - show["model_a"]).round(4)
        st.dataframe(show, hide_index=True, width="stretch")


# =========================  VIEW 2: A/B COMPARISON  ========================
elif view.startswith("2"):
    st.title("Head-to-head A/B comparison")
    st.markdown(
        "We treat each model's **accept decisions that match ground truth** as a "
        "Bernoulli success, then run a **two-proportion z-test** on the two "
        "correctness rates. (Both models score the *same* items, so this is a "
        "conservative framing of a paired setup — it still answers: is B's "
        "lift more than noise?)"
    )

    succ_a = int((df["pred_a"] == df["y_true"]).sum())
    succ_b = int((df["pred_b"] == df["y_true"]).sum())
    n = len(df)
    ab = M.two_proportion_ztest(succ_a, n, succ_b, n)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model A correct-rate", f"{ab.rate_a:.4f}", f"{succ_a:,} / {n:,}")
    c2.metric("Model B correct-rate", f"{ab.rate_b:.4f}", f"{succ_b:,} / {n:,}")
    c3.metric("Absolute lift", f"{ab.abs_lift*100:+.2f} pp",
              f"{ab.rel_lift*100:+.2f}% relative")
    c4.metric("p-value (two-sided)", f"{ab.p_value:.4g}",
              "significant" if ab.significant else "not significant",
              delta_color="normal" if ab.significant else "off")

    st.pyplot(chart_ab(ab), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Test details**")
        st.table(pd.DataFrame({
            "quantity": ["z-statistic", "p-value (two-sided)",
                         "95% CI low (B−A)", "95% CI high (B−A)",
                         "n per model", "α"],
            "value": [f"{ab.z:.3f}", f"{ab.p_value:.4g}",
                      f"{ab.ci_low*100:+.3f} pp", f"{ab.ci_high*100:+.3f} pp",
                      f"{n:,}", "0.05"],
        }))
    with right:
        st.markdown("**How to read it**")
        if ab.significant:
            st.success(
                f"The 95% CI for B−A is "
                f"[{ab.ci_low*100:+.2f}, {ab.ci_high*100:+.2f}] pp and **excludes 0**, "
                f"with p = {ab.p_value:.4g} < 0.05. We reject the null that the two "
                f"models are equally correct: **Model B is reliably better** on "
                f"overall correctness.", icon="✅")
        else:
            st.info(
                f"The 95% CI for B−A includes 0 (p = {ab.p_value:.4g}). We cannot "
                f"conclude a real difference in overall correctness from this sample.",
                icon="ℹ️")
        st.caption(
            "z-test & p-value use the pooled proportion; the CI on the difference "
            "uses the unpooled standard error — the conventional A/B-testing setup.")


# =========================  VIEW 3: ERROR ANALYSIS  ========================
elif view.startswith("3"):
    st.title("Error analysis")
    st.markdown(
        "Where do the two models disagree with ground truth, and **who is affected**? "
        "Aggregate accuracy hides segment-level regressions — this view surfaces them."
    )

    st.subheader("Confusion matrices")
    st.caption("Positive class = response is *acceptable to show*. "
               "Red cells are the two error types.")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.pyplot(chart_confusion(df["y_true"], df["pred_a"],
                                  "Model A (incumbent)", GREY),
                  width="stretch")
    with cc2:
        st.pyplot(chart_confusion(df["y_true"], df["pred_b"],
                                  "Model B (challenger)", ACCENT),
                  width="stretch")

    st.divider()
    st.subheader("Breakdown by task type")
    st.pyplot(chart_segment_delta(seg_task, "task type"), width="stretch")

    worst = seg_task.iloc[0]
    if worst["acc_delta"] < 0:
        st.warning(
            f"**Watch-out:** on **{worst['segment']}** tasks, Model B is "
            f"**{abs(worst['acc_delta'])*100:.1f} pp worse** than Model A "
            f"({worst['n_items']:,} items). Every other segment improves.",
            icon="⚠️")

    show = seg_task.copy()
    for col in ["acc_a", "acc_b", "acc_delta", "f1_a", "f1_b",
                "ndcg_a", "ndcg_b", "ndcg_delta"]:
        show[col] = show[col].round(3)
    st.dataframe(show, hide_index=True, width="stretch")

    st.subheader("Breakdown by difficulty")
    showd = seg_diff.copy()
    for col in ["acc_a", "acc_b", "acc_delta", "f1_a", "f1_b",
                "ndcg_a", "ndcg_b", "ndcg_delta"]:
        showd[col] = showd[col].round(3)
    st.dataframe(showd, hide_index=True, width="stretch")


# ============================  VIEW 4: SO WHAT  ============================
else:
    st.title("So what — recommendation")

    ndcg_a = M.mean_ndcg_at_k(df, "score_a", NDCG_K)
    ndcg_b = M.mean_ndcg_at_k(df, "score_b", NDCG_K)
    succ_a = int((df["pred_a"] == df["y_true"]).sum())
    succ_b = int((df["pred_b"] == df["y_true"]).sum())
    n = len(df)
    ab = M.two_proportion_ztest(succ_a, n, succ_b, n)
    worst = seg_task.iloc[0]

    st.markdown("### Recommendation: **ship Model B — with a guardrail**")
    st.markdown(
        f"""
Model B is the better scorer on this labeled set and the gain is **statistically
real**, not noise:

- **Accuracy** rises from **{ma['accuracy']:.3f} → {mb['accuracy']:.3f}**
  ({(mb['accuracy']-ma['accuracy'])*100:+.1f} pp).
- **F1** rises **{(mb['f1']-ma['f1'])*100:+.1f} pp** and
  **NDCG@{NDCG_K}** (ranking quality) rises **{(ndcg_b-ndcg_a)*100:+.1f} pp**,
  so the best response surfaces higher more often.
- The overall correctness lift of **{ab.abs_lift*100:+.2f} pp** has
  **p = {ab.p_value:.4g}** with a 95% CI of
  **[{ab.ci_low*100:+.2f}, {ab.ci_high*100:+.2f}] pp** — the interval excludes zero.
"""
    )

    if worst["acc_delta"] < 0:
        st.markdown(
            f"""
**The guardrail.** B is **not** uniformly better. On **{worst['segment']}**
tasks it is **{abs(worst['acc_delta'])*100:.1f} pp worse**
({worst['n_items']:,} items). A naive full cutover would quietly degrade that
segment.

**Proposed rollout**
1. Route **{worst['segment']}** traffic to **Model A** (or an ensemble) until B's
   regression there is fixed.
2. Ship **Model B** for all other task types, where it is clearly better.
3. Monitor segment-level accuracy weekly; re-run this z-test on live labels before
   removing the guardrail.
"""
        )

    st.divider()
    cols = st.columns(3)
    cols[0].metric("Overall accuracy lift", f"{(mb['accuracy']-ma['accuracy'])*100:+.1f} pp")
    cols[1].metric(f"NDCG@{NDCG_K} lift", f"{(ndcg_b-ndcg_a)*100:+.1f} pp")
    cols[2].metric("Decision", "Ship B + guardrail")

    st.caption(
        "Framework note: metrics chosen to match the decision — accept/reject is a "
        "classification problem (accuracy/precision/recall/F1), surfacing the best "
        "answer is a ranking problem (NDCG@k), and 'is the lift real?' is an "
        "inference problem (two-proportion z-test). All data is synthetic.")
