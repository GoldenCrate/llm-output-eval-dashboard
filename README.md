# LLM Output Evaluation Scorecard

A small, honest Streamlit app that answers a realistic data-science question:
**should we replace an incumbent LLM response scorer ("Model A") with a
challenger ("Model B")?** It judges both models on the same labeled dataset
across three lenses — binary *accept/reject* quality (classification), *ranking*
quality (NDCG@k), and *statistical inference* (is the lift real?) — and ends with
a plain-English recommendation.

> **Synthetic data.** Every number in this app comes from `data_gen.py`, which
> generates a fully synthetic labeled dataset from a fixed random seed. It is
> **not** real model output and contains **no** proprietary or customer data.
> The scenario is designed to be realistic (and deliberately nuanced) so the
> analysis has something substantive to chew on.

## What each view shows

1. **Scorecard** — headline KPI tiles (accuracy, precision, recall, F1, and
   **NDCG@3** for ranking) comparing Model B to Model A, plus a decluttered
   grouped-bar chart and the raw metric table.
2. **A/B Comparison** — a **two-proportion z-test** (SciPy) on the two models'
   overall correctness rates, reporting absolute & relative **lift**, the
   **p-value**, and a **95% confidence interval** on the difference, visualized
   as a point-estimate-with-CI plot.
3. **Error Analysis** — side-by-side **confusion matrices** plus a per-**segment**
   breakdown (by task type and by difficulty) that surfaces where Model B
   *regresses* — aggregate accuracy hides it.
4. **So What** — a direct recommendation ("ship Model B, with a guardrail") that
   ties each metric back to the decision it informs.

## Analysis / methods notes

- **Classification metrics** (`metrics.py`) are computed from first principles off
  the confusion counts, so accuracy/precision/recall/F1 are auditable.
- **NDCG@k** ranks each query's candidate responses by model score and compares to
  the ideal ordering, using gain `2^rel − 1`.
- **Two-proportion z-test**: the z-statistic and p-value use the *pooled*
  proportion (the standard test of H0: p_A = p_B); the 95% CI on the difference
  uses the *unpooled* standard error (the conventional CI for a difference of
  proportions).
- **Storytelling with Data**: every chart title states the *takeaway*, not the
  mechanics; clutter (gridlines, extra spines, redundant ticks) is stripped;
  series are directly labeled; color is restrained to one neutral (incumbent) +
  one accent (challenger), with a reserved red/green only for regressions/gains.

## Run it

```bash
pip install -r requirements.txt
python data_gen.py        # writes the synthetic eval_data.csv (optional; the app
                          # regenerates in-memory if the CSV is absent)
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Multi-view Streamlit app (4 views) |
| `data_gen.py` | Synthetic labeled-data generator (fixed seed) |
| `metrics.py` | Pure, testable compute functions (metrics, z-test, NDCG) |
| `requirements.txt` | Dependencies |
| `README.md` | This file |

---

Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)
