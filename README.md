# SC_results_WF — Futures Quantitative Strategy Research & Audit

**Executive Summary:** This repository contains the complete quantitative research, Hierarchical Bayesian modeling, permutation null testing, 25-cycle walk-forward evaluation, and conditioning filter audit for the **Calendar-Slot Reversal Strategy** across 2,832,740 trade records in NQ, ES, CL, and FDAX.

> **Primary Report:** See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the complete comprehensive research report, methodology, verified numbers, and forensic root-cause failure analysis.  
> **Repository Index:** See [`INDEX.md`](INDEX.md) for the full file and directory navigation map.  
> **Numerical Reconciliation:** See [`results/reconciliation_log.md`](results/reconciliation_log.md) for line-by-line verification against raw source files.

---

## Key Verified Results Summary

* **Static System 1-Year Holdout (Jul 2025 – Jul 2026):** Realized net PnL of **`-$1,528,785.01`** across 59,810 trades (`-$27.16` / trade).
* **25-Cycle Rolling-Retrain Chained OOS:** Realized net PnL of **`-$2,569,959.98`** across 25 forward monthly cycles (`-$34.67` / trade).
* **Conditioning Filters (Macro News, VIX, VWAP, ATR):** None turned the portfolio out-of-sample expectancy positive (`-$25.14` to `-$34.07` / trade).
* **Forensic Verdict:** The calendar-slot mean-reversion signal does not carry durable out-of-sample edge. Slippage, news shocks, and volatility regimes were systematically ruled out as causal drivers.

---

## Interactive Visual Dashboards (HTML)

All dashboards are standalone, self-contained interactive web applications (viewable directly or via `http://127.0.0.1:7799/`):

1. **[`active_filter_slots_matrix_dashboard.html`](active_filter_slots_matrix_dashboard.html)** — Multi-asset switcher (`NQ`, `ES`, `CL`, `FDAX`) featuring the 5 rescued/profitable slots, loss-reduction slots, and news overlap windows.
2. **[`nq_active_filters_matrix_dashboard.html`](nq_active_filters_matrix_dashboard.html)** — 100% isolated NQ (Nasdaq 100) dashboard comparing all 7 NQ candidate slots across Macro, VIX, VWAP, and Combo filters.
3. **[`full_dataset_matrix_dashboard.html`](full_dataset_matrix_dashboard.html)** — Continuous 2.5-year history (no In/Out sample split, 2.9M trades) comparing Raw vs. Macro vs. VIX vs. ATR filters.
4. **[`macro_news_matrix_dashboard.html`](macro_news_matrix_dashboard.html)** — Macro news blackout matrix (±30 min window, IS vs OOS).
5. **[`vix_matrix_dashboard.html`](vix_matrix_dashboard.html)** — VIX volatility regime matrix (IS vs OOS).
6. **[`atr_matrix_dashboard.html`](atr_matrix_dashboard.html)** — Asset-specific 14-day ATR volatility regime matrix.
7. **[`three_filters_dashboard.html`](three_filters_dashboard.html)** — Full portfolio-level 3-filter dashboard.

---

## Quick Navigation

```
├── FINAL_REPORT.md             # Master Research & Audit Report
├── INDEX.md                    # Complete Directory & File Navigation Index
├── model_spec.md               # Mathematical Specification & Priors
├── 00_data_prep.py ... 08_...  # 9 Core Pipeline Execution Scripts
├── active_filter_...html       # Interactive Visual Dashboards (7 Total)
├── data/                       # Preprocessed Parquet Datasets
└── results/                    # CSVs, Posteriors, Logs, and Reconciliation Data
```
