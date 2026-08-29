# Repository Index & File Map

## 1. Primary Documentation & Executive Reports
* [FINAL_REPORT.md](FINAL_REPORT.md) — **Master Comprehensive Research Report** (Methodology, verified findings, forensic failure analysis, and research closure).
* [INDEX.md](INDEX.md) — Full repository navigation map and file directory (this document).
* [model_spec.md](model_spec.md) — Mathematical specification of the hierarchical model, priors, and fold structure.
* [results/reconciliation_log.md](results/reconciliation_log.md) — Step-by-step numeric verification log matching all claims directly to source files.

---

## 2. Interactive Analytical Dashboards (HTML)
All dashboards are standalone, self-contained HTML applications viewable directly in browser (or served via local server):

1. **[active_filter_slots_matrix_dashboard.html](active_filter_slots_matrix_dashboard.html)**
   * Multi-asset switcher (`NQ`, `ES`, `CL`, `FDAX`) focusing on the 5 rescued/profitable slots, loss reduction slots, and news overlap windows.
2. **[nq_active_filters_matrix_dashboard.html](nq_active_filters_matrix_dashboard.html)**
   * 100% isolated NQ (Nasdaq 100) dashboard comparing all 7 NQ candidate slots across Macro, VIX, VWAP, and Combo filters with 4 side-by-side matrices.
3. **[full_dataset_matrix_dashboard.html](full_dataset_matrix_dashboard.html)**
   * Complete 2.5-year history (no In/Out sample split, 2.9M trades) comparing Raw vs. Macro vs. VIX vs. ATR filters side-by-side.
4. **[macro_news_matrix_dashboard.html](macro_news_matrix_dashboard.html)**
   * Dedicated macro news blackout matrix (±30 min window) with 4 side-by-side panels per asset (`IS Raw`, `IS + Macro`, `OOS Raw`, `OOS + Macro`).
5. **[vix_matrix_dashboard.html](vix_matrix_dashboard.html)**
   * Volatility regime matrix filtering out high-volatility sessions (VIX(t-1) >= 20).
6. **[atr_matrix_dashboard.html](atr_matrix_dashboard.html)**
   * Asset-specific 14-day normalized ATR volatility regime matrix.
7. **[three_filters_dashboard.html](three_filters_dashboard.html)**
   * Portfolio-level decomposition of Macro News, VIX Regime, and VWAP Trend filters.

---

## 3. Core Pipeline Scripts (`C:/Model-/`)
1. [00_data_prep.py](00_data_prep.py) — Database cleaning, ghost-fill resolution, and slot indexing.
2. [01_fold_structure.py](01_fold_structure.py) — 5-fold cross-validation partition & 1-year holdout assignment.
3. [02_fit_model.py](02_fit_model.py) — Hierarchical Bayesian model fitting via NumPyro NUTS.
4. [03_select_candidates.py](03_select_candidates.py) — Benjamini-Hochberg FDR candidate slot selection (alpha=0.05).
5. [04_permutation_null.py](04_permutation_null.py) — 1,000-run permutation null distribution testing.
6. [05_feature_inventory.py](05_feature_inventory.py) — Temporal and market feature metadata extraction.
7. [06_holdout_eval.py](06_holdout_eval.py) — Static 1-year out-of-sample holdout evaluation.
8. [07_report.py](07_report.py) — Diagnostic output generation and performance metrics compilation.
9. [08_rolling_retrain.py](08_rolling_retrain.py) — 25-cycle expanding/rolling window retrain and forward holdout evaluation.

---

## 4. Key Data Files (`data/`)
* `data/trades_clean.parquet` — 2,832,740 cleaned trade records with slot identifiers.
* `data/fold_assignments.parquet` — In-sample CV fold and holdout split flags per trade.
* `data/slot_index.parquet` — Mapping of 960 candidate slots (Asset x Day x 30-min Bucket).
* `data/macro_blackout_windows.parquet` — 14,000+ macroeconomic news blackout windows (±30 min).
* `data/vix_daily.parquet` — Daily VIX closing prices and lagged t-1 values.
* `data/vwap_hourly.parquet` — Hourly session volume-weighted average prices.

---

## 5. Key Results Files (`results/`)
* `results/walkforward_portfolio_monthly.csv` — Monthly static holdout performance (-$1,528,785.01 total).
* `results/rolling_retrain_chain_monthly.csv` — Monthly chained rolling retrain performance (-$2,569,959.98 total).
* `results/three_filters_full_comparison.csv` — Bootstrap & monthly summary of the 3 conditioning filters.
* `results/gross_vs_net_21slot.csv` — Gross vs. Net friction audit across all 21 candidate slots.
* `results/bh_fdr_candidates.csv` — List of 21 Benjamini-Hochberg candidate slots.
* `results/candidate_slots_trades_sample.csv` — Master combined trade record export for active candidate slots.
* `results/trades_by_slot/` — Individual trade record CSVs for key candidate slots.
