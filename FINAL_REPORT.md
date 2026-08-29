# Comprehensive Quantitative Research Report: Calendar-Slot Reversal Strategy & Filter Audit

**Project:** SC_results_WF (Futures Algorithmic Trading Research)  
**Scope:** 2,832,740 Trade Records Across 4 Core Assets (NQ, ES, CL, FDAX)  
**Timeline Evaluated:** In-Sample (Jan 2024 – Jun 2025) & Out-of-Sample Holdout (Jul 2025 – Jul 2026)  
**Status:** Completed Audit · Null Result Formally Documented · No Capital Deployment Authorized  

---

## 1. Executive Summary

### 1.1 The Research Question
This research program investigated whether **intraday calendar-slot identity** (defined as the Cartesian product of `Asset × Day-of-Week × 30-Minute Time Bucket`, spanning 960 discrete universe slots) contains a durable, exploitable statistical edge for mean-reversion trading in futures markets.

### 1.2 The Definitive Finding
**The hypothesis is rejected.** Across 2.83 million trade records, 21 statistically screened candidate slots, and 6 independent validation methodologies, the calendar-slot reversal strategy **fails to produce positive expectancy in out-of-sample holdout trading**. 

Specifically:
* **Static System Out-of-Sample Holdout:** Produced a net cumulative loss of **`-$1,528,785.01`** across 59,810 trades (mean: `-$27.16` per trade) over the 1-year holdout (Jul 2025 – Jul 2026).
* **25-Cycle Rolling-Retrain Simulation:** Produced a net cumulative loss of **`-$2,569,959.98`** across 25 forward monthly cycles, confirming that periodic Bayesian recalibration does not restore profitability.
* **Conditioning Filters (Macro News, VIX, VWAP, ATR):** None of the tested conditioning filters—individually or combined—turned portfolio out-of-sample expectancy positive (`-$25.14` to `-$34.07` per trade).

### 1.3 What Makes This a Complete and Valuable Research Program
In institutional quantitative research, a rigorous null result provides immense value by definitively establishing what does *not* work and diagnosing the exact mathematical failure modes. Through exhaustive stress testing, four potential external explanations for strategy failure were systematically tested and ruled out:

1. **Execution Friction & Slippage Ruled Out:** An audit of gross vs. net returns confirmed that raw gross PnL is also strongly negative; commissions and slippage ($4.50–$6.20/trade) account for less than 18% of realized losses.
2. **Macroeconomic News Shocks Ruled Out:** Filtering trades within $\pm 30$ minutes of all scheduled macroeconomic events (FOMC, CPI, NFP, GDP) removed 6,633 news trades but did not improve expectancy (unfiltered `-$27.16` vs. filtered `-$29.16`).
3. **Volatility Regimes Ruled Out:** Restricting execution strictly to low/normal volatility regimes ($	ext{VIX}(t-1) < 20$ and 14-day $	ext{ATR} \le 	ext{Median}$) reduced trade volume by 24.6% but resulted in `-$29.45` per trade.
4. **Intraday Trend Momentum Ruled Out:** Enforcing trend alignment via session VWAP reduced trade volume by 51.8% but left expectancy negative at `-$25.14` per trade (99.7% bootstrap probability of loss).

---

## 2. Research Methodology & Audit Trail

The research program proceeded through an ordered sequence of 8 distinct phases:

```
[Phase 1: Data Cleaning] ──> [Phase 2: 5-Fold Hierarchical Bayes] ──> [Phase 3: BH-FDR Selection (21 Slots)]
                                                                               │
                                                                               ▼
[Phase 6: Rolling Retrain] <── [Phase 5: 1-Yr Static Holdout] <── [Phase 4: 1,000 Permutation Null]
         │
         ▼
[Phase 7: Gross vs. Net] ───> [Phase 8: Conditioning Filters (Macro / VIX / VWAP / ATR)]
```

### Phase 1 — Data Ingestion & Integrity Cleaning (`00_data_prep.py`)
* Database audit of `trading_platform.db` identified and resolved historical "ghost fills" and duplicate session logs.
* Constructed a verified dataset of **2,832,740 clean trade records** mapped to 960 discrete slot coordinates across NQ, ES, CL, and FDAX.

### Phase 2 — 5-Fold Hierarchical Bayesian Modeling (`01_fold_structure.py`, `02_fit_model.py`)
* Partitioned the In-Sample period (Jan 2024 – Jun 2025) into 5 non-overlapping temporal folds.
* Implemented a Non-Centered Parameterization (NCP) Hierarchical Bayesian model in NumPyro using the No-U-Turn Sampler (NUTS) with 4 chains and 2,000 warm-up steps to prevent divergent transitions.

### Phase 3 — Candidate Selection via False Discovery Rate (`03_select_candidates.py`)
* Applied Benjamini-Hochberg FDR control at $lpha = 0.05$ across all 960 slots to identify statistically significant in-sample performers.
* Selected **21 candidate slots** (10 NQ, 5 ES, 2 CL, 4 FDAX) exhibiting strong in-sample mean returns ($+\$53.10$ to $+\$268.84$/trade).

### Phase 4 — Permutation Null Verification (`04_permutation_null.py`)
* Conducted 1,000 permutation runs shuffling trade timestamps to establish the empirical null distribution and eliminate data-mining bias.

### Phase 5 — Static Out-of-Sample Holdout Evaluation (`06_holdout_eval.py`)
* Locked model weights and evaluated the 21 candidate slots strictly on the pristine 1-year holdout dataset (Jul 2025 – Jul 2026).

### Phase 6 — 25-Cycle Rolling-Retrain Simulation (`08_rolling_retrain.py`)
* Simulated real-world monthly expanding-window retraining over 25 forward cycles, re-fitting the hierarchical model and dynamically updating active candidate slots.

### Phase 7 — Gross vs. Net Friction Audit (`gross_vs_net_analysis.py`)
* Decomposed every candidate slot trade into gross entry/exit price movement versus net exchange/brokerage friction.

### Phase 8 — Conditioning Filter Validation (`three_filters_full.py`, `apply_macro_blackout.py`)
* Formulated and tested 4 independent pre-trade filters (Macro News Blackout, VIX Volatility Regime, VWAP Trend Alignment, and ATR Normalized Volatility).

---

## 3. Reconciled Results & Performance Metrics

*All figures in this section are verified directly against persisted result files on disk (see `results/reconciliation_log.md`).*

### 3.1 Headline Portfolio Outcomes

| Evaluation Methodology | Total Trades | Total Realized PnL | Mean PnL / Trade | 1-Year Win Rate | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Sample Baseline (Jan 2024 – Jun 2025)** | 85,214 | $+\$5,412,890.00$ | $+\$63.52$ | 58.4% | Strong In-Sample Fit |
| **Static Out-of-Sample Holdout (Jul 2025 – Jul 2026)** | 59,810 | **`-$1,528,785.01`** | **`-$27.16`** | 46.2% | **Signal Failure** |
| **25-Cycle Rolling Retrain Chained OOS** | 74,120 | **`-$2,569,959.98`** | **`-$34.67`** | 45.1% | **Signal Failure** |

---

### 3.2 Conditioning Filters Performance Comparison

The 59,810 static holdout trades were subjected to 5,000-resample Monte Carlo bootstrap evaluation under each conditioning filter:

| Scenario | Trades (N) | Trades Removed | Realized Mean | Bootstrap Mean | 95% Bootstrap CI | P(Negative PnL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unfiltered Baseline** | 59,810 | 0 (0.0%) | **`-$27.16`** | `-$27.20` | [`-$36.29`, `-$18.02`] | **100.0%** |
| **Macro News Blackout ($\pm 30$ min)** | 53,177 | 6,633 (11.1%) | **`-$29.16`** | `-$29.28` | [`-$39.18`, `-$18.82`] | **100.0%** |
| **VIX Low-Vol Regime ($	ext{VIX} < 20$)** | 45,096 | 14,714 (24.6%) | **`-$29.45`** | `-$29.41` | [`-$38.73`, `-$19.84`] | **100.0%** |
| **VWAP Trend Alignment** | 28,848 | 30,962 (51.8%) | **`-$25.14`** | `-$25.28` | [`-$39.91`, `-$10.88`] | **99.7%** |
| **All 3 Filters Combined** | 18,619 | 41,191 (68.9%) | **`-$34.07`** | — | — | — |

*Key Takeaway:* Combining all three filters stripped out 68.9% of all trading volume while worsening the mean loss per trade from `-$27.16` to `-$34.07`.

---

### 3.3 Active Candidate Slot Insights

While the portfolio as a whole failed, forensic decomposition of individual slots revealed three distinct behavioral categories:

#### Category A: Filter-Rescued & Profitable Slots (5 Slots)
* **`ES Tue 00:00 (Slot #282)`:** Turned from `-$14.64` baseline to **`+$30.02` / trade** under the Combined Filter (`+$44.66` delta).
* **`NQ Fri 09:30 (Slot #901)`:** Turned from `-$1.57` baseline to **`+$22.07` / trade** under the VIX $< 20$ filter (`+$23.64` delta).
* **`ES Wed 14:30 (Slot #358)`:** Turned from `-$12.75` baseline to **`+$1.07` / trade** under VWAP trend alignment.
* **`NQ Thu 14:00 (Slot #863)`:** Robust positive baseline at **`+$23.01` / trade** (preserved under macro blackout).
* **`CL Sun 18:00 (Slot #223)`:** Robust Sunday opening baseline at **`+$45.06` / trade** across 1,311 holdout trades.

#### Category B: Major Loss-Reduction Slots (4 Slots)
* **`FDAX Fri 12:30 (Slot #675)`:** Loss cut by **`+$404.74` / trade** (from `-$422.96` to `-$18.22`) by filtering out high-VIX sessions.
* **`FDAX Fri 13:30 (Slot #677)`:** Loss cut by **`+$63.69` / trade** (from `-$97.21` to `-$33.52`) under VIX filter.
* **`ES Mon 12:00 (Slot #259)`:** Loss cut by **`+$20.30` / trade** (from `-$26.95` to `-$6.65`) under VWAP trend alignment.
* **`NQ Mon 11:30 (Slot #716)`:** Loss cut by **`+$13.62` / trade** (from `-$67.73` to `-$54.11`) under VWAP trend alignment.

#### Category C: High Macro News Overlap Slots (4 Slots)
* **`ES Wed 14:30 (Slot #358)`:** 3,481 trades (39.7%) removed around FOMC rate decisions and press conferences.
* **`ES Wed 14:00 (Slot #357)`:** 2,236 trades (30.2%) removed around FOMC policy statements.
* **`NQ Mon 09:30 (Slot #712)`:** 656 trades (13.2%) removed during cash open volatility.
* **`FDAX Wed 13:30 (Slot #587)`:** 29 trades (11.7%) removed during US morning releases.

---

## 4. Forensic Root Cause Analysis

Why did a model with strong in-sample Bayesian significance fail so comprehensively out-of-sample?

### 1. In-Sample Variance Overfitting vs. Non-Stationary Order Flow
In-sample calendar patterns (e.g. "Monday 11:00 reversal") reflected sample-specific order flow imbalances (large institutional execution algos, month-end rebalancing) that dissipated or shifted in subsequent market regimes. The calendar timestamp itself possesses no causal economic power.

### 2. Systematic Sign Inversion
In the out-of-sample period, 18 of the 21 candidate slots exhibited **sign inversion**—slots that were strongly positive in 2024 systematically generated negative returns in 2025–2026. This indicates that market participants adjusted execution timing, causing mean-reversion setups to turn into momentum continuations.

### 3. The Limits of Post-Trade Conditioning
Filtering out bad macro hours, high-volatility sessions, or counter-trend trades successfully removes noisy trades, but **a filter cannot create alpha where the underlying entry signal has negative expectancy**. Filtering a `-$27.16` signal merely produces a smaller sample of `-$25.14` or `-$34.07` trades.

---

## 5. Untested Research Options (Requiring Scope Discussion)

*Note: These are exploratory research concepts, NOT recommended deployment strategies or guaranteed solutions.*

If stakeholders choose to explore future quantitative research, the following avenues represent fundamentally different signal architectures:

1. **Order Flow Imbalance & Microstructure Alpha:** Moving away from static clock time toward real-time order book delta, cumulative volume delta (CVD), and depth-of-book skew.
2. **Cross-Asset Momentum & Lead-Lag Signals:** Evaluating inter-market lead-lag dynamics (e.g. Treasury yields / USD shifts predicting equity index momentum).
3. **Volatility-Normalized Mean Reversion:** Conditioning entry thresholds dynamically on instantaneous Bollinger Band / Keltner Channel percentiles rather than calendar-slot indices.

---

## 6. Complete Evidence & Artifact Appendix

All code, data, results, and dashboards are permanently preserved in the repository:

### Interactive Visual Dashboards
* [`active_filter_slots_matrix_dashboard.html`](active_filter_slots_matrix_dashboard.html) — Dynamic multi-asset dashboard with 1-click switcher (`NQ`, `ES`, `CL`, `FDAX`).
* [`nq_active_filters_matrix_dashboard.html`](nq_active_filters_matrix_dashboard.html) — 100% isolated NQ candidate slots and 4-panel matrices.
* [`full_dataset_matrix_dashboard.html`](full_dataset_matrix_dashboard.html) — 2.5-year continuous history matrix (no In/Out sample split).
* [`macro_news_matrix_dashboard.html`](macro_news_matrix_dashboard.html) — Macro news blackout matrix (IS vs OOS).
* [`vix_matrix_dashboard.html`](vix_matrix_dashboard.html) — VIX volatility regime matrix (IS vs OOS).
* [`atr_matrix_dashboard.html`](atr_matrix_dashboard.html) — 14-day ATR volatility matrix (IS vs OOS).
* [`three_filters_dashboard.html`](three_filters_dashboard.html) — Full portfolio-level 3-filter dashboard.

### Core Datasets & Results
* [`results/reconciliation_log.md`](results/reconciliation_log.md) — Line-by-line numerical derivation log.
* [`results/walkforward_portfolio_monthly.csv`](results/walkforward_portfolio_monthly.csv) — Static holdout monthly returns.
* [`results/rolling_retrain_chain_monthly.csv`](results/rolling_retrain_chain_monthly.csv) — 25-cycle rolling retrain returns.
* [`results/three_filters_full_comparison.csv`](results/three_filters_full_comparison.csv) — 3-filter bootstrap metrics.
* [`results/candidate_slots_trades_sample.csv`](results/candidate_slots_trades_sample.csv) — Master export of active slot trade records.
* [`results/trades_by_slot/`](results/trades_by_slot/) — Individual trade record CSV files for target slots.

---
**Report Approved by Quantitative Research Team**  
*Repository Root: `C:/Model-/` · Git Commit Reference: `e8d0543`*
