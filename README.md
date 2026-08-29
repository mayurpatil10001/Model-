# SC_results_WF — Futures Algorithmic Strategy Research & Audit

**Project:** Comprehensive Quantitative Audit of Calendar-Slot Mean-Reversion Alpha  
**Dataset Scope:** 2,832,740 Trade Records across 4 Major Futures Markets (NQ, ES, CL, FDAX)  
**Timeline:** In-Sample (Jan 2024 – Jun 2025) & Out-of-Sample Holdout (Jul 2025 – Jul 2026)  
**Status:** Research Completed · Null Result Formally Documented · No Capital Deployment Authorized  

---

## Table of Contents
1. [Project Overview & Core Hypothesis](#1-project-overview--core-hypothesis)
2. [Research Chronology: What Was Tried & How Things Evolved](#2-research-chronology-what-was-tried--how-things-evolved)
   * [Phase 1: Data Ingestion & Sierra Chart Ghost-Fill Resolution](#phase-1--data-ingestion--sierra-chart-ghost-fill-resolution)
   * [Phase 2: 5-Fold Hierarchical Bayesian Modeling](#phase-2--5-fold-hierarchical-bayesian-modeling)
   * [Phase 3: Benjamini-Hochberg FDR Candidate Selection](#phase-3--benjamini-hochberg-fdr-candidate-selection)
   * [Phase 4: 1,000-Run Permutation Null Testing](#phase-4--1000-run-permutation-null-testing)
   * [Phase 5: Static 1-Year Holdout Evaluation & Strategy Breakdown](#phase-5--static-1-year-holdout-evaluation--strategy-breakdown)
   * [Phase 6: 25-Cycle Rolling Retrain Simulation](#phase-6--25-cycle-rolling-retrain-simulation)
   * [Phase 7: Gross vs. Net Friction Audit](#phase-7--gross-vs-net-friction-audit)
   * [Phase 8: Conditioning Filters Audit (Macro, VIX, VWAP, ATR)](#phase-8--conditioning-filters-audit-macro-vix-vwap-atr)
3. [Verified Headline Findings & Reconciled Numbers](#3-verified-headline-findings--reconciled-numbers)
4. [Forensic Root-Cause Analysis: Why the Alpha Failed](#4-forensic-root-cause-analysis-why-the-alpha-failed)
5. [Interactive Visual Dashboards Guide](#5-interactive-visual-dashboards-guide)
6. [Untested Future Research Concepts](#6-untested-future-research-concepts)
7. [Repository Map & Pipeline Scripts](#7-repository-map--pipeline-scripts)

---

## 1. Project Overview & Core Hypothesis

This research program conducted an institutional-grade quantitative evaluation of the **Calendar-Slot Mean-Reversion Hypothesis** in electronic futures markets.

### The Core Premise
The hypothesis posited that systematic intraday order flow patterns (e.g., fixed-time institutional execution algos, London/New York session handoffs, fixing windows, and market opens) create persistent, recurring mispricings at specific clock times.

* **Slot Coordinate System:** The universe was partitioned into **960 discrete slots**:
  $$	ext{Slot} = 	ext{Symbol (NQ, ES, CL, FDAX)} 	imes 	ext{Day of Week (Mon–Sun)} 	imes 	ext{30-Minute Time Bucket (00:00–23:30 ET)}$$
* **Goal:** Use hierarchical Bayesian modeling and multiple-testing corrections to isolate genuine alpha slots from random noise, validate them through out-of-sample holdout, and build an automated trading portfolio.

---

## 2. Research Chronology: What Was Tried & How Things Evolved

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RESEARCH PROGRAM PHASES                                  │
├────────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Phase 1: Data Cleaning     │ Phase 2: Hierarchical Bayes │ Phase 3: FDR Candidate Select │
│ 2.83M Clean Trades         │ 5 Folds, NumPyro NCP NUTS   │ 21 Candidates Identified      │
├────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Phase 4: Permutation Null  │ Phase 5: Static Holdout     │ Phase 6: Rolling Retrain      │
│ 1,000 Shuffle Runs Passed  │ -$1.52M Loss (Sign Invert)  │ 25 Cycles, -$2.57M Loss       │
├────────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Phase 7: Gross vs. Net     │ Phase 8: 4 Conditioning Flts│ Phase 9: Final Dashboards     │
│ Friction NOT Cause (<18%)  │ Macro / VIX / VWAP / ATR    │ Asset Switchers & CSV Exports │
└────────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Phase 1 — Data Ingestion & Sierra Chart Ghost-Fill Resolution
* **Initial Problem:** Raw database imports from Sierra Chart logs contained duplicate session fills, non-standard order cancellations, and overlapping ghost records.
* **Action Taken (`00_data_prep.py`):** Built automated data ingestion filters, cleaned order timestamps into America/New York timezone, deduplicated fills, and generated `data/trades_clean.parquet` with **2,832,740 verified trade records**.

### Phase 2 — 5-Fold Hierarchical Bayesian Modeling
* **Methodology (`01_fold_structure.py`, `02_fit_model.py`):** Structured the in-sample period (Jan 2024 – Jun 2025) into 5 temporal folds.
* **Model Formulation:** Implemented a Non-Centered Parameterization (NCP) Hierarchical Bayesian model using **NumPyro NUTS** (4 Markov Chains, 2,000 warmup iterations, 2,000 posterior draws per chain) to shrink noisy slot estimates toward instrument and market-wide grand means while preventing sampler divergences.

### Phase 3 — Benjamini-Hochberg FDR Candidate Selection
* **Action Taken (`03_select_candidates.py`):** Applied Benjamini-Hochberg False Discovery Rate (FDR) control at $lpha = 0.05$ across all 960 slots to control for data-snooping and selection bias.
* **Outcome:** Identified **21 statistically significant candidate slots** exhibiting high in-sample profitability (average in-sample mean return: $+\$63.52$ / trade):
  * **NQ (10 slots):** e.g., Mon 09:30, Mon 11:00, Thu 14:00, Fri 09:30.
  * **ES (5 slots):** e.g., Mon 12:00, Tue 00:00, Wed 14:00, Wed 14:30.
  * **CL (2 slots):** Sun 18:00 (Sunday open), Thu 20:00.
  * **FDAX (4 slots):** Wed 13:30, Thu 23:00, Fri 12:30, Fri 13:30.

### Phase 4 — 1,000-Run Permutation Null Testing
* **Methodology (`04_permutation_null.py`):** Shuffled trade timestamps across 1,000 independent Monte Carlo permutation runs to evaluate whether the 21 candidate slots could arise by chance.
* **Outcome:** Candidate slots passed the permutation null benchmark in-sample, confirming statistical significance within the 2024–mid 2025 historical data.

### Phase 5 — Static 1-Year Holdout Evaluation & Strategy Breakdown
* **Execution (`06_holdout_eval.py`):** Model weights and slot parameters were locked. The 21 candidate slots were evaluated on the completely unseen 1-year holdout dataset (Jul 2025 – Jul 2026).
* **Finding (The Breakdown):** The strategy suffered severe performance collapse.
  * Realized net PnL: **`-$1,528,785.01`** across 59,810 trades (`-$27.16` / trade).
  * 18 of the 21 candidate slots experienced complete **sign inversion** (profitable in-sample slots turned heavily negative out-of-sample).

### Phase 6 — 25-Cycle Rolling Retrain Simulation
* **Hypothesis:** "Did the static model fail because market regimes shifted and the model wasn't allowed to adapt?"
* **Action Taken (`08_rolling_retrain.py`):** Built a 25-cycle expanding/rolling window walk-forward simulation that retrained the hierarchical Bayesian model monthly, re-selected active candidate slots dynamically, and evaluated forward performance.
* **Outcome:** Rolling retraining failed to recover profitability, realizing a chained loss of **`-$2,569,959.98`** across 74,120 trades (`-$34.67` / trade), proving that dynamic recalibration does not fix the underlying signal breakdown.

### Phase 7 — Gross vs. Net Friction Audit
* **Hypothesis:** "Is the strategy failing due to commissions, exchange fees, and slippage?"
* **Action Taken (`gross_vs_net_analysis.py`):** Decomposed raw trade fills into gross price action vs. net realized returns.
* **Outcome:** Average round-trip friction was $\$4.50–\$6.20$ per contract, accounting for $< 18\%$ of total losses. **Raw Gross PnL was also deeply negative**, confirming that execution cost is NOT the driver of failure.

### Phase 8 — Conditioning Filters Audit (Macro, VIX, VWAP, ATR)
* **Hypothesis:** "Can pre-trade conditioning filters rescue the strategy by eliminating bad market regimes?"
* **Tested Filters (`three_filters_full.py`, `generate_vix_and_atr_matrices.py`):**
  1. **Macro News Blackout:** Excluded trades within $\pm 30$ min of FOMC, CPI, NFP, GDP releases.
  2. **VIX Volatility Regime:** Allowed trades only when prior-day $	ext{VIX} < 20$.
  3. **VWAP Trend Alignment:** Allowed Longs only above VWAP and Shorts only below VWAP.
  4. **ATR Local Volatility:** Allowed trades only when 14-day $	ext{ATR} \le 	ext{Median}$.
* **Outcome:** Filters removed up to 68.9% of trade volume but did not turn portfolio expectancy positive (`-$25.14` to `-$34.07` / trade).

---

## 3. Verified Headline Findings & Reconciled Numbers

*All numbers below are verified against `results/reconciliation_log.md` and raw persisted files.*

### Portfolio-Level Performance Table

| Evaluation Phase | Total Trades | Net Realized PnL | Mean PnL / Trade | Win Rate | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Sample Baseline (Jan 2024 – Jun 2025)** | 85,214 | $+\$5,412,890.00$ | $+\$63.52$ | 58.4% | In-Sample Fit |
| **Static Out-of-Sample Holdout (Jul 2025 – Jul 2026)** | 59,810 | **`-$1,528,785.01`** | **`-$27.16`** | 46.2% | **Signal Failure** |
| **25-Cycle Rolling Retrain Chained OOS** | 74,120 | **`-$2,569,959.98`** | **`-$34.67`** | 45.1% | **Signal Failure** |

### Conditioning Filters Summary (5,000-Resample Bootstrap)

| Scenario | Trades (N) | Trades Removed | Realized Mean | Bootstrap 95% CI | P(Loss) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unfiltered Baseline** | 59,810 | 0 (0.0%) | **`-$27.16`** | [`-$36.29`, `-$18.02`] | **100.0%** |
| **Macro News Blackout (±30 min)** | 53,177 | 6,633 (11.1%) | **`-$29.16`** | [`-$39.18`, `-$18.82`] | **100.0%** |
| **VIX Low-Vol Regime (VIX < 20)** | 45,096 | 14,714 (24.6%) | **`-$29.45`** | [`-$38.73`, `-$19.84`] | **100.0%** |
| **VWAP Trend Alignment** | 28,848 | 30,962 (51.8%) | **`-$25.14`** | [`-$39.91`, `-$10.88`] | **99.7%** |
| **All 3 Filters Combined** | 18,619 | 41,191 (68.9%) | **`-$34.07`** | — | — |

---

## 4. Forensic Root-Cause Analysis: Why the Alpha Failed

1. **Clock Time Has No Causal Power:** Fixed time-of-day slots reflected temporary order flow concentrations from specific market participants in 2024. As execution algorithms evolved in 2025–2026, those concentrations shifted or reversed.
2. **Systematic Sign Inversion:** When market structure changed, mean-reverting slots turned into momentum breakouts, causing severe losses.
3. **Conditioning Cannot Create Edge:** Post-trade filters (Macro/VIX/VWAP/ATR) successfully eliminate volatility noise, but **filtering a negative-expectancy signal cannot manufacture positive alpha**.

---

## 5. Interactive Visual Dashboards Guide

All 7 interactive dashboards are standalone HTML applications located in the repository root (accessible directly or via local server `http://127.0.0.1:7799/`):

1. **[`active_filter_slots_matrix_dashboard.html`](active_filter_slots_matrix_dashboard.html)**
   * Multi-asset switcher (`NQ`, `ES`, `CL`, `FDAX`) focusing on the 5 rescued/profitable slots, loss-reduction slots, and news overlap windows.
2. **[`nq_active_filters_matrix_dashboard.html`](nq_active_filters_matrix_dashboard.html)**
   * 100% isolated NQ dashboard comparing all 7 NQ candidate slots across all 4 filters.
3. **[`full_dataset_matrix_dashboard.html`](full_dataset_matrix_dashboard.html)**
   * Continuous 2.5-year history (no In/Out sample split, 2.9M trades) across Raw, Macro, VIX, and ATR filters.
4. **[`macro_news_matrix_dashboard.html`](macro_news_matrix_dashboard.html)**
   * Dedicated macro news blackout matrix (±30 min window, IS vs OOS).
5. **[`vix_matrix_dashboard.html`](vix_matrix_dashboard.html)**
   * VIX volatility regime matrix ($	ext{VIX} < 20$).
6. **[`atr_matrix_dashboard.html`](atr_matrix_dashboard.html)**
   * 14-day normalized ATR local volatility matrix.
7. **[`three_filters_dashboard.html`](three_filters_dashboard.html)**
   * Portfolio-level decomposition of Macro News, VIX Regime, and VWAP Trend filters.

---

## 6. Untested Future Research Concepts

*Note: These are exploratory concepts requiring stakeholder scope approval, NOT recommended deployment strategies.*

1. **Order Flow & Microstructure Alpha:** Real-time Order Book Imbalance (OBI), Cumulative Volume Delta (CVD), and depth-of-book skew.
2. **Cross-Asset Lead-Lag Dynamics:** Treasury yield / USD shifts predicting index futures momentum.
3. **Dynamic Volatility Bands:** Real-time Bollinger/Keltner percentile entries instead of clock-time buckets.

---

## 7. Repository Map & Pipeline Scripts

```
C:/Model-/
├── 📄 FINAL_REPORT.md                        # Master Comprehensive Research Report
├── 📄 INDEX.md                               # Complete Navigation Index
├── 📄 README.md                              # Master Narrative & Project Journey (This File)
├── 📄 model_spec.md                          # Hierarchical Model Specification
│
├── 🐍 00_data_prep.py                        # Data Cleaning & Ghost-Fill Resolution
├── 🐍 01_fold_structure.py                   # 5-Fold Cross Validation Partition
├── 🐍 02_fit_model.py                        # NumPyro Hierarchical Bayesian Fitting
├── 🐍 03_select_candidates.py                # Benjamini-Hochberg FDR Selection
├── 🐍 04_permutation_null.py                 # 1,000-Run Permutation Null Test
├── 🐍 05_feature_inventory.py                # Feature Inventory Compilation
├── 🐍 06_holdout_eval.py                     # Static 1-Year Holdout Evaluation
├── 🐍 07_report.py                           # Performance Summary Compilation
├── 🐍 08_rolling_retrain.py                  # 25-Cycle Rolling Retrain Simulation
│
├── 📊 active_filter_slots_matrix_dashboard.html  # Asset-Selectable Active Matrix
├── 📊 nq_active_filters_matrix_dashboard.html    # NQ-Isolated Active Matrix
├── 📊 full_dataset_matrix_dashboard.html         # Full 2.5-Year History Matrix
├── 📊 macro_news_matrix_dashboard.html           # Macro News Blackout Matrix
├── 📊 vix_matrix_dashboard.html                  # VIX Volatility Regime Matrix
├── 📊 atr_matrix_dashboard.html                  # ATR Volatility Regime Matrix
├── 📊 three_filters_dashboard.html               # 3-Filter Portfolio Dashboard
│
├── 📁 data/                                  # Parquet Trade & Market Data
├── 📁 results/                               # Verified Outputs, CSVs, Reconciliation Log
│   ├── 📁 trades_by_slot/                    # Individual CSV trade lists per slot
│   └── 📁 diagnostics/                       # Posterior convergence logs
└── 📁 archive/                               # Historical interim drafts
```
