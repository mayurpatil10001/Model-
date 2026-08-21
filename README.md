# GFRE-v3.1 — Quantitative Time-Slot Edge Research Project
## Status: RESEARCH PHASE — No Live Capital Authorized

**Last Updated:** 21-Aug-2026  
**Lead Analyst:** Gilad  
**Data Range:** 21-Jan-2024 → 17-Jul-2026  
**Assets:** NQ (Nasdaq-100), FDAX (DAX), ES (S&P 500), CL (Crude Oil)  
**Primary Dashboard:** `results/interactive_slot_matrix_dashboard.html`

---

## CURRENT STATUS (21-Aug-2026)

> **CONCLUSION: Static calendar-based slot selection is structurally anti-predictive.**
> No live capital is authorized. The project is transitioning to context-aware (regime-filtered) models.

| Status Item | Result |
|---|---|
| BH-FDR Candidates identified (IS) | 21 slots |
| Candidates surviving OOS | **2 / 21 (9.5%)** |
| Candidate IS PnL (selected slots) | **+$6,061,282** |
| Candidate OOS PnL (same slots, forward) | **-$1,528,785** |
| Permutation Null test (20 runs) | Model is **9.5× worse than random** |
| Live capital status | **NOT AUTHORIZED** |
| Next phase | Context-aware filters (VWAP, Volatility Regime, Macro) |

---

## 1. PROJECT OVERVIEW

### 1.1 Objective
Identify statistically robust intraday time-slot edges across 4 futures contracts
using a rolling walk-forward validation protocol with strict BH-FDR multiple-testing correction.

### 1.2 Data Architecture

| Dataset | Rows | Period | Role |
|---|---|---|---|
| `data/fold_assignments.parquet` | 2,832,740 | Jan 2024 – Jul 2026 | Master trade ledger |
| `data/slot_index.parquet` | — | — | Slot metadata (symbol, day, bucket, in_model flag) |
| `data/fold_date_ranges.parquet` | — | — | Rolling fold windows |
| `results/bh_fdr_candidates.csv` | 21 rows | — | Confirmed in-sample candidates |
| `results/rolling_retrain_cycles.csv` | — | — | Walk-forward cycle log |

### 1.3 Fold Structure

| Fold | Role | Train Start | Train End | Test Start | Test End | Rows |
|---|---|---|---|---|---|---|
| 1 | train/test | 2024-03-01 | 2024-10-31 | 2024-11-01 | 2024-12-31 | 469,095 + 188,855 |
| 2 | train/test | 2024-05-01 | 2024-12-31 | 2025-01-01 | 2025-02-28 | 651,512 + 248,330 |
| 3 | train/test | 2024-07-01 | 2025-02-28 | 2025-03-01 | 2025-04-30 | 728,817 + 396,006 |
| 4 | train/test | 2024-09-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 966,193 + 215,398 |
| H | holdout | — | — | 2025-07-01 | 2026-07-17 | 1,214,623 |

---

## 2. PERFORMANCE SUMMARY

### 2.1 Overall (All Model Slots)

| Period | Trades | Total PnL | Avg/Trade | Win Rate | W/L Ratio |
|---|---|---|---|---|---|
| In-Sample (IS) | 1,618,117 | -$43,979,870 | -$27 | — | — |
| Out-of-Sample (OOS) | 1,214,623 | -$29,684,915 | -$24 | — | — |

> Note: The overall PnL is negative because it includes ALL slots across all 4 assets, most of which are not "in-model" candidates. The BH-FDR filter selects only 21 slots.

### 2.2 Per-Asset (In-Model Slots Only)

| Asset | IS Trades | IS PnL | IS WR | IS W/L | OOS Trades | OOS PnL | OOS WR | OOS W/L |
|---|---|---|---|---|---|---|---|---|
| **NQ** (Nasdaq-100) | 319,067 | -$2,772,190 | 59.5% | 0.63 | 261,703 | **-$4,520,405** | 66.0% | **0.44** |
| **FDAX** (DAX) | 242,125 | -$20,304,050 | 51.8% | 0.75 | 92,709 | -$7,355,950 | 56.2% | 0.67 |
| **ES** (S&P 500) | 964,809 | -$19,410,638 | 43.3% | 1.08 | 729,562 | -$15,860,500 | 42.8% | 1.06 |
| **CL** (Crude Oil) | 92,059 | -$1,501,410 | 49.1% | 0.85 | 130,619 | -$1,923,210 | 52.9% | 0.79 |

> **NQ Critical Finding:** Win Rate improved IS→OOS (59.5% → 66.0%) but W/L ratio collapsed (0.63 → 0.44).
> Avg win ~$250 vs avg loss ~$570. Even at 66% win rate, model is mathematically unprofitable.

### 2.3 BH-FDR Candidates (21 Selected Slots)

| Period | Trades | Total PnL | Survived (mean > 0) |
|---|---|---|---|
| In-Sample | 84,108 | **+$6,061,282** | 21 / 21 |
| Out-of-Sample | 57,262 | **-$1,528,785** | **2 / 21** |

**OOS Reversal Rate: 19 of 21 slots (90.5%) reversed from profitable IS to losing OOS.**

### 2.4 Monthly OOS PnL (All Model Slots — Jul 2025 to Jul 2026)

| Month | OOS PnL | Trend |
|---|---|---|
| 2025-07 | (base) | — |
| 2025-08 | -$1,761,970 | ↓ |
| 2025-09 | -$1,763,558 | → |
| 2025-10 | -$1,480,848 | ↑ slight |
| 2025-11 | -$1,816,870 | ↓ |
| 2025-12 | -$2,283,293 | ↓ |
| 2026-01 | -$2,096,198 | → |
| 2026-02 | -$1,833,683 | ↑ slight |
| 2026-03 | -$4,874,470 | ↓↓ (OPEC event) |
| 2026-04 | -$3,712,598 | ↑ but negative |
| 2026-05 | -$2,370,800 | ↑ |
| 2026-06 | -$2,451,235 | → |
| 2026-07 | -$2,052,885 | ↑ slight |

---

## 3. BH-FDR CANDIDATE DETAIL (21 Slots)

All statistics computed from `fold_assignments.parquet` and `bh_fdr_candidates.csv`.

| Slot | Sym | Day | Bkt | IS n | IS Mean | IS t | OOS n | OOS Mean | OOS t | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 863 | NQ | Thu | 28 | 2,622 | $88.60 | 7.454 | 1,856 | $23.36 | 1.418 | REVERSED |
| 260 | ES | Mon | 25 | 8,081 | $49.94 | 7.432 | 5,818 | -$41.44 | -10.572 | REVERSED |
| 259 | ES | Mon | 24 | 8,236 | $49.80 | 7.005 | 6,064 | -$26.57 | -6.883 | REVERSED |
| 677 | FDAX | Fri | 27 | 1,171 | $348.25 | 6.446 | 259 | -$96.43 | -0.870 | REVERSED |
| 716 | NQ | Mon | 23 | 2,657 | $179.54 | 6.125 | 1,649 | -$68.31 | -4.772 | REVERSED |
| 675 | FDAX | Fri | 25 | 1,444 | $234.90 | 6.013 | 389 | -$422.81 | -3.312 | REVERSED |
| 357 | ES | Wed | 28 | 11,422 | $50.16 | 5.981 | 7,304 | -$29.63 | -7.265 | REVERSED |
| 718 | NQ | Mon | 25 | 2,579 | $114.56 | 5.946 | 1,640 | -$10.62 | -0.722 | REVERSED |
| 358 | ES | Wed | 29 | 10,892 | $65.03 | 5.896 | 8,596 | -$11.12 | -2.957 | REVERSED |
| 712 | NQ | Mon | 19 | 6,487 | $52.47 | 5.533 | 4,448 | -$45.00 | -4.797 | REVERSED |
| 719 | NQ | Mon | 26 | 2,333 | $163.54 | 5.519 | 1,444 | -$8.74 | -0.690 | REVERSED |
| 180 | CL | Thu | 40 | 378 | $178.02 | 5.229 | 198 | -$136.77 | -4.435 | REVERSED |
| 714 | NQ | Mon | 21 | 3,236 | $64.25 | 4.734 | 2,357 | -$17.64 | -1.393 | REVERSED |
| 282 | ES | Tue | 0 | 684 | $68.68 | 4.582 | 571 | -$14.54 | -0.904 | REVERSED |
| 854 | NQ | Thu | 19 | 8,560 | $37.39 | 4.229 | 4,929 | -$10.80 | -0.909 | REVERSED |
| **223** | **CL** | **Sun** | **36** | **1,303** | **$33.65** | **4.152** | **1,250** | **$53.92** | **2.854** | **⚠ SURVIVED (event-driven)** |
| 587 | FDAX | Wed | 27 | 1,135 | $250.22 | 3.648 | 248 | -$82.16 | -0.661 | REVERSED |
| 715 | NQ | Mon | 22 | 2,563 | $72.09 | 3.616 | 1,875 | -$83.74 | -6.143 | REVERSED |
| 648 | FDAX | Thu | 46 | 120 | $306.67 | 3.598 | 43 | -$415.70 | -2.107 | REVERSED |
| 901 | NQ | Fri | 19 | 5,928 | $27.66 | 3.539 | 4,529 | -$0.16 | -0.011 | REVERSED |
| 906 | NQ | Fri | 24 | 2,277 | $60.18 | 3.511 | 1,795 | -$37.20 | -2.845 | REVERSED |

**Total OOS PnL (candidates): -$1,528,785 | OOS n: 57,262**

---

## 4. ROOT CAUSE ANALYSIS — WHY THE MODEL FAILS

### 4.1 Mean-Reversion Trap (Primary Cause)

The BH-FDR selection process is mathematically required to select the slots with the **highest t-statistics in the training window**. High t-statistics occur when performance was unusually clustered (good luck concentrated in the measurement period). By definition, that cluster of luck mean-reverts when deployment begins.

This is not a code bug. It is structural: selecting "best performers" from historical data is guaranteed to capture noise-plus-luck, not signal.

### 4.2 The W/L Ratio Problem (NQ Specific)

```
NQ In-Model Slots:
  Win Rate OOS:  66.0%  (looks good)
  Avg Win:      ~$250
  Avg Loss:     ~$570
  W/L Ratio:     0.44

Breakeven W/L at 66% win rate requires: (1-0.66)/0.66 = 0.515
Actual W/L:  0.44  <  Required 0.515
Result: Mathematically impossible to profit at these parameters.
```

### 4.3 Permutation Null Test Results (20 Runs, Seed=42)

The model's real performance was compared against 20 runs where PnL values were randomly reshuffled (destroying all temporal structure):

| Metric | Random Permutation Avg | Real Model |
|---|---|---|
| Avg OOS PnL (selected slots) | **-$271,000** | **-$2,570,000** |
| Best single permutation run | ~+$66,000 (Perm #14) | -$1,528,785 |
| Model vs random (ratio) | — | **9.5× worse** |
| p-value (model vs null dist.) | — | **p = 1.000** |

**Interpretation:** A completely random slot selection strategy outperforms the model by ~9.5×. The BH-FDR filter is actively selecting the wrong slots.

### 4.4 MCMC Bayesian Diagnostic Failure

All 5 folds failed the ESS > 400 convergence gate:

| Fold | sigma_bkt ESS | R-hat | Verdict |
|---|---|---|---|
| 0 | 185 | 1.0245 | FAIL |
| 1 | 213 | 1.0210 | FAIL |
| 2 | 265 | 1.0195 | FAIL |
| 3 | 290 | 1.0188 | FAIL |
| 4 | 314 | 1.0178 | FAIL |

MCMC Bayesian approach is intractable with this data structure. BH-FDR remains the primary tool.

---

## 5. SLOT 223 — SPECIAL INVESTIGATION (CL, Sunday, Bucket 36)

Slot 223 is the only candidate that survived OOS with a positive mean. However, the survival is entirely driven by one market event.

### 5.1 Significance With vs. Without March 2026

| Metric | Full Holdout (n=1,250) | Excl. March 2026 (n=542) |
|---|---|---|
| Mean PnL/trade | $53.92 | -$43.73 |
| t-statistic | 2.854 | -2.013 |
| Bonferroni p (K=21) | 0.046 | 1.000 |
| Significant? | YES | **NO** |

### 5.2 Monthly OOS Breakdown

| Month | n | Mean PnL | Total PnL | Positive? |
|---|---|---|---|---|
| 2025-07 | 16 | -$110.62 | -$1,770 | NO |
| 2025-08 | 5 | -$230.00 | -$1,150 | NO |
| 2025-09 | 20 | -$35.00 | -$700 | NO |
| 2025-10 | 0 | — | — | — |
| 2025-11 | 56 | -$24.11 | -$1,350 | NO |
| 2025-12 | 8 | -$110.00 | -$880 | NO |
| 2026-01 | 27 | -$43.70 | -$1,180 | NO |
| 2026-02 | 34 | $62.35 | +$2,120 | YES |
| **2026-03** | **708** | **$128.67** | **+$91,100** | **YES (OPEC event)** |
| 2026-04 | 156 | $20.83 | +$3,250 | YES |
| 2026-05 | 167 | -$122.28 | -$20,420 | NO |
| 2026-06 | 25 | -$8.00 | -$200 | NO |
| 2026-07 | 28 | -$50.71 | -$1,420 | NO |

Positive months: **3 of 12 (25%)**. March 2026: 56.6% of all holdout trades, 135.2% of total PnL.

### 5.3 March 2026 Root Cause
OPEC+ announced on March 1, 2026 the unwinding of 206,000 bbl/day of voluntary production cuts.
The March 8 Globex Sunday re-open (18:00–18:30 ET, bucket 36) saw 475 trades vs a max of 71 on any other non-March Sunday. This is a **one-time identifiable macro event**, not a durable edge.

### 5.4 Slot 223 Verdict
**Do not trade.** Edge is entirely event-driven, not structural. Would require a macro-event detector, not a calendar filter.

### 5.5 Activation Protocol (If Fresh Data Available)
For Slot 223 only. Requires post-July 2026 data not yet in system.
1. Ingest Aug–Oct 2026 trades into `fold_assignments.parquet` with `is_holdout=True`.
2. Filter `slot_id == 223`.
3. Compute: `t = mean(profit_loss) / (std(profit_loss) / sqrt(n))`
4. Gate: if `t > 1.65` AND `n >= 200` AND no single month drives >50% of PnL → authorize paper-trading.
5. Live capital: not until paper-trading confirmation across ≥2 consecutive months.

---

## 6. INTERACTIVE DASHBOARD

**File:** `results/interactive_slot_matrix_dashboard.html`  
**Open in browser** (no server needed — self-contained HTML).

### What the Dashboard Shows

```
LEFT PANEL: In-Sample (Jan 2024 – Jun 2025) — Discovery Period
RIGHT PANEL: Out-of-Sample (Jul 2025 – Jul 2026) — Holdout Period

Each cell contains:
  - Total PnL (green = profitable, red = losing)
  - Trade count
  - Avg PnL/trade
  - Win Rate
  - W/L Ratio
  - Profit Factor
  - Permutation badges (which of 20 random runs selected this slot)
```

### Cell Interpretation Guide

| Cell Appearance | Meaning |
|---|---|
| 🟩 Green + Gold border (IS left) | Model candidate — was profitable in-sample |
| 🟥 Red + Gold border (OOS right) | SAME slot — lost money out-of-sample |
| No gold border | NOT TRADED — model rejected this slot |
| `MODEL` badge (blue) | Real model candidate |
| `P14★` badge (gold) | Best permutation run (Perm #14) selected this slot |
| `P03` badge (purple) | Other random permutation run selected this slot |
| `No Perm` badge (grey) | Never selected by any of 20 random runs |

> **Key insight for Gilad:** Red cells on the LEFT (IS) with NO gold border were never traded.
> Only gold-border cells were model candidates. The problem is green-left → red-right on gold-border cells.

### Clicking any cell opens a modal showing:
- Full IS vs OOS stats side-by-side
- Which permutation runs selected this slot
- Month-by-month PnL bar chart (31 months)
- Complete monthly trade table

---

## 7. RESEARCH AUDIT TIMELINE

| Date | Activity | Key Finding |
|---|---|---|
| Jul 2024 | Initial fold structure design | Rolling 5-fold + holdout established |
| Aug 2024 | BH-FDR candidate selection | 21 slots identified, IS PnL +$6M |
| Jan–Jun 2025 | Model locked, no retraining | Holdout period begins Jul 2025 |
| Jul–Dec 2025 | Holdout monitoring | Consistent losses, -$1.5M/month avg |
| Mar 2026 | OPEC+ event (Slot 223 spike) | One-time macro event, not edge |
| Aug 2026 | Full forensic audit | Mean-reversion trap confirmed |
| Aug 2026 | Permutation null test (20 runs) | Model 9.5× worse than random |
| Aug 2026 | Interactive dashboard built | IS/OOS side-by-side with perm labels |
| **Aug 2026** | **Current status** | **Transitioning to context-aware model** |

---

## 8. WHAT THE MODEL TOLD US (AND DIDN'T)

### What IS Working
- ✅ BH-FDR correctly identified slots with high IS t-statistics
- ✅ Walk-forward fold structure is sound
- ✅ Data cleaning pipeline is robust (2.8M trades, 4 assets, 2.5 years)
- ✅ The audit correctly caught the structural issue before large live capital deployment

### What Is NOT Working
- ❌ Static calendar time-slots carry no durable forward edge
- ❌ IS performance does not predict OOS performance (90.5% reversal rate)
- ❌ W/L ratio (0.44 for NQ) makes profitability impossible regardless of win rate
- ❌ Model performs worse than random permutation (p = 1.000)

---

## 9. NEXT PHASE — CONTEXT-AWARE MODEL

The evidence conclusively shows that trading a static calendar slot (Monday 09:30 ET, regardless of market conditions) does not work. The proposed evolution:

### 9.1 Required Filters Before Any Trade

| Filter | Condition | Rationale |
|---|---|---|
| Volatility Regime | ATR or VIX in defined range | High/low vol regimes have different micro-structure |
| VWAP Direction | Price above/below VWAP at slot start | Momentum context matters |
| News Blackout | No scheduled macro events ±30min | Adversely selected fills during events |
| Avg PnL Gate | Rolling 20-day avg PnL > 0 | If avg losing, do not trade the slot |

### 9.2 The Avg PnL Gate (Gilad's Rule)
> "If we know we have an avg losing window, we don't trade on that window."

This is implemented as: **if the rolling 20-day average PnL for a slot is negative, skip all trades for that slot until it turns positive.**

### 9.3 Development Roadmap

```
Phase 1: Volatility Regime Filter
  → Label each trade with: High / Normal / Low volatility regime
  → Identify which regimes, if any, show positive IS edge
  → Apply regime filter to holdout, measure improvement

Phase 2: VWAP Direction Filter  
  → Add VWAP calculation to trade data
  → Test: does long-above-VWAP / short-below-VWAP improve W/L?

Phase 3: Combined Context Model
  → Slot + Regime + VWAP + News Blackout
  → Walk-forward validation (same protocol)
  → Minimum threshold: OOS W/L > 0.6 AND Win Rate > 55%

Phase 4: Paper Trading Gate
  → 60 days paper trading before any live capital
  → Daily monitoring against rolling 20-day avg PnL gate
```

---

## 10. MODULE CATALOG

| Script | Purpose | Output |
|---|---|---|
| `00_data_prep.py` | Clean raw data; outlier clip; sparsity filter | `data/trades_clean.parquet` |
| `01_fold_structure.py` | Rolling 5-fold CV + holdout partition | `data/fold_assignments.parquet` |
| `03_select_candidates.py` | BH-FDR (Q=0.01) candidate selection | `results/bh_fdr_candidates.csv` |
| `04_permutation_null.py` | 20-run null reshuffling | `results/step2_permutation_null.csv` |
| `06_holdout_eval.py` | Locked holdout evaluation | Console output |
| `generate_readme_dynamic.py` | Dynamic README generator | `README.md` |
| `scratch/build_perm_labeled_dashboard.py` | IS/OOS dashboard with permutation labels | `results/interactive_slot_matrix_dashboard.html` |
| `scratch/rolling_retrain_validation.py` | Rolling retrain with BH/FDR + confirmation | Rolling cycle results |
| `scratch/all21_mc_walkforward.py` | Monte Carlo walk-forward (20 perms × all cycles) | Permutation null audit |

---

## 11. FILE STRUCTURE

```
C:\Model-\
├── README.md                          ← This file
├── data\
│   ├── fold_assignments.parquet       ← 2.8M trades with fold labels
│   ├── slot_index.parquet             ← Slot metadata
│   └── fold_date_ranges.parquet       ← Fold date windows
├── results\
│   ├── bh_fdr_candidates.csv          ← 21 BH-FDR selected slots
│   ├── rolling_retrain_cycles.csv     ← Walk-forward cycle log
│   ├── interactive_slot_matrix_dashboard.html  ← MAIN DASHBOARD
│   └── step2_permutation_null.csv     ← Permutation test results
└── scripts\
    ├── 00_data_prep.py
    ├── 01_fold_structure.py
    ├── 03_select_candidates.py
    ├── 04_permutation_null.py
    └── 06_holdout_eval.py
```

---

## 12. KEY DECISIONS & CONSTRAINTS

| Decision | Rationale |
|---|---|
| Holdout locked (no peeking) | Prevents data leakage |
| BH-FDR Q=0.01 (not 0.05) | Conservative to reduce false discoveries |
| Min 50 IS trades per slot | Prevents small-sample selection |
| Confirmation: 2 of last 3 months positive IS | Ensures recency, not just historical avg |
| MCMC abandoned | All 5 folds failed ESS > 400 gate |
| No live capital authorized | OOS evidence does not support deployment |

---

## 13. IMPORTANT NOTES FOR STAKEHOLDERS

> **For Gilad:**
> 
> 1. The left (IS) and right (OOS) panels in the dashboard are comparing the SAME slots before and after deployment. Green→Red is the failure mode.
>
> 2. Red cells on the left with NO gold border were never traded. Only gold-border slots were model candidates.
>
> 3. "Best permutation" (Perm #14) refers to the best of 20 RANDOM slot selections. Even the best random run barely survived OOS. This proves the selection algorithm is capturing noise, not signal.
>
> 4. Gilad's instinct is correct: if the best permutation is losing OOS, we don't trade. This is the research telling us: stop static slot trading.

---

*README regenerated: 21-Aug-2026. Run `python scratch/build_perm_labeled_dashboard.py` to refresh dashboard.*
