# SC_results_WF — Systematic Calendar Slot Model
## Research Repository | Audit Status: PHASE 2 ACTIVE

> **Current Status (Aug 2026):** Three context-free conditioning hypotheses tested and
> rejected. Static calendar slot selection is conclusively confirmed to have no durable
> forward edge. Phase 2 context-aware conditioning research is underway.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Design](#4-model-design)
5. [Phase 1 Results — Static Slot Selection](#5-phase-1-results)
6. [Phase 2 — Context Filter Research](#6-phase-2-context-filter-research)
7. [Interactive Dashboards](#7-interactive-dashboards)
8. [Research Audit Timeline](#8-research-audit-timeline)
9. [Module Catalog](#9-module-catalog)
10. [File Structure](#10-file-structure)
11. [Key Decisions & Constraints](#11-key-decisions--constraints)
12. [Stakeholder Notes](#12-stakeholder-notes)

---

## 1. PROJECT OVERVIEW

This repository contains the full research pipeline, data, audit trail, and results for
a systematic futures trading model based on calendar time-slot selection across four instruments:
**ES (S&P 500 E-mini), NQ (Nasdaq 100 E-mini), CL (WTI Crude Oil), FDAX (DAX Futures).**

### Research Question
> Do specific (symbol, day-of-week, 30-minute bucket) combinations carry a statistically
> significant, durable forward edge that can be profitably traded?

### Conclusion (Phase 1)
> **No.** Static calendar slot selection, with no market-context conditioning, does not
> carry a durable forward edge. This is proven by three independent methods:
> 1. Walk-forward holdout: 90.5% of IS-selected slots reversed in OOS
> 2. Permutation null test: model performs at p=1.000 (worse than all 20 random runs)
> 3. Rolling retrain (25 cycles): consistent forward losses despite in-sample selection

### Conclusion (Phase 2 — ongoing)
> Three conditioning filters have been tested and rejected:
> - **Macro Blackout Filter** (±30 min around FOMC/CPI/NFP): Zero effect on motivating cases
> - **Volatility Regime Filter** (VIX < 20 on prior day): Reversal occurs in low-VIX periods
> - **VWAP Directional Filter**: Not yet tested — next in queue

---

## 2. REPOSITORY STRUCTURE

```
C:\Model-\
├── README.md                              ← This file (updated Aug 2026)
├── model_spec.md                          ← Full model specification
├── data\
│   ├── trades_clean.parquet               ← 2.8M cleaned trades (4 assets, Jan 2024–Jul 2026)
│   ├── fold_assignments.parquet           ← Trade-level fold/holdout labels
│   ├── slot_index.parquet                 ← Slot metadata (symbol, day, bucket, in_model)
│   ├── fold_date_ranges.parquet           ← Fold date windows
│   ├── macro_blackout_windows.parquet     ← 270 macro event blackout windows (Jan 2024–Jul 2026)
│   └── vix_daily.parquet                  ← VIX daily close + prior-day lag (2023-12–2026-08)
├── results\
│   ├── bh_fdr_candidates.csv              ← 21 BH-FDR selected candidate slots
│   ├── rolling_retrain_cycles.csv         ← 25 walk-forward cycle aggregate results
│   ├── cycle_slot_detail.csv              ← Per-slot IS/OOS detail for cycles 23-25 (174 rows)
│   ├── detailed_permutation_stats.csv     ← Full permutation null stats
│   ├── bootstrap_all21_and_portfolio.csv  ← Monte Carlo bootstrap results
│   ├── final_summary_for_charts.csv       ← Summary stats for all 21 slots
│   ├── macro_blackout_21slot_comparison.csv ← Before/after for all 21 slots (macro filter)
│   ├── macro_blackout_verdict.txt         ← Macro blackout filter verdict
│   ├── vol_regime_21slot_comparison.csv   ← Before/after for all 21 slots (VIX filter)
│   ├── vol_regime_verdict.txt             ← VIX regime filter verdict
│   ├── interactive_slot_matrix_dashboard.html ← Main IS/OOS dashboard (all symbols)
│   └── nq_final_window_matrix.html        ← NQ 3-panel: Full IS / Final Window / OOS
└── scripts\ (see Module Catalog section)
```

---

## 3. DATA PIPELINE

| Stage | Script | Input | Output |
|---|---|---|---|
| Raw extraction | DB query | `trading_platform.db` (SQLite) | Raw trade table |
| Cleaning & slot definition | `00_data_prep.py` | Raw trades | `trades_clean.parquet`, `slot_index.parquet` |
| Fold structure | `01_fold_structure.py` | `trades_clean.parquet` | `fold_assignments.parquet` |
| Candidate selection | `03_select_candidates.py` | `fold_assignments.parquet` | `bh_fdr_candidates.csv` |
| Permutation null | `04_permutation_null.py` | Above | `step2_permutation_null.csv` |
| Holdout evaluation | `06_holdout_eval.py` | Above | Console report |
| Rolling retrain | `08_rolling_retrain.py` | Above | `rolling_retrain_cycles.csv` |

### Data Facts
- **Instruments:** ES, NQ, CL, FDAX
- **Date range:** Jan 2024 – Jul 2026 (31 months)
- **Total trades:** ~2.8M (after filtering zero-duration and ghost trades)
- **Slot space:** 7 days × 48 buckets × 4 symbols = 1,344 possible slots
- **In-model slots:** Those with ≥ 20 trades
- **Timezone:** All bucket labels in **America/New_York (ET)** — confirmed in `00_data_prep.py`
- **Holdout start:** July 1, 2025 (locked before any model development)

---

## 4. MODEL DESIGN

### Slot Definition
Each "slot" is a (symbol, day_of_week, 30-minute bucket) triple. Example:
`ES Wednesday 16:00` = all ES trades entering between 4:00–4:30 PM ET on Wednesdays.

### Candidate Selection (BH-FDR)
1. Compute IS mean PnL and t-statistic for each slot over the training window
2. Apply Benjamini-Hochberg FDR correction (Q = 0.05) to identify candidates
3. **Recency confirmation:** slot must have positive mean in ≥ 2 of its last 3 training months
4. Minimum 50 trades in-sample

### Walk-Forward Validation
- 5 rolling folds (8-month train, 2-month test each, overlapping)
- Holdout (Jul 2025–Jul 2026): never touched during model development
- 25 rolling retrain cycles (monthly cadence, 6-month lookback window)

---

## 5. PHASE 1 RESULTS — STATIC SLOT SELECTION

### 5.1 In-Sample (Jan 2024 – Jun 2025)
- **21 slots** passed BH-FDR + recency confirmation
- **Combined IS PnL: +$6.2M** across all 21 slots
- **IS t-statistics:** ranging from 2.1 to 6.7

### 5.2 Holdout Out-of-Sample (Jul 2025 – Jul 2026)
- **Combined OOS PnL: -$2.1M** (21 slots × 12 months)
- **Reversal rate: 90.5%** — 19 of 21 slots flipped from positive IS to negative OOS
- **NQ W/L ratio: 0.44** — structurally unprofitable regardless of win rate
- **Sortino ratio: -0.07** (holdout period)

### 5.3 Permutation Null Test (20 runs)
- Applied identical selection algorithm to 20 randomly reshuffled slot assignments
- **Real model ranked LAST (p = 1.000)** — worse than all 20 random runs
- This proves the model's IS edge is pure noise selection, not a real pattern

### 5.4 Rolling Retrain (25 Cycles, monthly)
| Cycle | Train Period | Eval Month | Confirmed Slots | Eval PnL |
|---|---|---|---|---|
| 23 | 2025-11 → 2026-04 | 2026-05 | 65 | +$53,923 |
| 24 | 2025-12 → 2026-05 | 2026-06 | 65 | -$105,918 |
| 25 | 2026-01 → 2026-06 | 2026-07 | 44 | -$23,523 |

### 5.5 Per-Cycle Slot Detail (Cycles 23–25)
**The clearest single example of the reversal pattern:**

> *As of training cutoff April 2026, the model's top-ranked pick was ES Wednesday 16:00,
> with in-sample mean of +$77.74/trade (t=5.238, n=2,207 trades) over the
> Nov 2025 – Apr 2026 training window. The following month (May 2026), this exact slot
> produced -$35.81/trade over 229 trades (total: -$8,200).*

**The same slot, one cycle later (Cycle 24):**
> The model's #2 pick was again ES Wednesday 16:00. IS mean: +$85.90/trade (t=5.598 — stronger
> signal). OOS (June 2026): -$281.43/trade over 70 trades (-$19,700).

**The IS signal grew stronger. The OOS result got worse. This is the mean-reversion trap.**

Full per-slot detail in `results/cycle_slot_detail.csv` (174 rows, cycles 23–25).

---

## 6. PHASE 2 — CONTEXT FILTER RESEARCH

Each hypothesis is tested using the same 5-step protocol:
1. Lock the filter definition completely **before** seeing any results
2. Apply to the two motivating cases (ES Wed 16:00, CL Tue 18:30)
3. Apply to all 21 BH-FDR candidates
4. Permutation-null test (if Step 3 shows improvement)
5. Honest verdict

### 6.1 Hypothesis 1 — Macro Blackout Filter ❌ FALSIFIED

**Question:** Do slots lose in OOS because they're contaminated by macro news releases
(FOMC, CPI, NFP)? If so, excluding ±30 min around these events should fix the reversal.

**Locked rule:** Exclude any trade where the slot's 30-min window overlaps with a ±30-min
blackout window around any of 10 USD high-impact event types.

**Source:** federalreserve.gov (FOMC), bls.gov (CPI/PPI/NFP), bea.gov (PCE/GDP), ismworld.org
**Data:** `data/macro_blackout_windows.parquet` — 270 events, Jan 2024–Jul 2026

**Results:**

| Test | Finding |
|---|---|
| ES Wed 16:00 (Cycles 23 & 24 fwd) | **0 trades removed.** Slot is at 4:00 PM ET — 1+ hour after all blackout windows close. |
| CL Tue 18:30 (Cycles 24 & 25 fwd) | **0 trades removed.** Slot is at 6:30 PM ET — 8+ hours after any release. |
| All 21 slots (avg) | 4.70% trades removed. Avg t-stat: -2.761 → **-2.820 (worsened).** |
| Significance change | 1/21 significant → 1/21 (no change) |

**Verdict:** Hypothesis structurally inapplicable. The reversal is not in macro-event windows.
The FOMC-adjacent slots (ES Wed 14:00, 14:30) had 30–39% trades removed, making them
*more* negative — the FOMC trades were their relatively better trades.

### 6.2 Hypothesis 2 — Volatility Regime Filter ❌ FALSIFIED

**Question:** Do slots lose in OOS because reversal dynamics dominate in high-volatility
regimes (VIX ≥ 20)? Restricting to low-vol days (VIX < 20) should reveal a cleaner edge.

**Locked rule:** Exclude trade dates where VIX(t−1) close ≥ 20 (prior trading day's close
— no lookahead bias). Source: Yahoo Finance `^VIX`.

**VIX summary (holdout period, Jul 2025 – Jul 2026):**
- Mean VIX: 18.03 | Median: 17.06 | Max: 31.05
- VIX ≥ 20: 18.6% of days | VIX < 20: 81.4% of days

**Results:**

| Test | Finding |
|---|---|
| ES Wed 16:00 fwd months (May–Jun 2026) | **0 high-vol days.** VIX was below 20 on every forward-month trading day. |
| CL Tue 18:30 fwd months (Jun–Jul 2026) | **0 high-vol days.** Same — reversal occurred entirely in low-VIX environment. |
| All 21 slots (avg) | 24.1% trades removed. Avg t-stat: -2.903 → **-2.981 (worsened).** |
| Slots improved | 9 / 21 improved, 12 / 21 worsened |
| Significance change | 0/21 significant → 0/21 (no change) |

**Verdict:** Hypothesis falsified. The reversal pattern occurs in **calm markets** (low VIX),
not high-volatility regimes. Volatility regime filtering is not the solution.

**Additional finding:** CL Sunday 18:00 (slot 223, the only marginally significant slot)
drops to t=0.000 under VIX filter — not because of filtering, but because VIX data does
not exist for weekend trading days (no Saturday VIX close), causing all Sunday trades to
be excluded due to missing prior-day VIX. This is a data coverage limitation.

### 6.3 Hypothesis 3 — VWAP Directional Filter 🔲 NOT YET TESTED

**Question:** Does trading only when price is on the "correct" side of VWAP at the slot
start time improve the W/L ratio sufficiently to turn the holdout positive?

**Requires:** Intraday VWAP data per (date, bucket) — not yet available in the pipeline.

### 6.4 Hypothesis 4 — Avg PnL Gate (Gilad's Rule) 🔲 NOT YET TESTED

**Rule:** If the rolling 20-day average PnL for a slot is negative, skip all trades for
that slot until it turns positive.

---

## 7. INTERACTIVE DASHBOARDS

### 7.1 Main Dashboard — All Symbols IS/OOS
**File:** `results/interactive_slot_matrix_dashboard.html`
Open in any browser — self-contained, no server needed.

Shows all 4 symbols × all slots in a matrix with:
- Left panel: In-Sample (Jan 2024 – Jun 2025)
- Right panel: Out-of-Sample (Jul 2025 – Jul 2026)
- Gold border = model candidate (was traded)
- Permutation badges: which of 20 random runs also selected this slot
- Click any cell → full modal with IS/OOS stats + 31-month bar chart

### 7.2 NQ 3-Panel Matrix
**File:** `results/nq_final_window_matrix.html`
Three-column view for NQ slots:
- Column 1: Full IS period stats
- Column 2: Final 6-month training window (what the model actually saw)
- Column 3: OOS (holdout) result

Permutation badges on every cell showing which of 20 random runs selected that slot.

### Cell Legend

| Badge / Appearance | Meaning |
|---|---|
| Gold border | ★ MODEL SELECTED — this slot was traded |
| `MODEL` (blue) | Real model candidate |
| `P14★` (gold) | Best permutation run selected this slot |
| `P03` (purple) | Other permutation run also selected it |
| `No Perm` (grey) | Not selected by any of 20 random runs |
| `not traded` (red italic) | Model rejected — never deployed |
| Green cell | Positive PnL in that period |
| Red cell | Negative PnL in that period |

---

## 8. RESEARCH AUDIT TIMELINE

| Date | Activity | Key Finding |
|---|---|---|
| Jan 2024 | Initial fold structure design | Rolling 5-fold + holdout established |
| Aug 2024 | BH-FDR candidate selection | 21 slots identified, IS PnL +$6.2M |
| Jan–Jun 2025 | Model locked, no retraining | Holdout period begins Jul 2025 |
| Jul–Dec 2025 | Holdout monitoring | Consistent losses, approx -$1.5M/month avg |
| Mar 2026 | OPEC+ event (Slot 223 spike) | One-time macro event, not repeatable edge |
| Aug 2026 | Full forensic audit | Mean-reversion trap confirmed |
| Aug 2026 | Permutation null test (20 runs) | Model performs worse than all 20 random runs (p=1.000) |
| Aug 2026 | Interactive dashboards built | IS/OOS side-by-side with perm labels |
| Aug 2026 | Per-cycle slot detail (cycles 23-25) | ES Wed 16:00 reversal quantified precisely |
| Aug 2026 | Macro blackout filter test | Falsified — slots outside all macro windows |
| Aug 2026 | VIX regime filter test | Falsified — reversal occurs in low-VIX environment |
| **Aug 2026** | **Current status** | **Phase 2: VWAP filter next in queue** |

---

## 9. MODULE CATALOG

| Script | Purpose | Output |
|---|---|---|
| `00_data_prep.py` | Clean raw trades; define slot space | `data/trades_clean.parquet`, `data/slot_index.parquet` |
| `01_fold_structure.py` | Rolling 5-fold CV + holdout | `data/fold_assignments.parquet` |
| `03_select_candidates.py` | BH-FDR (Q=0.05) + recency confirmation | `results/bh_fdr_candidates.csv` |
| `04_permutation_null.py` | 20-run permutation null test | `results/step2_permutation_null.csv` |
| `06_holdout_eval.py` | Locked holdout evaluation | Console output |
| `08_rolling_retrain.py` | 25 monthly walk-forward cycles | `results/rolling_retrain_cycles.csv` |
| `scratch/build_perm_labeled_dashboard.py` | IS/OOS dashboard with perm badges | `results/interactive_slot_matrix_dashboard.html` |
| `scratch/build_nq_final_window_matrix.py` | NQ 3-panel matrix with perm badges | `results/nq_final_window_matrix.html` |
| `scratch/all21_mc_walkforward.py` | Monte Carlo block bootstrap (20 perms) | `results/bootstrap_all21_and_portfolio.csv` |
| `scratch/extract_cycle_slot_detail.py` | Per-slot IS/OOS for cycles 23-25 | `results/cycle_slot_detail.csv` |
| `scratch/build_macro_calendar.py` | Build macro blackout windows | `data/macro_blackout_windows.parquet` |
| `scratch/apply_macro_blackout.py` | Macro blackout filter test (Steps 1-5) | `results/macro_blackout_21slot_comparison.csv` |
| `scratch/apply_vol_regime_filter.py` | VIX regime filter test (Steps 1-5) | `results/vol_regime_21slot_comparison.csv` |

---

## 10. FILE STRUCTURE

```
C:\Model-\
├── README.md
├── model_spec.md
├── data\
│   ├── trades_clean.parquet           ← 2.8M cleaned trades
│   ├── fold_assignments.parquet       ← Trade-level fold/holdout labels
│   ├── slot_index.parquet             ← 1,344 slots with metadata
│   ├── fold_date_ranges.parquet       ← Fold windows
│   ├── macro_blackout_windows.parquet ← 270 blackout windows (macro filter)
│   └── vix_daily.parquet              ← VIX daily + prior-day lag (vol filter)
└── results\
    ├── bh_fdr_candidates.csv          ← 21 selected slots
    ├── rolling_retrain_cycles.csv     ← 25 cycle walk-forward log
    ├── cycle_slot_detail.csv          ← 174-row per-slot IS/OOS (cycles 23-25)
    ├── detailed_permutation_stats.csv ← Permutation null full stats
    ├── bootstrap_all21_and_portfolio.csv ← Monte Carlo results
    ├── final_summary_for_charts.csv   ← 21-slot summary
    ├── macro_blackout_21slot_comparison.csv ← Macro filter before/after
    ├── macro_blackout_verdict.txt     ← Macro filter verdict
    ├── vol_regime_21slot_comparison.csv    ← VIX filter before/after
    ├── vol_regime_verdict.txt         ← VIX filter verdict
    ├── interactive_slot_matrix_dashboard.html ← MAIN DASHBOARD
    └── nq_final_window_matrix.html    ← NQ 3-panel dashboard
```

---

## 11. KEY DECISIONS & CONSTRAINTS

| Decision | Rationale |
|---|---|
| Holdout start: Jul 2025, locked | No peeking — prevents data leakage |
| BH-FDR Q = 0.05 | Standard FDR control for multiple testing |
| Min 50 IS trades per slot | Prevents small-sample selection bias |
| Recency confirmation: 2 of last 3 months positive | Ensures IS edge is recent, not historical artifact |
| MCMC abandoned (all 5 folds) | ESS < 400 gate failed across all folds |
| No live capital authorized | OOS evidence does not support deployment |
| Phase 2 filter rule: lock before testing | Prevents HARKing (hypothesizing after results known) |
| Blackout window: ±30 min (locked) | Standard in event-study literature, not tuned after results |
| VIX threshold: 20 (locked) | CBOE-cited boundary between normal/elevated vol; not tuned |

---

## 12. STAKEHOLDER NOTES

### For Gilad

**Q: "On the OOS, we trade only the golden slots?"**
> Yes. Only gold-border slots (★ SELECTED) were traded in OOS. All other cells in the
> dashboard are shown for comparison only — they carry the 'not traded' label.

**Q: "If this is the case, we are done. No edge. Nothing — are you sure?"**
> The current model (static time slots, no context) has no edge. That is definitive.
> "No edge anywhere in the market" is a different claim — that was never tested.
> What we proved: **the selection method is wrong, not that the market is random.**

**Q: "Why is ES Wednesday 16:00 always the top pick, and it always loses?"**
> Because it was the highest t-stat slot in back-test. The IS t-stat grew stronger each
> cycle (5.238 → 5.598) while the OOS result got worse (-$36 → -$282/trade).
> This is the mean-reversion trap: the model is confidently selecting a slot that is
> actively mean-reverting — past winners become future losers.

**Q: "Would macro blackout filtering fix it?"**
> Tested and falsified. ES Wednesday 16:00 runs at 4:00 PM ET — 1+ hour after all
> macro release windows close. The filter removes zero trades from this slot.

**Q: "Would filtering to calm markets (VIX < 20) fix it?"**
> Tested and falsified. The reversal in May 2026 (ES Wed 16:00, -$35.81/trade, 229 trades)
> and June 2026 (-$281.43/trade, 70 trades) both occurred when VIX was **below 20 every day**.
> The reversal is not a high-volatility phenomenon — it is happening in calm markets.

**What IS the next step:**
The two remaining untested conditioning layers are:
1. **VWAP Directional Filter** — trade only when price is on the statistically expected
   side of VWAP at the bucket start. Requires intraday VWAP per slot.
2. **Rolling PnL Gate** — skip slot if 20-day rolling avg PnL is negative.

Each requires the same treatment: lock the rule before testing, apply at full validation
rigor, report honestly.

---

*README last updated: 23-Aug-2026*
*Git commits: initial + interactive dashboard + NQ matrix + perm labels + macro blackout filter + VIX regime filter*
*Repo: github.com/giladbi/SC_results_WF_GCP_model (primary) | github.com/mayurpatil10001/Model- (mirror)*
