# SC_results_WF — Final Research Closeout Report

**Date:** August 23, 2026  
**Status:** COMPLETE — All planned hypotheses tested and reported  
**Research question:** Which (symbol, day-of-week, 30-min bucket) permutations generate Sortino > 3.0, validated out-of-sample?

---

## EXECUTIVE SUMMARY

**Final answer to the research question:** None.

No method tested across the full research program produced a validated, stress-tested, positive out-of-sample edge. Every conditioning filter proposed was tested with pre-specified, locked rules and reported honestly — including cases where the filter made things worse. This is the primary conclusion, not a footnote.

---

## COMPLETE SUMMARY OF EVERYTHING TESTED

| # | Method | Verdict |
|---|---|---|
| 1 | **Static Calendar Slot Selection** (21 BH-FDR slots) | **REJECTED** — 90.5% reversal rate OOS; p=1.000 vs permutation null |
| 2 | **Hierarchical Bayesian / MCMC** | **ABANDONED** — All 5 folds failed ESS ≥ 400 convergence gate |
| 3 | **25-Cycle Rolling Retrain** (monthly BH-FDR re-selection) | **REJECTED** — Consistent forward losses across all 25 cycles |
| 4 | **Macro Blackout Filter** (±30 min, 270 events) | **REJECTED** — Zero trades removed from motivating slots; t-stat worsened |
| 5 | **Volatility Regime Filter** (VIX < 20, prior-day close) | **REJECTED** — Reversal occurs in low-VIX periods; t-stat worsened |
| 6 | **VWAP Directional Filter** (6h rolling, Yahoo Finance 1h OHLCV) | **REJECTED** — Mean PnL worsened; t-stat shift is subsampling artifact |

---

## DETAILED RESULTS

### 1. Static Calendar Slot Selection

- **21 slots** selected via BH-FDR + recency confirmation  
- IS PnL: +$6.2M (Jan 2024 – Jun 2025)  
- OOS PnL: −$2.1M (Jul 2025 – Jul 2026)  
- **Reversal rate: 90.5%** (19 of 21 slots flipped positive→negative)  
- **Permutation null: p = 1.000** — real model ranked last of 20 random runs  
- NQ W/L ratio: 0.44 — structurally unprofitable

**Clearest single example — ES Wednesday 16:00:**

| Cycle | IS Mean | IS t-stat | Forward Month | OOS Mean | OOS Total |
|---|---|---|---|---|---|
| 23 | +$77.74/trade | 5.238 | May 2026 (n=229) | **−$35.81/trade** | −$8,200 |
| 24 | +$85.90/trade | 5.598 | Jun 2026 (n=70) | **−$281.43/trade** | −$19,700 |

The IS signal grew stronger while OOS got worse — the mean-reversion trap in its clearest form.

---

### 2. Hierarchical Bayesian / MCMC

Abandoned at the diagnostic stage. All 5 folds failed the ESS ≥ 400 convergence gate. No posterior samples were usable. No results are reported, as none met minimum reliability standards.

---

### 3. Rolling Retrain (25 Cycles)

Monthly re-application of BH-FDR + confirmation on a 6-month sliding window.

| Cycle | Forward Month | Confirmed Slots | Forward PnL |
|---|---|---|---|
| 23 | May 2026 | 65 | +$53,923 |
| 24 | Jun 2026 | 65 | −$105,918 |
| 25 | Jul 2026 | 44 | −$23,523 |

The single positive month (Cycle 23) did not survive. No durable forward predictability.

---

### 4. Macro Blackout Filter

**Locked rule:** Exclude ±30 min around 10 USD high-impact event types (FOMC, CPI, NFP, PPI, PCE, GDP, Retail Sales, ISM). 270 events, Jan 2024–Jul 2026, from official US sources.

| Test | Result |
|---|---|
| ES Wed 16:00 (fwd months) | **0 trades removed** — slot is 60+ min after all blackout windows |
| CL Tue 18:30 (fwd months) | **0 trades removed** — slot is 8+ hours after any release |
| All 21 slots avg t-stat | −2.761 → **−2.820** (worsened) |

The motivating slots operate in after-hours windows structurally outside all macro release windows.

---

### 5. Volatility Regime Filter

**Locked rule:** Exclude trade dates where VIX(t−1) ≥ 20. Source: Yahoo Finance ^VIX, prior-day close (no lookahead bias). Threshold not adjusted after seeing results.

Holdout VIX (Jul 2025 – Jul 2026): Mean 18.03, Median 17.06. VIX ≥ 20 on only 18.6% of days.

| Test | Result |
|---|---|
| ES Wed 16:00 (May + Jun 2026) | **0 trades removed** — VIX < 20 every day in both forward months |
| CL Tue 18:30 (Jun + Jul 2026) | **0 trades removed** — same |
| All 21 slots avg t-stat | −2.903 → **−2.981** (worsened) |

The reversal occurred entirely in calm markets (low VIX). Filtering high-vol days is inapplicable.

---

### 6. VWAP Directional Filter

**Locked rule:** Include LONG trades only if entry_price > 6h-rolling-VWAP; include SHORT trades only if entry_price < 6h-rolling-VWAP. 6-bar rolling window using 1-hour OHLCV (Yahoo Finance, ES=F / NQ=F / CL=F). FDAX excluded — no Yahoo Finance 1h data (4 of 21 slots).

**Data:** 39,498 1-hour bars; entry prices from `processed_trades` DB (zero nulls). Slot assignments rebuilt directly from DB timestamps. Filter coverage: 94.4% of holdout trades matched VWAP data.

**Motivating cases:**

| Slot | Period | ALL mean | VWAP-confirmed mean |
|---|---|---|---|
| ES Wed 16:00 | May 2026 (n=231→44) | −$36.31/trade | **−$52.56/trade (WORSE)** |
| ES Wed 16:00 | Jun 2026 (n=72→45) | −$276.56/trade | **−$413.61/trade (WORSE)** |
| CL Tue 18:30 | Jul 2026 (n=3→1) | −$263.33/trade | **−$330.00/trade (WORSE)** |

**All 17 non-FDAX slots:**

| Metric | All trades | VWAP-confirmed |
|---|---|---|
| Avg t-stat | −3.252 | −2.223 (+1.030) |
| **Avg mean PnL/trade** | **−$30.89** | **−$38.32 (WORSE)** |
| Significant (Bonferroni) | 0 / 17 | 0 / 17 |
| Slots improved (mean PnL) | — | 7 / 17 |
| Avg trades removed | — | 47.2% |

> **On the +1.030 avg t-stat change:** This is a **statistical artifact** of removing 47% of trades.  
> Halving n increases standard error (SE ∝ 1/√n), mechanically pushing t toward zero.  
> The correct performance metric is mean PnL/trade, which got **worse** (−$30.89 → −$38.32).  
> VWAP-confirmed trades are the *more* losing trades — the filter selected for losses.

Step 4 (full validation stack) not triggered: threshold requires avg t > +0.5 AND ≥10/17 improved AND >0 newly significant. Only the first criterion was nominally met, and as noted, that reflects the subsampling artifact.

---

## FINAL ANSWER

> **"Which permutation, per day/time, generates Sortino > 3, validated out-of-sample?"**

**No such permutation was found.** The full research program — 1,344 candidate slots, three selection frameworks, 25 rolling retrain cycles, three conditioning filters, permutation-null testing, and Monte Carlo bootstrap — produced no validated positive edge meeting any reasonable significance threshold.

---

## WHAT WAS LEARNED

1. **Calendar slot identity (symbol + day + time) alone does not carry a durable forward edge.** This is the primary, clearly proven conclusion. The market context at the time of the slot matters; the slot identity without context does not.

2. **High IS t-statistics are not reliable predictors of OOS performance at this resolution.** ES Wednesday 16:00 had IS t = 5.598 and OOS mean = −$281.43/trade the very next month. High IS confidence predicted OOS failure.

3. **Three specific causal hypotheses for the reversal were eliminated:**
   - Not macro-event contamination (slots are outside those windows)
   - Not high-volatility driving the losses (reversal occurs in low-VIX periods)
   - Not wrong directional positioning relative to VWAP (VWAP confirmation made it worse)

4. **After-hours slots have specific structural properties.** The after-close (ES 16:00 ET) and evening Globex (CL 18:30 ET) windows show consistent reversal that is not explained by macro events, volatility regime, or price direction. The most likely remaining explanation is liquidity-driven: these sessions have wider spreads and thinner books, causing slippage that structurally disadvantages the model's entry logic — but this is untested speculation, not a finding.

---

## POSSIBLE FUTURE RESEARCH

> These are untested hypotheses, not recommendations. Each requires new stakeholder scope discussion before any research or capital commitment.

| Hypothesis | What It Requires | Why It Might Differ |
|---|---|---|
| **Liquidity/spread filter** | Bid-ask spread data at slot entry time | Wide spreads in after-hours sessions may cause systematic slippage — a fixable execution problem, not a signal problem |
| **Rolling PnL gate** | Daily monitoring of rolling 20-day avg PnL per slot | Adaptive stop — skips slot while losing; requires live monitoring infrastructure |
| **Microstructure-based ML** | Tick-level features (imbalance, spread, depth) | At tick resolution, within-slot dynamics may carry exploitable information |
| **Different instruments or markets** | New data pipeline | The reversal pattern is specific to these 4 futures; other instruments untested |

None of these is recommended without first defining what evidence would falsify the hypothesis — the same standard applied throughout this program.

---

## REPOSITORY STATE

All scripts, data, results, and audit trails are committed and pushed.

| File | Description |
|---|---|
| `data/macro_blackout_windows.parquet` | 270 blackout windows (macro filter) |
| `data/vix_daily.parquet` | VIX daily with prior-day lag (vol filter) |
| `data/vwap_hourly.parquet` | 39,498 1-hour VWAP rows ES/NQ/CL (VWAP filter) |
| `results/macro_blackout_21slot_comparison.csv` | Before/after, all 21 slots, macro filter |
| `results/macro_blackout_verdict.txt` | Macro filter verdict |
| `results/vol_regime_21slot_comparison.csv` | Before/after, all 21 slots, vol filter |
| `results/vol_regime_verdict.txt` | Vol filter verdict |
| `results/vwap_21slot_comparison.csv` | Before/after, 17 non-FDAX slots, VWAP filter |
| `results/vwap_verdict.txt` | VWAP filter verdict (corrected) |
| `final_closeout_report.md` | This document |
| `README.md` | Full project documentation |

**Capital deployment:** Not authorized. OOS evidence does not support it.

---

*This report is the definitive, final answer to the research question as stated.*  
*Generated: Aug 23, 2026 | Repo: github.com/giladbi/SC_results_WF_GCP_model*
