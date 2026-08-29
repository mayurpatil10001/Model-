# Gross vs. Net PnL — Diagnostic Report

**Date:** August 26, 2026  
**Question:** Is the OOS reversal a signal problem (wrong directional calls) or an execution problem (slippage/commissions eating profitable gross PnL)?  
**Script:** `scratch/gross_vs_net_analysis.py`  
**Output:** `results/gross_vs_net_21slot.csv`

---

## STEP 1 — Data Inventory

### What exists in the DB

| Field | Table | Status | Notes |
|---|---|---|---|
| `entry_price`, `exit_price` | `processed_trades` | ✅ Present, fully populated | **Actual fill prices**, not order-submission prices |
| `profit_loss` | `processed_trades` | ✅ Present | = `(exit−entry) × qty × multiplier` exactly — **pure gross PnL** |
| `commission` | `processed_trades` | ❌ NULL for all 2,919,411 rows | Never populated in main table |
| `commission` | `rar_trades` | ✅ Present, populated | **$4.20/contract** for NQ (account TM_7, 8,766 rows) |
| `price` (order price) | `sierra_chart_fills` | ✅ Schema exists | Would enable order-price vs fill-price slippage measurement |
| `fill_price` (fill price) | `sierra_chart_fills` | ✅ Schema exists | Table is **0 rows — never imported** |
| Signal/order-submission price | Any table | ❌ Does not exist | No separate "signal price" field anywhere |
| Bid/ask quote at order time | Any table | ❌ Does not exist | No quote data in any table |

### What can be computed

| Comparison | Possible? |
|---|---|
| **Gross PnL** (price difference only) | ✅ = `processed_trades.profit_loss` (already computed in all prior analysis) |
| **Estimated net PnL** (after commission) | ✅ Estimated using `quantity × $4.20/contract` from rar_trades |
| **Fill slippage** (order price vs fill price) | ❌ Not possible — `sierra_chart_fills` is empty |

**PnL formula verification:** Checked 5,000 holdout trades per symbol.  
`profit_loss` matches `(exit−entry)×qty×multiplier` exactly (max diff = $0.00, 100%) for CL, ES, NQ, FDAX.  
The `profit_loss` in `processed_trades` is **confirmed pure gross PnL** — no commission deducted.

> **Note:** The two positive-gross slots observed (Slot 223 CL Sun 18:00, Slot 863 NQ Thu 14:00) should be interpreted with caution: Slot 223 previously showed a March 2026 concentration problem and Slot 863 has t = 1.448 (not significant after Bonferroni correction).

---

## STEP 2 — Two Motivating Cases

Commission assumption: **$4.20/contract** (from `rar_trades`, confirmed for NQ, applied uniformly).

### ES Wednesday 16:00

avg_qty = 1.89 contracts → avg commission = **$7.93/trade**

| Period | Gross mean | Commission | Net mean | Commission as % of gross loss | Verdict |
|---|---|---|---|---|---|
| Cycle 23 fwd (May 2026, n=231) | −$36.31/trade | $7.15/trade | **−$43.45/trade** | 19.7% | **BOTH NEGATIVE — signal problem** |
| Cycle 24 fwd (Jun 2026, n=72) | −$276.56/trade | $10.03/trade | **−$286.60/trade** | 3.6% | **BOTH NEGATIVE — signal problem** |

### CL Tuesday 18:30

avg_qty = 1.57 contracts → avg commission = **$6.59/trade**

| Period | Gross mean | Commission | Net mean | Verdict |
|---|---|---|---|---|
| Cycle 24 fwd (Jun 2026, n=2) | +$150.00/trade | $4.20/trade | **+$145.80/trade** | Both positive (n=2, not significant) |
| Cycle 25 fwd (Jul 2026, n=3) | −$263.33/trade | $9.80/trade | **−$273.13/trade** | **BOTH NEGATIVE — signal problem** |

In both motivating cases, **gross PnL is already deeply negative** before any commission or slippage consideration. Commission adds 3–20% to the losses but does not cause them.

---

## STEP 3 — All 21 Candidate Slots

| slot_id | Symbol | Day | Bucket | Session | Gross $/tr | Gross t | Net $/tr | Comm drag | Comm % of loss |
|---|---|---|---|---|---|---|---|---|---|
| 180 | CL | Thu | 20:00 | after-hours | −$133.33 | −4.451 | −$141.55 | −$8.22 | 6% |
| 223 | CL | Sun | 18:00 | after-hours | **+$45.06** | +2.462 | **+$38.27** | −$6.79 | 15%† |
| 259 | ES | Mon | 12:00 | regular | −$26.95 | −7.062 | −$35.72 | −$8.77 | 33% |
| 260 | ES | Mon | 12:30 | regular | −$41.58 | −10.694 | −$50.29 | −$8.71 | 21% |
| 282 | ES | Tue | 00:00 | after-hours | −$14.64 | −0.912 | −$23.39 | −$8.75 | 60% |
| 357 | ES | Wed | 14:00 | regular | −$30.14 | −7.478 | −$38.40 | −$8.26 | 27% |
| 358 | ES | Wed | 14:30 | regular | −$12.75 | −3.449 | −$21.06 | −$8.31 | 65% |
| 587 | FDAX | Wed | 13:30 | regular | −$82.16 | −0.661 | −$90.29 | −$8.13 | 10% |
| 648 | FDAX | Thu | 23:00 | after-hours | −$415.70 | −2.107 | −$423.44 | −$7.74 | 2% |
| 675 | FDAX | Fri | 12:30 | regular | −$422.96 | −3.339 | −$431.13 | −$8.17 | 2% |
| 677 | FDAX | Fri | 13:30 | regular | −$97.21 | −0.881 | −$104.65 | −$7.44 | 8% |
| 712 | NQ | Mon | 09:30 | regular | −$42.86 | −5.061 | −$49.56 | −$6.70 | 16% |
| 714 | NQ | Mon | 10:30 | regular | −$17.51 | −1.450 | −$24.43 | −$6.92 | 40% |
| 715 | NQ | Mon | 11:00 | regular | −$81.37 | −6.169 | −$88.47 | −$7.10 | 9% |
| 716 | NQ | Mon | 11:30 | regular | −$67.73 | −4.838 | −$74.95 | −$7.22 | 11% |
| 718 | NQ | Mon | 12:30 | regular | −$12.05 | −0.839 | −$18.72 | −$6.67 | 55% |
| 719 | NQ | Mon | 13:00 | regular | −$8.08 | −0.646 | −$14.73 | −$6.65 | 82% |
| 854 | NQ | Thu | 09:30 | regular | −$14.90 | −1.409 | −$21.50 | −$6.60 | 44% |
| 863 | NQ | Thu | 14:00 | regular | **+$23.01** | +1.448 | **+$16.32** | −$6.69 | 29%‡ |
| 901 | NQ | Fri | 09:30 | regular | −$1.57 | −0.117 | −$8.05 | −$6.48 | 413% |
| 906 | NQ | Fri | 12:00 | regular | −$36.71 | −2.888 | −$43.41 | −$6.70 | 18% |

† Slot 223 previously identified as a March 2026 concentration artifact.  
‡ Slot 863 gross t = 1.448, not significant after Bonferroni correction.

### Summary statistics

| Metric | Value |
|---|---|
| Slots both gross and net **negative** | **19 / 21** |
| Slots gross **positive** but net **negative** (commission the cause) | **0 / 21** |
| Slots both positive | 2 / 21 (both not cleanly validated — see footnotes) |
| Avg gross mean PnL | −$71.05/trade |
| Avg commission drag | −$7.48/trade |
| Avg net mean PnL | −$78.53/trade |

### After-hours concentration test

The earlier hypothesis was that after-hours sessions (ES 16:00 ET, CL 18:30 ET) might have disproportionate execution costs due to thin liquidity.

| Session | Slots | Avg gross PnL | Avg commission drag |
|---|---|---|---|
| After-hours (16:00 ET+) | 4 | −$129.65/trade | −$7.88/trade |
| Regular hours | 17 | −$57.27/trade | −$7.38/trade |

**Hypothesis: refuted.** Commission drag is nearly identical across sessions ($7.88 vs $7.38). After-hours slots are far more negative at gross (−$129.65 vs −$57.27), so the larger total loss is entirely attributable to worse directional performance, not execution cost.

---

## STEP 4 — VERDICT

### OUTCOME 3: Execution cost does not explain the reversal.

**19 of 21 candidate slots are negative at the gross PnL level** — before commission, before any slippage consideration. The directional calls are wrong, and commission adds ~$7.50/trade (~10% of average gross loss) on top.

**Not a single slot** (0/21) shows the pattern that would indicate execution is the problem: gross PnL positive but net PnL negative after subtracting commission. Commission uniformly makes things incrementally worse, as expected, but is not the cause of the reversal.

**What this means for the overall research conclusion:**  
The final_closeout_report.md conclusion stands unchanged. The reversal is a **signal problem** — calendar slot identity does not contain durable predictive information — not an execution problem that could be fixed with better order routing or commission reduction.

### What remains unmeasurable

Fill-level slippage (order-submission price vs actual fill price) cannot be computed because `sierra_chart_fills` is empty. This is the one remaining unknown. However, slippage would only *add* to already deeply negative gross losses. For slippage to "explain" the reversal, it would need to be both:  
(a) large enough to turn gross-positive signals into net-negative ones, AND  
(b) the gross PnL would need to be positive in the first place.  
Neither condition holds for 19 of 21 slots.

**For those two slots (223, 863) that are gross-positive:** Their sample sizes, lack of statistical significance after Bonferroni correction, and the previously identified March 2026 concentration in Slot 223 prevent these from constituting evidence of a real edge.

---

*Script: `scratch/gross_vs_net_analysis.py` | Output: `results/gross_vs_net_21slot.csv`*
