# NUMERICAL RECONCILIATION LOG
**Generated for Final Documentation Package**
Every headline number is derived directly from persisted source files on disk.

---

### 1. Static System OOS Portfolio PnL
- **Source File:** `results/walkforward_portfolio_monthly.csv`
- **Exact Total OOS PnL:** `$-1,528,785.01`
- **Total Months:** `13` months
- **Monthly PnL Range:** `$-274,767.51` to `$28,447.50`
- **Discrepancy Correction Note:** Confirmed exact figure is `-$1,528,785.01` (correcting historical -$2.1M typo from preliminary unaggregated note).

### 2. Rolling Retrain Chained Portfolio PnL
- **Source File:** `results/rolling_retrain_chain_monthly.csv`
- **Exact Chained OOS PnL:** `$-2,569,959.98`
- **Evaluated Months:** `25` forward months

### 3. Conditioning Filters Full Comparison (Portfolio Level)
- **Source File:** `results/three_filters_full_comparison.csv`
| Scenario | Trades (N) | Mean/Trade | Bootstrap Mean | 95% CI Range | P(Loss) | Worst Month % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unfiltered** | 59,810 | $-27.16 | $-27.20 | [$-36.29, $-18.02] | 100.0% | 17.5% |
| **Macro alone** | 53,177 | $-29.16 | $-29.28 | [$-39.18, $-18.82] | 100.0% | 18.7% |
| **VIX alone** | 45,096 | $-29.45 | $-29.41 | [$-38.73, $-19.84] | 100.0% | 19.7% |
| **VWAP alone** | 28,848 | $-25.14 | $-25.28 | [$-39.91, $-10.88] | 99.7% | 26.8% |
| **All 3 combined** | 18,619 | $-34.07 | — | — | — | — |

### 4. Gross vs. Net Friction Impact
- **Source File:** `results/gross_vs_net_21slot.csv`
- **Average Friction per Trade:** `$4.50 – $6.20` across instruments
- **Key Finding:** Gross PnL is also strongly negative across all candidate slots, proving friction/commissions are NOT the cause of signal failure.

### 5. Benjamini-Hochberg Candidate Slots Count
- **Source File:** `results/bh_fdr_candidates.csv`
- **Total Candidates Identified:** `21` slots
- **Per Asset:** NQ: `10`, ES: `5`, CL: `2`, FDAX: `4`
