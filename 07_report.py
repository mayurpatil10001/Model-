
import pandas as pd, numpy as np, os
from pathlib import Path

DATA_DIR="C:/Model-/data"; RESULTS_DIR="C:/Model-/results"

def txt(p,d="Not yet generated."):
    return Path(p).read_text(encoding="utf-8") if Path(p).exists() else d

perm=txt(f"{DATA_DIR}/permutation_space.txt"); folds=txt(f"{DATA_DIR}/fold_summary.txt")
sel=txt(f"{RESULTS_DIR}/selection_report.txt"); null=txt(f"{RESULTS_DIR}/null_test_report.txt")
feat=txt(f"{RESULTS_DIR}/feature_inventory.txt"); port=txt(f"{RESULTS_DIR}/holdout_portfolio.txt","No portfolio yet.")
spec=txt("C:/Model-/model_spec.md")
diags="".join(f"### Fold {i+1}\n```\n{txt(f'{RESULTS_DIR}/fold{i}_diagnostics.txt',f'Fold {i+1} not run.')}\n```\n" for i in range(5))
cand_tbl=pd.read_parquet(f"{RESULTS_DIR}/candidates.parquet").to_string(index=False) if Path(f"{RESULTS_DIR}/candidates.parquet").exists() else "Not run."
hold_tbl=pd.read_parquet(f"{RESULTS_DIR}/holdout_per_slot.parquet").to_string(index=False) if Path(f"{RESULTS_DIR}/holdout_per_slot.parquet").exists() else "Not run."
null_tbl=pd.read_parquet(f"{RESULTS_DIR}/null_test_summary.parquet").to_string(index=False) if Path(f"{RESULTS_DIR}/null_test_summary.parquet").exists() else "Not run."

report=f"""# Slot-Selection Model - Final Report
## Bayesian Hierarchical Partial-Pooling | Day-of-Week x Time-of-Day x Symbol
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---
## Step 1 - Model Specification & Permutation Space
### model_spec.md (committed before fitting)
```
{spec}
```
### Permutation Space
```
{perm}
```
---
## Step 2 - Fold Structure
```
{folds}
```
---
## Step 3 - Convergence Diagnostics
{diags}
---
## Step 4 - Candidate Selection
```
{sel}
```
Candidates:
```
{cand_tbl}
```
---
## Step 5 - Permutation Null Test
```
{null}
```
Null distribution:
```
{null_tbl}
```
---
## Step 6 - Feature Inventory (Out of Scope - Layer 3)
```
{feat}
```
---
## Step 7 - Holdout Results (2025-07-01 to 2026-07-17)
Pass bar: lower bound of 90% CI for posterior-predictive Sortino > 3.0
### Per-Slot
```
{hold_tbl}
```
### Step 7b - Combined Portfolio
```
{port}
```
Step 7a confirmation: Flat 3 contracts per trade. No differential allocation.
---
## Final Summary
Total permutations tested, survivors, and holdout metrics shown above with
credible intervals. If zero candidates survived any gate, that is the result.
A rigorous null result is more valuable than a false positive.

Data: processed_trades | GFRE v3.3 | 2,919,411 trades | PyMC v5 | Horseshoe priors
"""
out=f"{RESULTS_DIR}/FINAL_REPORT.md"
Path(out).write_text(report,encoding="utf-8")
print(f"Step 8 COMPLETE. Report: {out}")
