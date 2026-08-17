
import sqlite3, pandas as pd

DB_PATH="C:/SC_results_WF/trading_platform.db"; RESULTS_DIR="C:/Model-/results"
print("="*70); print("  STEP 6 - Feature Inventory (OUT OF SCOPE for Layer 2)"); print("="*70)
inventory=[
    ("account_name",      "Available","Account family/strategy. Layer 3 grouping candidate."),
    ("side",              "Available","LONG/SHORT. Layer 2 is direction-agnostic."),
    ("duration_minutes",  "NULL",     "Trade holding time. NULL for all rows."),
    ("quantity",          "Available","Contract count. Flat 3 per stakeholder."),
    ("commission",        "NULL",     "NULL for all rows."),
    ("hour_of_day",       "NULL",     "Pre-computed field - NULL. Re-derived in Step 1."),
    ("day_of_week",       "NULL",     "Pre-computed field - NULL. Re-derived in Step 1."),
    ("minute_of_hour_ny", "Zero bug", "All rows=0 in processed_trades. Re-derived in Step 1."),
    ("note/tag metadata", "Not promoted","In source .data files, not in processed_trades."),
    ("gfre_version",      "Uniform",  "All GFRE v3.3. Not a useful covariate."),
]
print()
for field,status,desc in inventory:
    print(f"  {field:25s}  [{status:12s}]  {desc}")
print("\n  All items OUT OF SCOPE for Layer 2. Layer 3 candidates.")
con=sqlite3.connect(DB_PATH)
samp=pd.read_sql_query("SELECT * FROM processed_trades LIMIT 5000",con); con.close()
print("\n  Null-check (n=5000 sample):")
for col in ["side","duration_minutes","commission","hour_of_day","day_of_week","minute_of_hour_ny"]:
    if col in samp.columns: print(f"    {col:25s}  nulls={samp[col].isna().sum()}")
with open(f"{RESULTS_DIR}/feature_inventory.txt","w") as f:
    for field,status,desc in inventory:
        f.write(f"{field} [{status}]: {desc}\n  [OUT OF SCOPE - Layer 3]\n\n")
print("\nStep 6 COMPLETE. Run: python 06_holdout_eval.py")
