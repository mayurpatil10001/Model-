"""
01_fold_structure.py  Step 2
Define 5 rolling chronological folds + holdout.

ROLLING WINDOW DESIGN:
  - Each fold has an independent ~8-month training window and ~2-month test window.
  - Training windows OVERLAP across folds (same trade appears in multiple fold trains).
  - Only the holdout is exclusive and locked away permanently.
  - Fold membership is computed by date-range at fitting time, NOT by exclusive label.

OUTPUTS
  data/fold_assignments.parquet   (trade_date, in_model, is_holdout per trade)
  data/fold_date_ranges.parquet   (fold definitions: train/test start/end dates)
  data/fold_summary.txt
"""
import pandas as pd
import numpy as np

DATA_DIR = "C:/Model-/data"

print("=" * 70)
print("  STEP 2 - Chronological Fold Structure")
print("=" * 70)
print("  Rolling window design: ~8-month train, ~2-month test window per fold.")
print("  Folds OVERLAP — same trade used in multiple training windows.")

df = pd.read_parquet(f"{DATA_DIR}/trades_clean.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
print(f"\n  Total trades: {len(df):,}")
print(f"  Date range:   {df['trade_date'].min().date()} to {df['trade_date'].max().date()}")

HOLDOUT_START = "2025-07-01"
FOLDS = [
    (0, "2024-01-22", "2024-08-31", "2024-09-01", "2024-10-31"),
    (1, "2024-03-01", "2024-10-31", "2024-11-01", "2024-12-31"),
    (2, "2024-05-01", "2024-12-31", "2025-01-01", "2025-02-28"),
    (3, "2024-07-01", "2025-02-28", "2025-03-01", "2025-04-30"),
    (4, "2024-09-01", "2025-04-30", "2025-05-01", "2025-06-30"),
]

# Mark holdout (exclusive, never touched during model development)
df["is_holdout"] = df["trade_date"] >= HOLDOUT_START

# Report fold sizes by filtering on the fly
print()
print(f"  {'Split':20s}  {'Start':12s}  {'End':12s}  {'Trades':>10s}  {'In-model':>9s}")
print("-" * 72)

fold_rows = []
for fold_i, tr_s, tr_e, te_s, te_e in FOLDS:
    for kind, start, end in [("train", tr_s, tr_e), ("test", te_s, te_e)]:
        mask = (df["trade_date"] >= start) & (df["trade_date"] <= end) & ~df["is_holdout"]
        sub  = df[mask]
        n    = len(sub)
        nim  = int(sub["in_model"].sum())
        tag  = f"Fold {fold_i+1} {kind}"
        print(f"  {tag:20s}  {start:12s}  {end:12s}  {n:>10,}  {nim:>9,}")
        fold_rows.append({"fold_i": fold_i, "split": kind,
                          "start": start, "end": end,
                          "n_trades": n, "n_inmodel": nim})

print("-" * 72)
h = df[df["is_holdout"]]
dmax = str(df["trade_date"].max().date())
print(f"  {'HOLDOUT':20s}  {HOLDOUT_START:12s}  {dmax:12s}  {len(h):>10,}  {int(h['in_model'].sum()):>9,}")
fold_rows.append({"fold_i": -1, "split": "holdout",
                  "start": HOLDOUT_START, "end": dmax,
                  "n_trades": len(h), "n_inmodel": int(h["in_model"].sum())})

# Sanity checks
min_train = min(r["n_trades"] for r in fold_rows if r["split"] == "train")
max_train = max(r["n_trades"] for r in fold_rows if r["split"] == "train")
print()
print(f"  Train fold range: {min_train:,} – {max_train:,} trades")
if min_train < 50000:
    print(f"  WARNING: min train fold < 50K — check fold dates.")
else:
    print(f"  Min training-fold size: {min_train:,}  OK (>= 50K threshold)")

# Gap / overlap analysis
non_holdout = df[~df["is_holdout"]]
pre_holdout_range = (non_holdout["trade_date"].min().date(),
                     non_holdout["trade_date"].max().date())
print(f"  Pre-holdout date range: {pre_holdout_range[0]} to {pre_holdout_range[1]}")

# Save: trades get is_holdout flag; fold date ranges saved separately
cols = ["account_name", "symbol", "profit_loss", "trade_date",
        "day_of_week", "bucket_idx", "slot_id", "in_model", "is_holdout"]
df[cols].to_parquet(f"{DATA_DIR}/fold_assignments.parquet", index=False)

fold_dates_df = pd.DataFrame(fold_rows)
fold_dates_df.to_parquet(f"{DATA_DIR}/fold_date_ranges.parquet", index=False)

with open(f"{DATA_DIR}/fold_summary.txt", "w") as f:
    f.write(fold_dates_df.to_string(index=False))
    f.write(f"\n\nMin training-fold trades: {min_train:,}\n")
    f.write(f"Holdout locked: {HOLDOUT_START} onwards\n")

print(f"\n  fold_assignments.parquet  ({len(df):,} rows, is_holdout column added)")
print(f"  fold_date_ranges.parquet  ({len(fold_dates_df)} rows)")
print("\nStep 2 COMPLETE. Run: python 02_fit_model.py")
