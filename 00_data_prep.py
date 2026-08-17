
import sqlite3, os
import pandas as pd
import numpy as np

DB_PATH         = "C:/SC_results_WF/trading_platform.db"
OUT_DIR         = "C:/Model-/data"
SYMBOLS         = ["ES", "NQ", "FDAX", "CL"]
MIN_SLOT_TRADES = 20
NY_TZ           = "America/New_York"
DAY_NAMES       = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

print("=" * 70)
print("  STEP 1 - Data Preparation & Permutation Space Definition")
print("=" * 70)

print("\nLoading processed_trades ...")
sym_list = ",".join(f"'{s}'" for s in SYMBOLS)
con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    f"SELECT account_name, symbol, profit_loss, entry_time, exit_time, side, quantity "
    f"FROM processed_trades WHERE symbol IN ({sym_list})", con)
con.close()
print(f"  Raw rows loaded: {len(df):,}")

print("\nFiltering ...")
n_before = len(df)
df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True, format="ISO8601")
df["exit_dt"]  = pd.to_datetime(df["exit_time"], utc=True, format="ISO8601")
dur_s = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds().abs()
mask_zero = dur_s < 1.0
print(f"  Zero-duration (<1s):             {mask_zero.sum():,}")
df = df[~mask_zero].copy(); dur_s = dur_s[~mask_zero]
mask_ghost = (df["profit_loss"] == 0.0) & (dur_s < 5.0)
print(f"  Zero-PnL micro-duration (<5s):   {mask_ghost.sum():,}")
df = df[~mask_ghost].copy()
print(f"  Rows after filtering: {len(df):,}  (removed {n_before - len(df):,})")

print("\nConverting UTC -> America/New_York ...")
df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True, format="ISO8601")
df["entry_ny"] = df["entry_dt"].dt.tz_convert(NY_TZ)

fdax_hours = df[df["symbol"] == "FDAX"]["entry_ny"].dt.hour.value_counts().sort_index()
print("  FDAX entry-hour (NY time), top 8:")
for h, cnt in fdax_hours.nlargest(8).items():
    flag = "  <-- UNEXPECTED" if not (1 <= h <= 14) else ""
    print(f"    {h:02d}:xx  {cnt:8,}{flag}")
print("  [Expected bulk 02:00-14:00 ET for FDAX.]")

print("\nDeriving day_of_week and 30-min bucket ...")
df["day_of_week"]  = df["entry_ny"].dt.dayofweek
df["hour"]         = df["entry_ny"].dt.hour
df["minute"]       = df["entry_ny"].dt.minute
df["bucket_idx"]   = df["hour"] * 2 + (df["minute"] >= 30).astype(int)
df["bucket_label"] = (df["hour"].apply(lambda h: f"{h:02d}") + ":" +
                      df["minute"].apply(lambda m: "00" if m < 30 else "30"))
df["trade_date"]   = df["entry_ny"].dt.date.astype(str)

print("\nBuilding slot index ...")
slot_counts = (df.groupby(["symbol","day_of_week","bucket_idx","bucket_label"])
               .size().reset_index(name="n_trades"))
slot_counts = slot_counts.sort_values(["symbol","day_of_week","bucket_idx"]).reset_index(drop=True)
slot_counts["slot_id"]  = slot_counts.index
slot_counts["in_model"] = slot_counts["n_trades"] >= MIN_SLOT_TRADES
df = df.merge(slot_counts[["symbol","day_of_week","bucket_idx","slot_id","in_model"]],
              on=["symbol","day_of_week","bucket_idx"], how="left")

print()
print("=" * 70)
print("  PERMUTATION SPACE REPORT")
print("=" * 70)
print(f"  {'Symbol':8s}  {'Days':>6s}  {'Buckets':>8s}  {'Total':>8s}  {'In-model':>8s}  {'Sparse':>7s}")
print("-" * 60)
t_all = t_in = t_sp = 0
for sym in SYMBOLS:
    sub   = slot_counts[slot_counts["symbol"] == sym]
    n_all = len(sub); n_in = int(sub["in_model"].sum()); n_sp = n_all - n_in
    days  = sub["day_of_week"].nunique(); bpd = sub["bucket_idx"].nunique()
    print(f"  {sym:8s}  {days:>6d}  {bpd:>8d}  {n_all:>8,}  {n_in:>8,}  {n_sp:>7,}")
    t_all += n_all; t_in += n_in; t_sp += n_sp
print("-" * 60)
print(f"  {'TOTAL':8s}  {'':>6s}  {'':>8s}  {t_all:>8,}  {t_in:>8,}  {t_sp:>7,}")
print(f"\n  Full permutation space : {t_all:,}")
print(f"  In-model permutations  : {t_in:,} (>= {MIN_SLOT_TRADES} trades)")
print(f"  Excluded (sparse)      : {t_sp:,}")
for sym in SYMBOLS:
    sub = slot_counts[slot_counts["symbol"] == sym]
    print(f"  {sym} active days: {[DAY_NAMES[d] for d in sorted(sub['day_of_week'].unique())]}")

print("\nSaving ...")
cols = ["account_name","symbol","profit_loss","trade_date",
        "day_of_week","bucket_idx","bucket_label","slot_id","in_model","side","quantity"]
df[cols].to_parquet(f"{OUT_DIR}/trades_clean.parquet", index=False)
slot_counts.to_parquet(f"{OUT_DIR}/slot_index.parquet", index=False)
with open(f"{OUT_DIR}/permutation_space.txt", "w") as f:
    f.write(f"Full permutation space:  {t_all}\n")
    f.write(f"In-model permutations:   {t_in}\n")
    f.write(f"Excluded (too sparse):   {t_sp}\n")
    f.write(f"MIN_SLOT_TRADES floor:   {MIN_SLOT_TRADES}\n")
print(f"  trades_clean.parquet  ({len(df):,} rows)")
print(f"  slot_index.parquet    ({len(slot_counts):,} slots)")
print("\nStep 1 COMPLETE. Run: python 01_fold_structure.py")
