"""
Rolling 6-Month Slot Selection — Production-Mode Retrain
Uses only the last 6 months of data (Feb 2026 - Jul 2026) to find currently-active slots.
This is what gets run monthly in production to adapt to regime changes.

Output: C:/Model-/results/rolling_active_slots.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd, numpy as np
from scipy import stats as sps
import os

DATA_DIR    = "C:/Model-/data"
RESULTS_DIR = "C:/Model-/results"

# ── Parameters ────────────────────────────────────────────────────────────────
WINDOW_MONTHS   = 6        # rolling look-back
MIN_TRADES      = 50       # minimum trades per slot in window
BH_Q            = 0.05     # looser Q for shorter window (less data)
MIN_T           = 1.5      # minimum t-stat regardless of BH
CONFIRM_MONTHS  = 2        # must be positive in last N months
CONFIRM_MIN_N   = 30       # minimum trades per month to count

AS_OF_DATE = pd.Timestamp("2026-07-31")  # last date of available data

# ── Load ──────────────────────────────────────────────────────────────────────
df         = pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
slot_index = pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")

df["trade_date"] = pd.to_datetime(df["trade_date"])
df["year_month"] = df["trade_date"].dt.to_period("M")

window_start = AS_OF_DATE - pd.DateOffset(months=WINDOW_MONTHS)
window_df = df[
    (df["trade_date"] >= window_start) &
    (df["trade_date"] <= AS_OF_DATE) &
    df["in_model"]
].copy()

print("=" * 70)
print(f"  ROLLING 6-MONTH RETRAIN  ({window_start.date()} to {AS_OF_DATE.date()})")
print("=" * 70)
print(f"  Trades in window: {len(window_df):,}")
print(f"  Date range:  {window_df['trade_date'].min().date()} to {window_df['trade_date'].max().date()}")

# ── BH-FDR on window ─────────────────────────────────────────────────────────
g = window_df.groupby("slot_id")["profit_loss"]
stats = g.agg(["mean","std","count","sum"]).reset_index()
stats = stats[stats["count"] >= MIN_TRADES].copy()
stats["se"] = stats["std"] / np.sqrt(stats["count"])
stats["t"]  = stats["mean"] / stats["se"]
# one-sided: positive mean only
stats = stats[stats["t"] > 0].copy()
stats["p"] = sps.t.sf(stats["t"], df=stats["count"] - 1)
stats = stats.sort_values("p").reset_index(drop=True)
n = len(stats)
stats["bh_thr"] = BH_Q * (stats.index + 1) / n
bh = stats[(stats["p"] <= stats["bh_thr"]) | (stats["t"] >= MIN_T)].copy()

print(f"\n  Slots with >={MIN_TRADES} trades: {len(stats) + len(stats[stats['t']<=0])}")
print(f"  Positive-mean slots tested:     {len(stats)}")
print(f"  BH-FDR (Q={BH_Q}) or t>={MIN_T}: {len(bh)} candidates")

# ── Monthly confirmation gate ────────────────────────────────────────────────
# Must be positive in at least CONFIRM_MONTHS of the last 3 months
last3_start = pd.Period(AS_OF_DATE, "M") - 2   # 3 months back
monthly = (window_df.groupby(["slot_id","year_month"])["profit_loss"]
             .agg(["mean","count"]).reset_index())
monthly.columns = ["slot_id","year_month","mean_pnl","n"]

def count_positive_recent(sid):
    m = monthly[(monthly["slot_id"]==sid) & (monthly["year_month"]>=last3_start)]
    return ((m["mean_pnl"]>0) & (m["n"]>=CONFIRM_MIN_N)).sum()

bh["pos_recent"] = bh["slot_id"].apply(count_positive_recent)
confirmed = bh[bh["pos_recent"] >= CONFIRM_MONTHS].copy()

print(f"  After confirmation (pos in >={CONFIRM_MONTHS}/3 recent months): {len(confirmed)}")

# ── Annotate ─────────────────────────────────────────────────────────────────
si = slot_index.set_index("slot_id")
day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

def enrich(df_in):
    out = df_in.copy()
    out["symbol"]      = out["slot_id"].map(lambda x: si.loc[x,"symbol"] if x in si.index else "?")
    out["day_of_week"] = out["slot_id"].map(lambda x: si.loc[x,"day_of_week"] if x in si.index else -1)
    out["day_name"]    = out["day_of_week"].map(day_names)
    out["bucket_idx"]  = out["slot_id"].map(lambda x: si.loc[x,"bucket_idx"] if x in si.index else -1)
    return out

bh       = enrich(bh)
confirmed= enrich(confirmed)

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  ALL BH-FDR CANDIDATES (last 6 months, Q={BH_Q} or t>={MIN_T})")
print("=" * 70)
print(f"  {'slot_id':>7} {'sym':>5} {'day':>4} {'bkt':>4} {'mean_pnl':>10} {'t':>7} {'p':>10} {'n':>7} {'pos_rec':>8}")
for _, r in bh.sort_values("t", ascending=False).iterrows():
    marker = " **" if r["slot_id"] in confirmed["slot_id"].values else ""
    print(f"  {int(r['slot_id']):>7} {r['symbol']:>5} {r['day_name']:>4} {int(r['bucket_idx']):>4} "
          f"{r['mean']:>10.2f} {r['t']:>7.3f} {r['p']:>10.2e} {int(r['count']):>7} {int(r['pos_recent']):>8}{marker}")

print()
print("=" * 70)
print(f"  CONFIRMED ACTIVE SLOTS (trade now, as of {AS_OF_DATE.date()})")
print("=" * 70)
if len(confirmed) == 0:
    print("  NO SLOTS CONFIRMED. Do not trade any slots in the current regime.")
else:
    print(f"  {'slot_id':>7} {'symbol':>6} {'day':>4} {'bucket':>6} {'6mo_mean':>10} {'t':>7} {'n_6mo':>7} {'pos_3mo':>8}")
    for _, r in confirmed.sort_values("t", ascending=False).iterrows():
        print(f"  {int(r['slot_id']):>7} {r['symbol']:>6} {r['day_name']:>4} {int(r['bucket_idx']):>6} "
              f"{r['mean']:>10.2f} {r['t']:>7.3f} {int(r['count']):>7} {int(r['pos_recent']):>8}/3")

# ── Monthly detail for confirmed slots ───────────────────────────────────────
if len(confirmed) > 0:
    print()
    print("  Monthly breakdown for confirmed slots:")
    print(f"  {'slot_id':>7} {'sym':>5} {'day':>4} {'bkt':>4} {'month':<8} {'mean_pnl':>10} {'n':>6}")
    for _, r in confirmed.iterrows():
        m = monthly[monthly["slot_id"]==r["slot_id"]].sort_values("year_month")
        for _, mr in m.iterrows():
            arr = " ^" if mr["mean_pnl"]>0 else " v"
            print(f"  {int(r['slot_id']):>7} {r['symbol']:>5} {r['day_name']:>4} {int(r['bucket_idx']):>4} "
                  f"{str(mr['year_month']):<8} {mr['mean_pnl']:>10.2f}{arr} {int(mr['n']):>6}")
        print()

# ── Compute Sortino on the 6-month window for confirmed slots ─────────────────
if len(confirmed) > 0:
    print("  Sortino ratios on 6-month window (confirmed slots only):")
    cand_trades = window_df[window_df["slot_id"].isin(confirmed["slot_id"].tolist())]
    for sid, grp in cand_trades.groupby("slot_id"):
        pnl = grp["profit_loss"].values
        neg = pnl[pnl < 0]
        if len(neg) > 1:
            sor = pnl.mean() / np.std(neg, ddof=1)
            info = confirmed[confirmed["slot_id"]==sid].iloc[0]
            print(f"    slot {int(sid)} {info['symbol']} {info['day_name']} Bkt{int(info['bucket_idx'])}: "
                  f"Sortino={sor:.4f}  mean={pnl.mean():.2f}  n={len(pnl)}")
    # Portfolio
    all_pnl = cand_trades["profit_loss"].values
    neg_all = all_pnl[all_pnl < 0]
    if len(neg_all) > 1:
        port_sor = all_pnl.mean() / np.std(neg_all, ddof=1)
        print(f"\n    Portfolio ({len(confirmed)} slots): Sortino={port_sor:.4f}  "
              f"mean/trade={all_pnl.mean():.2f}  total={all_pnl.sum():.0f}  n={len(all_pnl)}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = confirmed[["slot_id","symbol","day_name","bucket_idx","mean","t","count","pos_recent"]].copy()
out.columns = ["slot_id","symbol","day","bucket_idx","mean_pnl_6mo","t_stat","n_6mo","pos_recent_3mo"]
out.to_csv(f"{RESULTS_DIR}/rolling_active_slots.csv", index=False)
print(f"\n  Saved: {RESULTS_DIR}/rolling_active_slots.csv")
print(f"\nNext retrain due: ~2026-09-01 (using Mar-Aug 2026 data)")
