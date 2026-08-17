
import pandas as pd, numpy as np, scipy.stats as sps, os
from statsmodels.stats.multitest import multipletests

DATA_DIR="C:/Model-/data"; RESULTS_DIR="C:/Model-/results"
LB_GATE=1.0; MIN_FOLDS=3; BH_Q=0.01; N_FOLDS=5
DAY_NAMES=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

print("="*70); print("  STEP 4 - Candidate Selection"); print("="*70)

all_folds=pd.concat([pd.read_parquet(f"{RESULTS_DIR}/fold{i}_posteriors.parquet") for i in range(N_FOLDS)],ignore_index=True)
slot_index=pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")
df_clean=pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
print(f"  {N_FOLDS} fold posteriors, {len(all_folds):,} slot-fold rows.")

all_folds["passes"]=all_folds["sortino_lo90"].fillna(-99)>LB_GATE
print(f"\n  Primary gate (sortino_lo90 > {LB_GATE}) per fold:")
for i in range(N_FOLDS):
    fd=all_folds[all_folds["fold"]==i]
    print(f"    Fold {i+1}: {int(fd['passes'].sum())} / {len(fd)} slots pass")

surv=all_folds.groupby("slot_id")["passes"].sum().reset_index()
surv.columns=["slot_id","folds_passed"]
cands=surv[surv["folds_passed"]>=MIN_FOLDS]["slot_id"].tolist()
print(f"\n  Cross-fold (>={MIN_FOLDS}/5): {len(cands)} candidates")

def sortino_ratio(pnl):
    neg=pnl[pnl<0]
    if len(neg)<2: return np.nan
    ds=np.std(neg,ddof=1)
    return float(np.mean(pnl)/ds) if ds>0 else np.nan

fold_dates_c=pd.read_parquet(f"{DATA_DIR}/fold_date_ranges.parquet")
df_clean["trade_date"]=pd.to_datetime(df_clean["trade_date"])
# Union of all training windows
train_masks=[
    (df_clean["trade_date"]>=row["start"])&(df_clean["trade_date"]<=row["end"])
    for _,row in fold_dates_c[fold_dates_c["split"]=="train"].iterrows()
]
import functools, operator
train_mask_union=functools.reduce(operator.or_, train_masks)
train_df=df_clean[train_mask_union & df_clean["in_model"] & ~df_clean["is_holdout"]].drop_duplicates().copy()
ss=train_df.groupby("slot_id")["profit_loss"].agg(["mean","std","count"]).reset_index()
ss.columns=["slot_id","raw_mean","raw_std","raw_n"]
ss=ss[ss["raw_n"]>=5].copy()
ss["t_stat"]=ss["raw_mean"]/(ss["raw_std"]/np.sqrt(ss["raw_n"]))
dfdummy=max(int(ss["raw_n"].median())-1,1)
ss["p_val"]=ss["t_stat"].apply(lambda t:float(2*sps.t.sf(abs(t),df=dfdummy)))
rej,padj,_,_=multipletests(ss["p_val"],alpha=BH_Q,method="fdr_bh")
ss["bh_reject"]=rej; ss["p_adj"]=padj
flat_sr=train_df.groupby("slot_id")["profit_loss"].apply(sortino_ratio).reset_index()
flat_sr.columns=["slot_id","flat_sortino"]
ss=ss.merge(flat_sr,on="slot_id",how="left")
bh_cands=ss[ss["bh_reject"]]["slot_id"].tolist()
print(f"  BH-FDR (Q={BH_Q}) candidates: {len(bh_cands)}")

cand_set=set(cands); bh_set=set(bh_cands)
agree=cand_set&bh_set; hier_only=cand_set-bh_set; bh_only=bh_set-cand_set
print(f"\n  Agreement: both={len(agree)}  hier_only={len(hier_only)} [monitor]  bh_only={len(bh_only)} [over-fit risk]")

if not cands:
    print("\n  RESULT: ZERO candidates survived. Null result.")
    pd.DataFrame().to_parquet(f"{RESULTS_DIR}/candidates.parquet",index=False)
else:
    agg2=all_folds[all_folds["slot_id"].isin(cands)].groupby("slot_id").agg(
        folds_passed=("passes","sum"),sortino_mean_avg=("sortino_mean","mean"),
        sortino_lo90_min=("sortino_lo90","min"),sortino_lo90_avg=("sortino_lo90","mean"),
        p_mu_pos_avg=("p_mu_pos","mean"),mu_post_mean_avg=("mu_post_mean","mean"),
        n_trades_total=("n_trades","sum")).reset_index()
    agg2=agg2.merge(slot_index[["slot_id","symbol","day_of_week","bucket_label","n_trades"]],on="slot_id")
    agg2["bh_agrees"]=agg2["slot_id"].isin(agree)
    agg2["hierarchical_only"]=agg2["slot_id"].isin(hier_only)
    agg2=agg2.merge(ss[["slot_id","flat_sortino"]],on="slot_id",how="left")
    agg2["day_name"]=agg2["day_of_week"].apply(lambda d:DAY_NAMES[d])
    candidates_df=agg2.sort_values("sortino_lo90_avg",ascending=False)
    print(f"\n  FINAL CANDIDATES ({len(candidates_df)}):")
    print(f"  {'id':>7s}  {'sym':>5s}  {'day':>4s}  {'time':>5s}  {'folds':>5s}  {'lo90_avg':>8s}  {'BH':>3s}  {'hier':>5s}")
    print("-"*60)
    for _,r in candidates_df.iterrows():
        print(f"  {int(r['slot_id']):>7d}  {r['symbol']:>5s}  {r['day_name']:>4s}  {r['bucket_label']:>5s}  "
              f"{int(r['folds_passed']):>5d}  {r['sortino_lo90_avg']:>8.3f}  "
              f"{'yes' if r['bh_agrees'] else '---':>3s}  {'YES' if r['hierarchical_only'] else '---':>5s}")
    candidates_df.to_parquet(f"{RESULTS_DIR}/candidates.parquet",index=False)
    with open(f"{RESULTS_DIR}/selection_report.txt","w") as f:
        f.write(f"Primary gate: sortino_lo90>{LB_GATE} in >={MIN_FOLDS}/5 folds\n")
        f.write(f"Candidates: {len(cands)}  BH: {len(bh_cands)}  Agree: {len(agree)}  Hier-only: {len(hier_only)}\n\n")
        f.write(candidates_df.to_string(index=False))
print("\nStep 4 COMPLETE. Run: python 04_permutation_null.py")
