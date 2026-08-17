import os as _os
_os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

"""
06_holdout_eval.py  Step 7
Fit final model on all 5 training folds combined.
Evaluate EXACTLY ONCE on holdout (2025-07-01 to 2026-07-17). No retuning.
Pass bar: lower bound of 90% CI for posterior-predictive Sortino > 3.0
Step 7a: flat 3 contracts per trade, NO differential allocation.
"""
import pandas as pd, numpy as np, pymc as pm, arviz as az
import scipy.stats as sps
import warnings, os
warnings.filterwarnings("ignore")

DATA_DIR="C:/Model-/data"; RESULTS_DIR="C:/Model-/results"
HOLDOUT_GATE=3.0; CHAINS=4; TUNE=1000; DRAWS=1000; TARGET_ACCEPT=0.9
PP_DRAWS=2000; P0_FRAC=0.10
DAY_NAMES=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

print("="*70); print("  STEP 7 - Final Holdout Evaluation"); print("="*70)

cand_path=f"{RESULTS_DIR}/candidates.parquet"
if not os.path.exists(cand_path): print("ERROR: candidates.parquet missing. Run Steps 3-5."); exit(1)
candidates=pd.read_parquet(cand_path); n_cands=len(candidates)
print(f"  Candidates: {n_cands}")
if n_cands==0: print("  ZERO candidates. FINAL RESULT: Null."); exit(0)

null_rpt=f"{RESULTS_DIR}/null_test_report.txt"
if os.path.exists(null_rpt):
    with open(null_rpt) as f: nt=f.read()
    if "FAILED" in nt:
        print("  WARNING: Null test FAILED. Results not distinguishable from noise. Proceeding with caution.")

df_all=pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
slot_index=pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")

def sortino_ratio(pnl):
    pnl=np.asarray(pnl,dtype=float); neg=pnl[pnl<0]
    if len(neg)<2: return np.nan
    ds=np.std(neg,ddof=1); return float(np.mean(pnl)/ds) if ds>0 else np.nan

def slot_summaries(df_s):
    g=df_s[df_s["in_model"]].groupby("slot_id")["profit_loss"]
    a=g.agg(["mean","std","count"]).reset_index()
    a.columns=["slot_id","slot_mean","slot_std","slot_n"]; a=a[a["slot_n"]>=2].copy()
    a["slot_std"]=a["slot_std"].clip(lower=1e-6); a["slot_se"]=a["slot_std"]/np.sqrt(a["slot_n"])
    return a

df_all["trade_date"]=pd.to_datetime(df_all["trade_date"])
train_df=df_all[~df_all["is_holdout"] & df_all["in_model"]].copy()
holdout_df=df_all[df_all["is_holdout"]].copy()
print(f"  Training: {len(train_df):,}  Holdout: {len(holdout_df):,}")

agg=slot_summaries(train_df)
m=agg.merge(slot_index[["slot_id","symbol","day_of_week","bucket_idx"]],on="slot_id")
syms=sorted(m["symbol"].unique()); days=sorted(m["day_of_week"].unique())
bkts=sorted(m["bucket_idx"].unique()); sbp=sorted(set(zip(m["symbol"],m["bucket_idx"])))
se={s:i for i,s in enumerate(syms)}; de={d:i for i,d in enumerate(days)}
be={b:i for i,b in enumerate(bkts)}; sbe={p:i for i,p in enumerate(sbp)}
si=m["symbol"].map(se).values.astype(int); di=m["day_of_week"].map(de).values.astype(int)
bi=m["bucket_idx"].map(be).values.astype(int)
sbi=np.array([sbe.get((s,b),0) for s,b in zip(m["symbol"],m["bucket_idx"])],dtype=int)
yo=m["slot_mean"].values.astype(float); ys=m["slot_se"].values.astype(float)
N=len(m); D=N; p0=max(1,int(P0_FRAC*D))
sig_e=float(np.median(ys))*float(np.sqrt(np.median(m["slot_n"])))
t0=(p0/(D-p0+1e-9))*(sig_e/np.sqrt(N))

print("\n  Fitting final model on combined training data ...")
with pm.Model() as model:
    a_=pm.Normal("alpha",mu=0,sigma=100)
    ts=pm.HalfCauchy("tau_sym",beta=t0);  ls=pm.HalfCauchy("lam_sym",beta=1,shape=len(syms)); zs=pm.Normal("z_sym",mu=0,sigma=1,shape=len(syms))
    td=pm.HalfCauchy("tau_day",beta=t0);  ld=pm.HalfCauchy("lam_day",beta=1,shape=len(days)); zd=pm.Normal("z_day",mu=0,sigma=1,shape=len(days))
    tb=pm.HalfCauchy("tau_bkt",beta=t0);  lb=pm.HalfCauchy("lam_bkt",beta=1,shape=len(bkts)); zb=pm.Normal("z_bkt",mu=0,sigma=1,shape=len(bkts))
    tsb=pm.HalfCauchy("tau_sb",beta=t0);  lsb=pm.HalfCauchy("lam_sb",beta=1,shape=len(sbp)); zsb=pm.Normal("z_sb",mu=0,sigma=1,shape=len(sbp))
    tsg=pm.HalfNormal("tau_sig",sigma=1); zsg=pm.Normal("z_sig",mu=0,sigma=1,shape=N); ls0=pm.Normal("log_sig0",mu=np.log(200),sigma=1)
    mu=a_+zs[si]*ts*ls[si]+zd[di]*td*ld[di]+zb[bi]*tb*lb[bi]+zsb[sbi]*tsb*lsb[sbi]
    _=pm.Normal("obs",mu=mu,sigma=ys,observed=yo)
    trace_f=pm.sample(draws=DRAWS,tune=TUNE,chains=CHAINS,target_accept=TARGET_ACCEPT,
                      nuts_sampler='numpyro',
                        return_inferencedata=True,progressbar=True)

trace_f.to_netcdf(f"{RESULTS_DIR}/final_model_trace.nc")
post=trace_f.posterior
ap=post["alpha"].values.ravel()
zs_p=post["z_sym"].values.reshape(-1,len(syms)); ts_p=post["tau_sym"].values.ravel(); ls_p=post["lam_sym"].values.reshape(-1,len(syms))
zd_p=post["z_day"].values.reshape(-1,len(days)); td_p=post["tau_day"].values.ravel(); ld_p=post["lam_day"].values.reshape(-1,len(days))
zb_p=post["z_bkt"].values.reshape(-1,len(bkts)); tb_p=post["tau_bkt"].values.ravel(); lb_p=post["lam_bkt"].values.reshape(-1,len(bkts))
zsb_p=post["z_sb"].values.reshape(-1,len(sbp));  tsb_p=post["tau_sb"].values.ravel(); lsb_p=post["lam_sb"].values.reshape(-1,len(sbp))
ls0_p=post["log_sig0"].values.ravel(); tsg_p=post["tau_sig"].values.ravel()
zsg_p=post["z_sig"].values.reshape(-1,N); m["_i"]=np.arange(N)

print(f"\n  Evaluating {n_cands} candidates on holdout (EXACTLY ONCE) ...")
rows=[]; port_pnl=[]
for _,cand in candidates.iterrows():
    sid=int(cand["slot_id"]); rm=m[m["slot_id"]==sid]
    if len(rm)==0: continue
    i=int(rm["_i"].iloc[0])
    mu_s=(ap+zs_p[:,si[i]]*ts_p*ls_p[:,si[i]]+zd_p[:,di[i]]*td_p*ld_p[:,di[i]]
             +zb_p[:,bi[i]]*tb_p*lb_p[:,bi[i]]+zsb_p[:,sbi[i]]*tsb_p*lsb_p[:,sbi[i]])
    sig_s=np.exp(ls0_p+zsg_p[:,i]*tsg_p)
    slot_h=holdout_df[(holdout_df["slot_id"]==sid)&holdout_df["in_model"]]
    pnl=slot_h["profit_loss"].values
    rng=np.random.default_rng(42+sid); sr_d=[]
    for _ in range(PP_DRAWS):
        idx=rng.integers(0,len(mu_s))
        sim=sps.t.rvs(df=5,loc=mu_s[idx],scale=sig_s[idx],size=max(len(pnl),50),random_state=rng)
        neg=sim[sim<0]
        if len(neg)>=2 and np.std(neg,ddof=1)>0: sr_d.append(float(np.mean(sim)/np.std(neg,ddof=1)))
    sr_mean=float(np.mean(sr_d)) if sr_d else np.nan
    sr_lo=float(np.percentile(sr_d,5)) if sr_d else np.nan; sr_hi=float(np.percentile(sr_d,95)) if sr_d else np.nan
    n_h=len(pnl); tot=float(np.sum(pnl)) if n_h>0 else np.nan; avg=float(np.mean(pnl)) if n_h>0 else np.nan
    wr=float(np.mean(pnl>0)) if n_h>0 else np.nan
    gw=float(np.sum(pnl[pnl>0])) if n_h>0 and np.any(pnl>0) else 0.0
    gl=float(abs(np.sum(pnl[pnl<0]))) if n_h>0 and np.any(pnl<0) else 1e-9
    pf=gw/gl; act_sr=sortino_ratio(pnl); passes=np.isfinite(sr_lo) and sr_lo>HOLDOUT_GATE
    meta=slot_index[slot_index["slot_id"]==sid].iloc[0]
    rows.append({"slot_id":sid,"symbol":meta["symbol"],"day_name":DAY_NAMES[int(meta["day_of_week"])],
                 "bucket_label":meta["bucket_label"],"holdout_n":n_h,
                 "total_profit":tot,"avg_profit":avg,"win_rate":wr,"profit_factor":pf,
                 "actual_sortino":act_sr,"sortino_pp_mean":sr_mean,
                 "sortino_lo90":sr_lo,"sortino_hi90":sr_hi,"passes_sortino3":passes})
    port_pnl.extend(pnl.tolist())

res_df=pd.DataFrame(rows); res_df.to_parquet(f"{RESULTS_DIR}/holdout_per_slot.parquet",index=False)
print("\n"+"="*70); print("  HOLDOUT RESULTS - PER SLOT"); print("="*70)
print(f"  {'id':>6s}  {'sym':>5s}  {'day':>4s}  {'time':>5s}  {'n':>6s}  {'total$':>10s}  {'avg$':>7s}  {'WR':>5s}  {'PF':>5s}  {'lo90':>6s}  {'PASS':>5s}")
print("-"*80); n_pass=0
for _,r in res_df.iterrows():
    mk="YES" if r["passes_sortino3"] else "no"; n_pass+=int(r["passes_sortino3"])
    print(f"  {int(r['slot_id']):>6d}  {r['symbol']:>5s}  {r['day_name']:>4s}  {r['bucket_label']:>5s}  "
          f"{int(r['holdout_n']):>6d}  ${r['total_profit']:>9,.0f}  ${r['avg_profit']:>6,.1f}  "
          f"{r['win_rate']:>4.1%}  {r['profit_factor']:>5.2f}  {r['sortino_lo90']:>6.3f}  {mk:>5s}")
print(f"\n  Slots passing lower-bound > {HOLDOUT_GATE}: {n_pass} / {len(res_df)}")
print(f"  [Step 7a] Flat 3 contracts per trade. NO differential allocation.")
if port_pnl:
    pp_arr=np.array(port_pnl); pt=float(np.sum(pp_arr)); pa=float(np.mean(pp_arr)); pn=len(pp_arr)
    pwr=float(np.mean(pp_arr>0))
    pgw=float(np.sum(pp_arr[pp_arr>0])) if np.any(pp_arr>0) else 0
    pgl=float(abs(np.sum(pp_arr[pp_arr<0]))) if np.any(pp_arr<0) else 1e-9
    ppf=pgw/pgl; psr=sortino_ratio(pp_arr)
    print("\n"+"="*70); print("  STEP 7b - COMBINED PORTFOLIO"); print("="*70)
    print(f"  Trades:{pn:,}  Total:${pt:,.2f}  Avg:${pa:,.2f}  WR:{pwr:.1%}  PF:{ppf:.3f}  Sortino:{psr:.3f}")
    with open(f"{RESULTS_DIR}/holdout_portfolio.txt","w") as f:
        f.write(f"Trades: {pn:,}\nTotal profit: ${pt:,.2f}\nAvg/trade: ${pa:,.2f}\n"
                f"Win rate: {pwr:.1%}\nProfit factor: {ppf:.3f}\nSortino: {psr:.3f}\n"
                f"Flat sizing: 3 contracts (Step 7a).\n")
print("\nStep 7 COMPLETE. Run: python 07_report.py")
