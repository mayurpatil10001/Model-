
"""
04_permutation_null.py  Step 5
Shuffle profit_loss within each fold (>=20 seeds). Same pipeline as Steps 1-4.
If real candidates fall within null 95th percentile -> failed search, stop.
"""
import pandas as pd, numpy as np, pymc as pm, arviz as az
import scipy.stats as sps
import warnings, os
warnings.filterwarnings("ignore")

DATA_DIR="C:/Model-/data"; RESULTS_DIR="C:/Model-/results"
N_RUNS=20; LB_GATE=1.0; MIN_FOLDS=3; N_FOLDS=5
CHAINS=2; TUNE=500; DRAWS=500; P0_FRAC=0.10

print("="*70); print(f"  STEP 5 - Permutation Null Test ({N_RUNS} runs)"); print("="*70)

df=pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
fold_dates=pd.read_parquet(f"{DATA_DIR}/fold_date_ranges.parquet")
slot_index=pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")
df["trade_date"]=pd.to_datetime(df["trade_date"])
df=df[df["in_model"] & ~df["is_holdout"]].copy()
n_real=len(pd.read_parquet(f"{RESULTS_DIR}/candidates.parquet")) if os.path.exists(f"{RESULTS_DIR}/candidates.parquet") else 0
print(f"  Real candidates from Step 4: {n_real}")

def slot_summaries(df_s):
    g=df_s.groupby("slot_id")["profit_loss"]; a=g.agg(["mean","std","count"]).reset_index()
    a.columns=["slot_id","slot_mean","slot_std","slot_n"]; a=a[a["slot_n"]>=2].copy()
    a["slot_std"]=a["slot_std"].clip(lower=1e-6); a["slot_se"]=a["slot_std"]/np.sqrt(a["slot_n"])
    return a

def sortino_ratio(pnl):
    neg=pnl[pnl<0]
    if len(neg)<2: return np.nan
    ds=np.std(neg,ddof=1); return float(np.mean(pnl)/ds) if ds>0 else np.nan

def fit_null_fold(shuf_df, sidx_df, fold_i, rng):
    agg=slot_summaries(shuf_df)
    if len(agg)<10: return pd.DataFrame()
    m=agg.merge(sidx_df[["slot_id","symbol","day_of_week","bucket_idx"]],on="slot_id")
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
    with pm.Model():
        a_=pm.Normal("alpha",mu=0,sigma=100)
        ts=pm.HalfCauchy("tau_sym",beta=t0);  ls=pm.HalfCauchy("lam_sym",beta=1,shape=len(syms)); zs=pm.Normal("z_sym",mu=0,sigma=1,shape=len(syms))
        td=pm.HalfCauchy("tau_day",beta=t0);  ld=pm.HalfCauchy("lam_day",beta=1,shape=len(days)); zd=pm.Normal("z_day",mu=0,sigma=1,shape=len(days))
        tb=pm.HalfCauchy("tau_bkt",beta=t0);  lb=pm.HalfCauchy("lam_bkt",beta=1,shape=len(bkts)); zb=pm.Normal("z_bkt",mu=0,sigma=1,shape=len(bkts))
        tsb=pm.HalfCauchy("tau_sb",beta=t0);  lsb=pm.HalfCauchy("lam_sb",beta=1,shape=len(sbp)); zsb=pm.Normal("z_sb",mu=0,sigma=1,shape=len(sbp))
        tsg=pm.HalfNormal("tau_sig",sigma=1); zsg=pm.Normal("z_sig",mu=0,sigma=1,shape=N); ls0=pm.Normal("log_sig0",mu=np.log(200),sigma=1)
        mu=a_+zs[si]*ts*ls[si]+zd[di]*td*ld[di]+zb[bi]*tb*lb[bi]+zsb[sbi]*tsb*lsb[sbi]
        _=pm.Normal("obs",mu=mu,sigma=ys,observed=yo)
        tr=pm.sample(draws=DRAWS,tune=TUNE,chains=CHAINS,target_accept=0.9,
                     return_inferencedata=True,progressbar=False,compute_convergence_checks=False)
    post=tr.posterior
    ap=post["alpha"].values.ravel()
    zs_p=post["z_sym"].values.reshape(-1,len(syms)); ts_p=post["tau_sym"].values.ravel(); ls_p=post["lam_sym"].values.reshape(-1,len(syms))
    zd_p=post["z_day"].values.reshape(-1,len(days)); td_p=post["tau_day"].values.ravel(); ld_p=post["lam_day"].values.reshape(-1,len(days))
    zb_p=post["z_bkt"].values.reshape(-1,len(bkts)); tb_p=post["tau_bkt"].values.ravel(); lb_p=post["lam_bkt"].values.reshape(-1,len(bkts))
    zsb_p=post["z_sb"].values.reshape(-1,len(sbp));  tsb_p=post["tau_sb"].values.ravel(); lsb_p=post["lam_sb"].values.reshape(-1,len(sbp))
    ls0_p=post["log_sig0"].values.ravel(); tsg_p=post["tau_sig"].values.ravel()
    zsg_p=post["z_sig"].values.reshape(-1,N)
    results=[]
    for i in range(N):
        mu_s=(ap+zs_p[:,si[i]]*ts_p*ls_p[:,si[i]]+zd_p[:,di[i]]*td_p*ld_p[:,di[i]]
                 +zb_p[:,bi[i]]*tb_p*lb_p[:,bi[i]]+zsb_p[:,sbi[i]]*tsb_p*lsb_p[:,sbi[i]])
        sig_s=np.exp(ls0_p+zsg_p[:,i]*tsg_p)
        sr_d=[]
        for _ in range(200):
            idx=rng.integers(0,len(mu_s))
            sim=sps.t.rvs(df=5,loc=mu_s[idx],scale=sig_s[idx],size=50,random_state=rng)
            neg=sim[sim<0]
            if len(neg)>=2 and np.std(neg,ddof=1)>0: sr_d.append(float(np.mean(sim)/np.std(neg,ddof=1)))
        sr_lo=float(np.percentile(sr_d,5)) if len(sr_d)>10 else np.nan
        results.append({"slot_id":m.iloc[i]["slot_id"],"sortino_lo90":sr_lo})
    return pd.DataFrame(results)

# Fold date ranges loaded above; used inline below
null_counts=[]
for run in range(N_RUNS):
    print(f"\n  Null run {run+1}/{N_RUNS} (seed={run}) ...")
    rng=np.random.default_rng(seed=run); fold_passes={}
    for fold_i in range(N_FOLDS):
        fd=fold_dates[fold_dates["fold_i"]==fold_i]
        tr_s,tr_e=fd[fd["split"]=="train"][["start","end"]].values[0]
        fdf=df[(df["trade_date"]>=tr_s)&(df["trade_date"]<=tr_e)].copy()
        fdf["profit_loss"]=rng.permutation(fdf["profit_loss"].values)
        res=fit_null_fold(fdf,slot_index,fold_i,rng)
        if len(res)==0: continue
        res["passes"]=res["sortino_lo90"].fillna(-99)>LB_GATE
        for _,r in res.iterrows():
            sid=r["slot_id"]; fold_passes.setdefault(sid,0)
            if r["passes"]: fold_passes[sid]+=1
    nc=sum(1 for v in fold_passes.values() if v>=MIN_FOLDS)
    null_counts.append(nc); print(f"    Null candidates: {nc}")

null_arr=np.array(null_counts); null_95=float(np.percentile(null_arr,95))
margin=n_real-null_95
print("\n"+"="*70); print("  NULL TEST RESULTS"); print("="*70)
print(f"  Real candidates: {n_real}")
print(f"  Null mean: {null_arr.mean():.1f}  std: {null_arr.std():.1f}  95th pct: {null_95:.1f}")
if n_real==0: verdict="No real candidates."
elif n_real<=null_95: verdict=f"FAILED - real ({n_real}) within null 95th ({null_95:.1f}). Not distinguishable from noise. Do NOT proceed to Step 7."
else: verdict=f"PASS - real ({n_real}) above null 95th ({null_95:.1f}) by {margin:.1f} slots."
print(f"  VERDICT: {verdict}")
pd.DataFrame({"run":range(N_RUNS),"n_null_candidates":null_counts}).to_parquet(f"{RESULTS_DIR}/null_test_summary.parquet",index=False)
with open(f"{RESULTS_DIR}/null_test_report.txt","w") as f:
    f.write(f"N_RUNS:{N_RUNS}\nReal:{n_real}\nNull_mean:{null_arr.mean():.2f}\nNull_95th:{null_95:.2f}\nMargin:{margin:.2f}\nVERDICT:{verdict}\n")
print("\nStep 5 COMPLETE. Run: python 05_feature_inventory.py")
