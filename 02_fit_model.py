
"""
02_fit_model.py  Step 3
Fit Bayesian hierarchical model per fold. Check MCMC convergence.
Compute posterior-predictive Sortino per slot.

APPROXIMATION NOTE (see model_spec.md):
  Full StudentT on 500K individual trades per fold is impractical with NUTS.
  We use slot-level sufficient statistics (n, mean, std) and model slot means
  via Normal CLT approximation. Reduces effective N from ~500K to ~980 per fold.
  Posterior-predictive Sortino restored by simulating trades from StudentT(nu=5,
  mu_slot, sigma_slot) using posterior samples.
"""
import pandas as pd, numpy as np, pymc as pm, arviz as az
import scipy.stats as sps
import warnings, os
warnings.filterwarnings("ignore")

DATA_DIR    = "C:/Model-/data"
RESULTS_DIR = "C:/Model-/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CHAINS=4; TUNE=1000; DRAWS=1000; TARGET_ACCEPT=0.9
PP_DRAWS=2000; RHAT_GATE=1.01; ESS_GATE=400; P0_FRAC=0.10

def sortino_ratio(pnl):
    pnl=np.asarray(pnl,dtype=float); neg=pnl[pnl<0]
    if len(neg)<2: return np.nan
    ds=np.std(neg,ddof=1)
    return float(np.mean(pnl)/ds) if ds>0 else np.nan

def slot_summaries(df):
    g=df[df["in_model"]].groupby("slot_id")["profit_loss"]
    a=g.agg(["mean","std","count"]).reset_index()
    a.columns=["slot_id","slot_mean","slot_std","slot_n"]
    a=a[a["slot_n"]>=2].copy()
    a["slot_std"]=a["slot_std"].clip(lower=1e-6)
    a["slot_se"]=a["slot_std"]/np.sqrt(a["slot_n"])
    return a

def build_and_sample(agg, sidx_df):
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
    with pm.Model() as model:
        a_=pm.Normal("alpha",mu=0,sigma=100)
        ts=pm.HalfCauchy("tau_sym",beta=t0);  ls=pm.HalfCauchy("lam_sym",beta=1,shape=len(syms)); zs=pm.Normal("z_sym",mu=0,sigma=1,shape=len(syms))
        td=pm.HalfCauchy("tau_day",beta=t0);  ld=pm.HalfCauchy("lam_day",beta=1,shape=len(days)); zd=pm.Normal("z_day",mu=0,sigma=1,shape=len(days))
        tb=pm.HalfCauchy("tau_bkt",beta=t0);  lb=pm.HalfCauchy("lam_bkt",beta=1,shape=len(bkts)); zb=pm.Normal("z_bkt",mu=0,sigma=1,shape=len(bkts))
        tsb=pm.HalfCauchy("tau_sb",beta=t0);  lsb=pm.HalfCauchy("lam_sb",beta=1,shape=len(sbp)); zsb=pm.Normal("z_sb",mu=0,sigma=1,shape=len(sbp))
        tsg=pm.HalfNormal("tau_sig",sigma=1); zsg=pm.Normal("z_sig",mu=0,sigma=1,shape=N); ls0=pm.Normal("log_sig0",mu=np.log(200),sigma=1)
        mu=a_+zs[si]*ts*ls[si]+zd[di]*td*ld[di]+zb[bi]*tb*lb[bi]+zsb[sbi]*tsb*lsb[sbi]
        _=pm.Normal("obs",mu=mu,sigma=ys,observed=yo)
        trace=pm.sample(draws=DRAWS,tune=TUNE,chains=CHAINS,target_accept=TARGET_ACCEPT,
                        return_inferencedata=True,progressbar=True)
    return trace,m,si,di,bi,sbi,syms,days,bkts,sbp

def pp_sortino(trace,m,si,di,bi,sbi,syms,days,bkts,sbp,n_pp=PP_DRAWS):
    post=trace.posterior
    ap=post["alpha"].values.ravel()
    zs_p=post["z_sym"].values.reshape(-1,len(syms));   ts_p=post["tau_sym"].values.ravel();  ls_p=post["lam_sym"].values.reshape(-1,len(syms))
    zd_p=post["z_day"].values.reshape(-1,len(days));   td_p=post["tau_day"].values.ravel();  ld_p=post["lam_day"].values.reshape(-1,len(days))
    zb_p=post["z_bkt"].values.reshape(-1,len(bkts));   tb_p=post["tau_bkt"].values.ravel();  lb_p=post["lam_bkt"].values.reshape(-1,len(bkts))
    zsb_p=post["z_sb"].values.reshape(-1,len(sbp));    tsb_p=post["tau_sb"].values.ravel();  lsb_p=post["lam_sb"].values.reshape(-1,len(sbp))
    ls0_p=post["log_sig0"].values.ravel(); tsg_p=post["tau_sig"].values.ravel()
    zsg_p=post["z_sig"].values.reshape(-1,len(m))
    out=[]; rng=np.random.default_rng(42)
    for i in range(len(m)):
        mu_s=(ap+zs_p[:,si[i]]*ts_p*ls_p[:,si[i]]+zd_p[:,di[i]]*td_p*ld_p[:,di[i]]
                 +zb_p[:,bi[i]]*tb_p*lb_p[:,bi[i]]+zsb_p[:,sbi[i]]*tsb_p*lsb_p[:,sbi[i]])
        sig_s=np.exp(ls0_p+zsg_p[:,i]*tsg_p)
        sr_d=[]; n_act=max(int(m.iloc[i]["slot_n"]),50)
        for _ in range(n_pp):
            idx=rng.integers(0,len(mu_s))
            sim=sps.t.rvs(df=5,loc=mu_s[idx],scale=sig_s[idx],size=n_act,random_state=rng)
            sr=sortino_ratio(sim)
            if np.isfinite(sr): sr_d.append(sr)
        out.append({"slot_id":m.iloc[i]["slot_id"],"n_trades":int(m.iloc[i]["slot_n"]),
                    "slot_mean_pnl":float(m.iloc[i]["slot_mean"]),
                    "sortino_mean":float(np.mean(sr_d)) if sr_d else np.nan,
                    "sortino_lo90":float(np.percentile(sr_d,5)) if sr_d else np.nan,
                    "sortino_hi90":float(np.percentile(sr_d,95)) if sr_d else np.nan,
                    "p_mu_pos":float(np.mean(mu_s>0)),
                    "mu_post_mean":float(np.mean(mu_s)),
                    "mu_post_lo90":float(np.percentile(mu_s,5)),
                    "mu_post_hi90":float(np.percentile(mu_s,95))})
    return pd.DataFrame(out)

df=pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
fold_dates=pd.read_parquet(f"{DATA_DIR}/fold_date_ranges.parquet")
slot_index=pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")
df["trade_date"]=pd.to_datetime(df["trade_date"])
df=df[df["in_model"] & ~df["is_holdout"]].copy()

print("="*70); print("  STEP 3 - Hierarchical Model Fitting (5 folds)"); print("="*70)

for fold_i in range(5):
    print(f"\n{'='*70}\n  FOLD {fold_i+1}/5\n{'='*70}")
    fd=fold_dates[fold_dates["fold_i"]==fold_i]
    tr_s,tr_e=fd[fd["split"]=="train"][["start","end"]].values[0]
    te_s,te_e=fd[fd["split"]=="test"][["start","end"]].values[0]
    tr_df=df[(df["trade_date"]>=tr_s)&(df["trade_date"]<=tr_e)]
    te_df=df[(df["trade_date"]>=te_s)&(df["trade_date"]<=te_e)]
    print(f"  Train: {len(tr_df):,}  Test: {len(te_df):,}")
    agg=slot_summaries(tr_df)
    print(f"  Slots with >=2 trades in train: {len(agg):,}")
    trace,m,si,di,bi,sbi,syms,days,bkts,sbp=build_and_sample(agg,slot_index)
    chk=["alpha","tau_sym","tau_day","tau_bkt","tau_sb","log_sig0","tau_sig"]
    summ=az.summary(trace,var_names=chk)
    rh=float(summ["r_hat"].max()); ess=float(summ["ess_bulk"].min())
    conv_ok=(rh<RHAT_GATE)and(ess>ESS_GATE)
    print(f"  Convergence: R-hat_max={rh:.4f} ({'PASS' if rh<RHAT_GATE else 'FAIL'}),  ESS_min={ess:.0f} ({'PASS' if ess>ESS_GATE else 'FAIL'})")
    if not conv_ok: print(f"  WARNING: convergence gate FAILED for fold {fold_i+1}.")
    with open(f"{RESULTS_DIR}/fold{fold_i}_diagnostics.txt","w") as f:
        f.write(f"Fold {fold_i+1}\nTrain: {len(tr_df):,}  Slots: {len(agg):,}\n")
        f.write(f"R-hat max: {rh:.4f}  gate<{RHAT_GATE} -> {'PASS' if rh<RHAT_GATE else 'FAIL'}\n")
        f.write(f"ESS min:   {ess:.0f}   gate>{ESS_GATE}  -> {'PASS' if ess>ESS_GATE else 'FAIL'}\n")
        f.write(f"Convergence: {'PASS' if conv_ok else 'FAIL'}\n\n"); f.write(summ.to_string())
    trace.to_netcdf(f"{RESULTS_DIR}/fold{fold_i}_trace.nc")
    print(f"  Computing posterior-predictive Sortino ({PP_DRAWS} draws/slot) ...")
    pp=pp_sortino(trace,m,si,di,bi,sbi,syms,days,bkts,sbp)
    pp["fold"]=fold_i; pp["conv_ok"]=conv_ok; pp["rhat_max"]=rh; pp["ess_min"]=ess
    te_agg=slot_summaries(te_df).rename(columns={"slot_mean":"test_mean_pnl","slot_std":"test_std_pnl","slot_n":"test_n_trades"})
    pp=pp.merge(te_agg[["slot_id","test_mean_pnl","test_std_pnl","test_n_trades"]],on="slot_id",how="left")
    pp.to_parquet(f"{RESULTS_DIR}/fold{fold_i}_posteriors.parquet",index=False)
    n1=int((pp["sortino_lo90"].fillna(-99)>1.0).sum())
    print(f"  Fold {fold_i+1}: sortino_lo90>1.0: {n1}")

print("\nStep 3 COMPLETE. Run: python 03_select_candidates.py")
