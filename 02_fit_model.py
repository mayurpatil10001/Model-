"""
REWRITE: 02_fit_model.py — Normal Hierarchical Model
Replaces Horseshoe (pathological funnel geometry with numpyro/cxx=) with
a 4-level Normal hierarchical model that converges reliably.
"""
import os as _os
_os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import warnings, os
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, pymc as pm, arviz as az

DATA_DIR    = "C:/Model-/data"
RESULTS_DIR = "C:/Model-/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CHAINS=4; TUNE=1000; DRAWS=1000; TARGET_ACCEPT=0.9
PP_DRAWS=2000; RHAT_GATE=1.05; ESS_GATE=200
N_FOLDS = 5

def sortino_ratio(pnl):
    pnl = np.asarray(pnl, dtype=float)
    neg = pnl[pnl < 0]
    if len(neg) < 2: return np.nan
    ds = np.std(neg, ddof=1)
    return float(np.mean(pnl) / ds) if ds > 0 else np.nan

def slot_summaries(df):
    g = df.groupby("slot_id")["profit_loss"]
    agg = g.agg(["mean","std","count"]).reset_index()
    agg.columns = ["slot_id","slot_mean","slot_std","slot_n"]
    agg = agg[agg["slot_n"] >= 2].copy()
    agg["slot_std"] = agg["slot_std"].clip(lower=1e-6)
    agg["slot_se"]  = agg["slot_std"] / np.sqrt(agg["slot_n"])
    return agg

def build_and_sample(agg, slot_index):
    m    = agg.merge(slot_index[["slot_id","symbol","day_of_week","bucket_idx"]], on="slot_id")
    syms = sorted(m["symbol"].unique())
    days = sorted(m["day_of_week"].unique())
    bkts = sorted(m["bucket_idx"].unique())
    sbp  = sorted(set(zip(m["symbol"], m["bucket_idx"])))

    se  = {s:i for i,s in enumerate(syms)}
    de  = {d:i for i,d in enumerate(days)}
    be  = {b:i for i,b in enumerate(bkts)}
    sbe = {p:i for i,p in enumerate(sbp)}

    si  = m["symbol"].map(se).values.astype(int)
    di  = m["day_of_week"].map(de).values.astype(int)
    bi  = m["bucket_idx"].map(be).values.astype(int)
    sbi = np.array([sbe.get((s,b),0) for s,b in zip(m["symbol"],m["bucket_idx"])], dtype=int)

    yo = m["slot_mean"].values.astype(float)
    ys = m["slot_se"].values.astype(float)

    sigma_data = float(np.std(yo)) if np.std(yo) > 0 else 1.0
    mu_data    = float(np.mean(yo))

    init = {
        "alpha":      mu_data,
        "sigma_sym":  sigma_data * 0.5,
        "sigma_day":  sigma_data * 0.3,
        "sigma_bkt":  sigma_data * 0.3,
        "sigma_slot": sigma_data * 0.2,
    }

    with pm.Model():
        alpha      = pm.Normal("alpha",      mu=mu_data,          sigma=sigma_data * 2)
        sigma_sym  = pm.HalfNormal("sigma_sym",  sigma=sigma_data)
        beta_sym   = pm.Normal("beta_sym",   mu=0, sigma=sigma_sym,  shape=len(syms))
        sigma_day  = pm.HalfNormal("sigma_day",  sigma=sigma_data * 0.5)
        beta_day   = pm.Normal("beta_day",   mu=0, sigma=sigma_day,  shape=len(days))
        sigma_bkt  = pm.HalfNormal("sigma_bkt",  sigma=sigma_data * 0.5)
        beta_bkt   = pm.Normal("beta_bkt",   mu=0, sigma=sigma_bkt,  shape=len(bkts))
        sigma_slot = pm.HalfNormal("sigma_slot", sigma=sigma_data * 0.3)
        beta_slot  = pm.Normal("beta_slot",  mu=0, sigma=sigma_slot, shape=len(sbp))

        mu_slot = (alpha + beta_sym[si] + beta_day[di]
                   + beta_bkt[bi] + beta_slot[sbi])
        _obs = pm.Normal("obs", mu=mu_slot, sigma=ys, observed=yo)

        trace = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            nuts_sampler='numpyro',
            return_inferencedata=True,
            progressbar=True,
            initvals=init,
        )

    return trace, m, si, di, bi, sbi, syms, days, bkts, sbp

# ── main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 3 - Hierarchical Model Fitting (5 folds) [Normal Hierarchical]")
print("=" * 70)

df         = pd.read_parquet(f"{DATA_DIR}/fold_assignments.parquet")
fold_dates = pd.read_parquet(f"{DATA_DIR}/fold_date_ranges.parquet")
slot_index = pd.read_parquet(f"{DATA_DIR}/slot_index.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df[df["in_model"] & ~df["is_holdout"]].copy()

for fold_i in range(N_FOLDS):
    print(f"\n{'='*70}\n  FOLD {fold_i+1}/5\n{'='*70}")
    _post_path = f"{RESULTS_DIR}/fold{fold_i}_posteriors.parquet"
    if os.path.exists(_post_path):
        print(f"  SKIPPING fold {fold_i+1} — posteriors already saved.")
        continue

    fd = fold_dates[fold_dates["fold_i"] == fold_i]
    tr_s, tr_e = fd[fd["split"]=="train"][["start","end"]].values[0]
    te_s, te_e = fd[fd["split"]=="test"][["start","end"]].values[0]
    tr_df = df[(df["trade_date"] >= tr_s) & (df["trade_date"] <= tr_e)]
    te_df = df[(df["trade_date"] >= te_s) & (df["trade_date"] <= te_e)]
    print(f"  Train: {len(tr_df):,}  Test: {len(te_df):,}")

    agg = slot_summaries(tr_df)
    print(f"  Slots with >=2 trades in train: {len(agg):,}")

    trace, m, si, di, bi, sbi, syms, days, bkts, sbp = build_and_sample(agg, slot_index)

    key_vars = ["alpha","sigma_sym","sigma_day","sigma_bkt","sigma_slot"]
    summ    = az.summary(trace, var_names=key_vars, round_to=4)
    rh_max  = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    conv_ok = (rh_max < RHAT_GATE) and (ess_min >= ESS_GATE)
    print(f"  Convergence: R-hat_max={rh_max:.4f} ({'PASS' if rh_max<RHAT_GATE else 'FAIL'}),  "
          f"ESS_min={ess_min:.0f} ({'PASS' if ess_min>=ESS_GATE else 'FAIL'})")
    if not conv_ok:
        print(f"  WARNING: convergence gate FAILED for fold {fold_i+1}.")
    print(summ[["mean","sd","ess_bulk","r_hat"]].to_string())

    trace.to_netcdf(f"{RESULTS_DIR}/fold{fold_i}_trace.nc")
    with open(f"{RESULTS_DIR}/fold{fold_i}_diagnostics.txt", "w") as f:
        f.write(f"Fold {fold_i+1}\nTrain: {len(tr_df):,}  Slots: {len(agg):,}\n")
        f.write(f"R-hat max: {rh_max:.4f}  gate<{RHAT_GATE} -> {'PASS' if rh_max<RHAT_GATE else 'FAIL'}\n")
        f.write(f"ESS min:   {ess_min:.0f}   gate>{ESS_GATE}  -> {'PASS' if ess_min>=ESS_GATE else 'FAIL'}\n")
        f.write(f"Convergence: {'PASS' if conv_ok else 'FAIL'}\n\n")
        f.write(summ[["mean","sd","hdi_3%","hdi_97%","mcse_mean","mcse_sd","ess_bulk","ess_tail","r_hat"]].to_string())

    # Posterior-predictive Sortino per slot
    print(f"  Computing posterior-predictive Sortino ({PP_DRAWS} draws/slot) ...")
    alpha_s    = trace.posterior["alpha"].values.reshape(-1)
    beta_sym_s = trace.posterior["beta_sym"].values.reshape(-1, len(syms))
    beta_day_s = trace.posterior["beta_day"].values.reshape(-1, len(days))
    beta_bkt_s = trace.posterior["beta_bkt"].values.reshape(-1, len(bkts))
    beta_slot_s= trace.posterior["beta_slot"].values.reshape(-1, len(sbp))
    S   = alpha_s.shape[0]
    idx = np.random.choice(S, size=PP_DRAWS, replace=(S < PP_DRAWS))
    mu_draws = (alpha_s[idx, None]
                + beta_sym_s[idx][:, si]
                + beta_day_s[idx][:, di]
                + beta_bkt_s[idx][:, bi]
                + beta_slot_s[idx][:, sbi])  # (PP_DRAWS, N_slots)

    rows = []
    for j in range(len(agg)):
        sid   = agg.iloc[j]["slot_id"]
        mu_j  = mu_draws[:, j]
        pnl_lo= float(np.percentile(mu_j, 10))
        neg_mu= mu_j[mu_j < 0]
        ds_mu = float(np.std(neg_mu, ddof=1)) if len(neg_mu) > 1 else 1.0
        sortino_lo90 = float(pnl_lo / ds_mu) if ds_mu > 0 else np.nan
        rows.append({
            "slot_id":     sid,
            "fold_i":      fold_i,
            "mu_post":     float(np.mean(mu_j)),
            "mu_lo90":     pnl_lo,
            "sortino_lo90": sortino_lo90,
            "slot_n":      int(agg.iloc[j]["slot_n"]),
            "conv_ok":     conv_ok,
        })

    post_df = pd.DataFrame(rows)
    post_df.to_parquet(_post_path, index=False)
    n1 = (post_df["sortino_lo90"] > 1.0).sum()
    print(f"  Fold {fold_i+1}: sortino_lo90>1.0: {n1}")

print("\nStep 3 COMPLETE. Run: python 03_select_candidates.py")
