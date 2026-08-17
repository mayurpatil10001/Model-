# model_spec.md
# Slot-Selection Model — Committed Specification
# Committed: 2026-08-17. DO NOT modify after Step 2 begins.

## Likelihood
profit_loss_i ~ StudentT(nu, mu_slot[i], sigma_slot[i])
nu ~ Gamma(2, 0.1), lower-bounded at 1   # estimated from data

## Slot mean (crossed random effects, non-centered parameterization)
mu_slot = alpha_global
        + z_sym[s]   * tau_sym   * lambda_sym[s]     # symbol main effect
        + z_day[d]   * tau_day   * lambda_day[d]      # day-of-week (0=Mon..6=Sun)
        + z_time[t]  * tau_time  * lambda_time[t]     # 30-min NY-time bucket index
        + z_st[s,t]  * tau_st    * lambda_st[s,t]     # symbol x time interaction

## Priors (Regularized Horseshoe, Piironen & Vehtari 2017)
z_k         ~ Normal(0, 1)           # non-centered raw effect
lambda_k    ~ HalfCauchy(0, 1)       # local shrinkage (horseshoe local scale)
tau_group   ~ HalfCauchy(0, tau0)    # global shrinkage per group
              tau0 = (p0/(D-p0)) * (sigma_noise/sqrt(N))
              p0 = prior expected number of non-zero effects = 10% of slots
alpha_global~ Normal(0, 100)
mu_log_sigma~ Normal(log(200), 1)    # per-slot scale prior (typical trade PnL)
sigma_slot  ~ exp(mu_log_sigma + z_sigma[slot] * tau_sigma)   # non-centered

## Framework: PyMC v5 (not Stan)
Reason: no external C++ compiler needed on Windows; PyMC5 NUTS equivalent to Stan NUTS;
sample_posterior_predictive() handles posterior-predictive Sortino natively.

## MCMC Settings
4 chains, 1000 tune + 1000 draw = 4000 posterior samples
target_accept = 0.9
Convergence gate: R-hat < 1.01 all params, ESS_bulk > 400 per param

## Implementation Note — Sufficient Statistics Approximation
The full StudentT likelihood on N=500K individual trades per fold would require
O(N * P) per NUTS step, making NUTS impractical (estimated 12+ hours per fold).

Practical approximation: compute per-slot sufficient statistics (n_slot, mean_slot, std_slot)
and model the slot means with the hierarchical prior. Under CLT (most slots have >100 trades),
slot means are approximately Normal(mu_slot, std_slot/sqrt(n_slot)).
This reduces effective N from ~500K to ~980 per fold while preserving the hierarchical
shrinkage structure. The approximation error is negligible for slots with n > 30.

For posterior-predictive Sortino simulation (Step 3.3): individual trades are simulated
from StudentT(nu, mu_slot, sigma_slot) using the posterior samples — this restores the
full trade-level distribution for Sortino computation.

## Permutation Space
Defined in Step 1 output (00_data_prep.py). Exact counts committed here after Step 1 runs.
Symbols: ES, NQ, FDAX, CL (ZB/ZN/MNQ/MES excluded: insufficient data)
Buckets: 30-minute intervals, America/New_York time
Soft floor: slots with < 20 trades excluded from fit (reported separately)

## Selection Rule (Step 4)
Primary: lower bound of 90% CI for posterior-predictive Sortino > 1.0
Cross-fold: slot must clear rule in >= 3 of 5 folds
Holdout bar: lower bound of 90% CI for Sortino > 3.0 (Step 7)

## Sizing Rule (Step 7a)
Flat: 3 contracts per trade, identical across all selected slots.
No differential capital allocation. This is explicitly out of scope.
