#!/usr/bin/env python3
"""
reproduce.py -- single-file reproduction of every number, table body and figure in

    "Normalization as Implicit Reweighting: A Pre-Ranking Audit for
     Multi-Criteria Decision Support"  (manuscript + supplementary)

Usage
-----
    python3 reproduce.py            # everything: self-tests, results, tables, figures
    python3 reproduce.py --no-figs  # skip matplotlib

Outputs (all in the same folder as this script)
------------------------------------------------
    results/*.csv      every reported number
    manuscript.tex     every generated table body is spliced in place between
    supplementary.tex  "%% BEGIN table:<name>" and "%% END table:<name>" markers
    fig-*.pdf          all figures, one accent color, grayscale-safe

Seeds
-----
    RNG_SEED       = 42   baseline Monte Carlo (AHP draws, then CPT draws per scenario)
    RNG_SEED + 1   = 43   SMAA uniform weight space
    RNG_SEED + 2   = 44   Beta-concentration sweep
    RNG_SEED + 3   = 45   simulated decision problems (generalization study)
    RNG_SEED + 4   = 46   bootstrap resampling of bin means
    RNG_SEED + 5   = 47   proposition self-tests

The order of random draws is fixed; changing it changes the fourth decimal of
some Monte Carlo quantities but no conclusion.

Requirements: numpy, pandas, matplotlib (figures only). Runs in about 15 s.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

# ===========================================================================
# 0. Configuration
# ===========================================================================
RNG_SEED = 42
N_ITER = 5000
V_VIKOR = 0.5
EPS = 1e-12
N_RANDOM = 4000
N_BOOT = 1000

# Everything is resolved relative to the folder that contains this script.
# manuscript.tex, supplementary.tex and the fig-*.pdf files live beside it;
# the CSV results go into a results/ subfolder of the same directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
RES = os.path.join(ROOT, "results")
FIG = ROOT
os.makedirs(RES, exist_ok=True)
TABLE_BODIES = {}   # name -> LaTeX rows, spliced into paper/*.tex by inline_tables()

COMPONENTS = ["Pump", "Turbine", "Sensor", "Valve"]
CRITERION_NAMES = ["Likelihood", "Severity", "Detectability cost"]

# Bayesian-network root priors, (Ok, NotOk)
ROOT_PRIORS = {
    "Power": (0.90, 0.10),
    "Temp": (0.80, 0.20),
    "Maint": (0.70, 0.30),
    "Vib": (0.85, 0.15),
}

# Leaf definitions: parents in CPT enumeration order (last varies fastest),
# lambda = per-demand base rate (HPCI 1998-2022), mult = elicited stress
# multipliers relative to the all-nominal configuration.
LEAF_DEFS = {
    "Pump":    {"parents": ["Temp", "Maint", "Vib"], "lambda": 7.09e-3,
                "mult": [1.0, 3.0, 4.0, 8.0, 22.0, 40.0, 55.0, 90.0]},
    "Turbine": {"parents": ["Temp", "Maint", "Vib"], "lambda": 7.09e-3,
                "mult": [1.0, 5.0, 3.5, 9.0, 18.0, 48.0, 42.0, 95.0]},
    "Sensor":  {"parents": ["Power", "Temp", "Maint"], "lambda": 5.00e-4,
                "mult": [1.0, 2.5, 3.0, 6.0, 8.0, 14.0, 18.0, 28.0]},
    "Valve":   {"parents": ["Maint"], "lambda": 8.45e-4,
                "mult": [1.0, 12.0]},
}

# SME assessments; criterion 3 is detectability in cost form, 1 - D.
CRITERIA = {
    "Pump":    {"severity": 0.95, "detect": 0.50},
    "Turbine": {"severity": 0.90, "detect": 0.35},
    "Sensor":  {"severity": 0.20, "detect": 0.90},
    "Valve":   {"severity": 0.70, "detect": 0.70},
}

SCENARIOS = {
    "Nominal": {},
    "Vibration": {"Vib": "NotOk"},
    "High Temp + Maintenance": {"Temp": "NotOk", "Maint": "NotOk"},
}
SCEN_SHORT = {"Nominal": "Nominal", "Vibration": "Vibration",
              "High Temp + Maintenance": "High T + M"}


# ===========================================================================
# 1. Bayesian network: calibration and exact inference
# ===========================================================================
def calibrate_cpt(leaf):
    """Two-stage CPT. Stage 1: lambda * mult. Stage 2: rescale by
    c_i = 1 / sum_j mult_ij P(pa_j) so the marginal recovers lambda exactly."""
    d = LEAF_DEFS[leaf]
    parents = d["parents"]
    configs = list(itertools.product([0, 1], repeat=len(parents)))
    mult = np.asarray(d["mult"], dtype=float)
    p_config = np.array([np.prod([ROOT_PRIORS[p][s] for p, s in zip(parents, cfg)])
                         for cfg in configs])
    c = 1.0 / float(np.sum(mult * p_config))
    mu = np.clip(c * d["lambda"] * mult, 1e-6, 1 - 1e-6)
    return configs, mu, c


CALIBRATED = {leaf: calibrate_cpt(leaf) for leaf in LEAF_DEFS}


def calibration_check():
    rows = []
    for leaf, d in LEAF_DEFS.items():
        configs, mu, c = CALIBRATED[leaf]
        p_config = np.array([np.prod([ROOT_PRIORS[p][s] for p, s in zip(d["parents"], cfg)])
                             for cfg in configs])
        implied = float(np.sum(mu * p_config))
        rows.append({"component": leaf, "lambda": d["lambda"], "c_calib": c,
                     "implied_marginal": implied, "ratio": implied / d["lambda"]})
    return pd.DataFrame(rows)


def scenario_weights(leaf, scenario):
    """P(pa_j | evidence) for every parent configuration of `leaf`."""
    d = LEAF_DEFS[leaf]
    configs, _, _ = CALIBRATED[leaf]
    evid = SCENARIOS[scenario]
    w = np.ones(len(configs))
    for idx, cfg in enumerate(configs):
        prob = 1.0
        for parent, state in zip(d["parents"], cfg):
            if parent in evid:
                fixed = 0 if evid[parent] == "Ok" else 1
                prob *= 1.0 if state == fixed else 0.0
            else:
                prob *= ROOT_PRIORS[parent][state]
        w[idx] = prob
    return w


def sample_posteriors(rng, scenario, s, n=N_ITER):
    """Beta(mu s, (1-mu) s) draws on every CPT entry, marginalized."""
    out = {}
    for leaf in LEAF_DEFS:
        _, mu, _ = CALIBRATED[leaf]
        w = scenario_weights(leaf, scenario)
        draws = rng.beta(mu * s, (1.0 - mu) * s, size=(n, len(mu)))
        out[leaf] = draws @ w
    return out


def build_decision_matrix(post):
    n = len(post["Pump"])
    X = np.zeros((n, 4, 3))
    for i, comp in enumerate(COMPONENTS):
        X[:, i, 0] = post[comp]
        X[:, i, 1] = CRITERIA[comp]["severity"]
        X[:, i, 2] = 1.0 - CRITERIA[comp]["detect"]
    return X


# ===========================================================================
# 2. Weights
# ===========================================================================
def sample_ahp(rng, n=N_ITER):
    """Stochastic AHP, triangular pairwise comparisons on Saaty's scale."""
    a12 = rng.triangular(2, 3, 4, size=n)
    a13 = rng.triangular(4, 5, 6, size=n)
    a23 = rng.triangular(2, 3, 4, size=n)
    W = np.zeros((n, 3))
    CR = np.zeros(n)
    RI = 0.58
    for k in range(n):
        A = np.array([[1.0, a12[k], a13[k]],
                      [1.0 / a12[k], 1.0, a23[k]],
                      [1.0 / a13[k], 1.0 / a23[k], 1.0]])
        ev, vec = np.linalg.eig(A)
        i = int(np.argmax(ev.real))
        w = np.abs(vec[:, i].real)
        W[k] = w / w.sum()
        CR[k] = ((ev.real[i] - 3.0) / 2.0) / RI
    return W, CR


def entropy_weights(X):
    n_alt = X.shape[1]
    colsum = X.sum(axis=1, keepdims=True)
    colsum = np.where(colsum <= 0, EPS, colsum)
    P = X / colsum
    Plog = np.where(P > 0, P * np.log(P), 0.0)
    e = -(1.0 / np.log(n_alt)) * Plog.sum(axis=1)
    d = 1.0 - e
    dsum = d.sum(axis=1, keepdims=True)
    return np.where(dsum <= EPS, np.full_like(d, 1.0 / d.shape[1]),
                    d / np.where(dsum <= EPS, 1.0, dsum))


# ===========================================================================
# 3. Normalizations and aggregation rules
# ===========================================================================
def minmax_normalize(X):
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    return (X - xmin) / (xmax - xmin + EPS)


def vector_normalize(X):
    rss = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    return X / (rss + EPS)


def additive(X, W):
    """Min-max additive value function (the SMAA-2 kernel). Reference cell."""
    return (W[:, None, :] * minmax_normalize(X)).sum(axis=2)


def additive_vector_norm(X, W):
    """Weighted sum of vector-normalized values (WSM-V). Normalization-only cell."""
    return (W[:, None, :] * vector_normalize(X)).sum(axis=2)


def vikor(X, W, v=V_VIKOR):
    D = minmax_normalize(X)
    Wb = W[:, None, :]
    S = (Wb * D).sum(axis=2)
    R = (Wb * D).max(axis=2)
    Ss, Sm = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    Rs, Rm = R.min(axis=1, keepdims=True), R.max(axis=1, keepdims=True)
    return v * (S - Ss) / (Sm - Ss + EPS) + (1 - v) * (R - Rs) / (Rm - Rs + EPS)


def topsis(X, W, normalizer="vector"):
    r = vector_normalize(X) if normalizer == "vector" else minmax_normalize(X)
    v = r * W[:, None, :]
    vp = v.max(axis=1, keepdims=True)
    vm = v.min(axis=1, keepdims=True)
    dp = np.sqrt(((v - vp) ** 2).sum(axis=2))
    dm = np.sqrt(((v - vm) ** 2).sum(axis=2))
    return dm / (dp + dm + EPS)


def promethee(X, W, theta=1.0):
    """PROMETHEE II net flow, V-shape, q = 0, p_j = theta_j * R_j."""
    n, m, c = X.shape
    R = X.max(axis=1) - X.min(axis=1)
    th = np.broadcast_to(np.asarray(theta, dtype=float), (c,))
    p = R * th[None, :] + EPS
    diff = X[:, :, None, :] - X[:, None, :, :]
    Pj = np.clip(diff / p[:, None, None, :], 0.0, 1.0)
    pi = (Pj * W[:, None, None, :]).sum(axis=3)
    return (pi.sum(axis=2) - pi.sum(axis=1)) / (m - 1)


# ===========================================================================
# 4. The audit: kappa, implied weights, nu, delta_w, inversion screen
# ===========================================================================
def kappa(X):
    """kappa_j = R_j / ||x_j||_2  (per iteration)."""
    R = X.max(axis=1) - X.min(axis=1)
    nrm = np.sqrt((X ** 2).sum(axis=1))
    return R / (nrm + EPS)


def implied_weights(X, W):
    k = kappa(X)
    raw = W * k
    return raw / (raw.sum(axis=1, keepdims=True) + EPS), k


def weight_shift(X, W):
    Wt, _ = implied_weights(X, W)
    return 0.5 * np.abs(Wt - W).sum(axis=1)


def kappa_spread(X):
    k = kappa(X)
    return k.max(axis=1) / (k.min(axis=1) + EPS)


def inversion_criterion(X, W, i, j):
    """Pairwise sign criterion. a_l = w_l (x_il - x_jl)/R_l;
    inversion iff sign(sum a_l) != sign(sum kappa_l a_l)."""
    R = X.max(axis=1) - X.min(axis=1) + EPS
    delta = X[:, i, :] - X[:, j, :]
    a = W * delta / R
    k = kappa(X)
    a_sum = a.sum(axis=1)
    ak_sum = (k * a).sum(axis=1)
    return a_sum, ak_sum, np.sign(a_sum) != np.sign(ak_sum)


def observed_inversion(sa_, sb_, i, j):
    sa = np.sign(sa_[:, i] - sa_[:, j])
    sb = np.sign(sb_[:, i] - sb_[:, j])
    return (sa != sb) & (sa != 0) & (sb != 0)


def non_dominated(X, i, j):
    d = X[:, i, :] - X[:, j, :]
    return ~(np.all(d >= 0, axis=1) | np.all(d <= 0, axis=1))


def audit(X, w, eps_R=1e-12, eps_m=0.0):
    """Algorithm 1 for one decision matrix X (n x m) and weight vector w.
    Returns a dict with kappa, implied weights, nu, delta_w, the pairwise
    warning set and the list of zero-range criteria."""
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    n, m = X.shape
    R = X.max(axis=0) - X.min(axis=0)
    zero = [j for j in range(m) if R[j] <= eps_R]
    keep = [j for j in range(m) if R[j] > eps_R]
    Xk, wk = X[:, keep], w[keep] / w[keep].sum()
    nrm = np.sqrt((Xk ** 2).sum(axis=0))
    k = R[keep] / nrm
    wt = wk * k / (wk * k).sum()
    nu = k.max() / k.min()
    dw = 0.5 * np.abs(wt - wk).sum()
    warn = []
    for i, i2 in itertools.combinations(range(n), 2):
        a = wk * (Xk[i] - Xk[i2]) / R[keep]
        s1, s2 = a.sum(), (k * a).sum()
        if abs(s1) > eps_m and abs(s2) > eps_m and np.sign(s1) != np.sign(s2):
            warn.append((i, i2))
    return {"kappa": k, "w_implied": wt, "nu": nu, "delta_w": dw,
            "warnings": warn, "zero_range": zero}


# ===========================================================================
# 5. Ranking and agreement
# ===========================================================================
def ranks_from_scores(scores):
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    n, m = scores.shape
    ranks[np.arange(n)[:, None], order] = np.tile(np.arange(1, m + 1), (n, 1))
    return ranks


def tau_rho(ra, rb):
    """Kendall tau (== tau_b without ties) and Spearman rho, vectorized."""
    m = ra.shape[1]
    pairs = list(itertools.combinations(range(m), 2))
    concord = np.zeros(ra.shape[0])
    for i, j in pairs:
        concord += np.sign(ra[:, i] - ra[:, j]) * np.sign(rb[:, i] - rb[:, j])
    tau = concord / len(pairs)
    rho = 1 - 6 * ((ra - rb) ** 2).sum(axis=1) / (m * (m ** 2 - 1))
    return tau, rho


def summarize_scores(scores, label, scenario):
    ranks = ranks_from_scores(scores)
    return pd.DataFrame([{
        "scenario": scenario, "method": label, "component": comp,
        "mean": scores[:, i].mean(),
        "ci_lo": np.percentile(scores[:, i], 2.5),
        "ci_hi": np.percentile(scores[:, i], 97.5),
        "P_R1": float(np.mean(ranks[:, i] == 1)),
        "E_rank": float(ranks[:, i].mean())} for i, comp in enumerate(COMPONENTS)])


def pairwise_agreement(rank_dict, scenario):
    rows = []
    for a, b in itertools.combinations(rank_dict.keys(), 2):
        ra, rb = rank_dict[a], rank_dict[b]
        tau, rho = tau_rho(ra, rb)
        rows.append({"scenario": scenario, "pair": f"{a}--{b}",
                     "tau_b": tau.mean(), "rho": rho.mean(),
                     "top1_agree": float(np.mean(np.argmin(ra, 1) == np.argmin(rb, 1))),
                     "identical": float(np.mean(np.all(ra == rb, axis=1)))})
    return pd.DataFrame(rows)


def localisation(rank_dict, scenario):
    rows = []
    pairs4 = list(itertools.combinations(range(4), 2))
    for a, b in itertools.combinations(rank_dict.keys(), 2):
        ra, rb = rank_dict[a], rank_dict[b]
        div = ~np.all(ra == rb, axis=1)
        row = {"scenario": scenario, "pair": f"{a}--{b}", "f_dis": float(div.mean())}
        if div.sum() > 0:
            for i, j in pairs4:
                sa = np.sign(ra[div, i] - ra[div, j])
                sb = np.sign(rb[div, i] - rb[div, j])
                row[f"{COMPONENTS[i][0]}-{COMPONENTS[j][0]}"] = float(
                    np.mean((sa != sb) & (sa != 0) & (sb != 0)))
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 6. SMAA-2
# ===========================================================================
def smaa_indices(X, W):
    U = additive(X, W)
    ranks = ranks_from_scores(U)
    n_alt = X.shape[1]
    b = np.zeros((n_alt, n_alt))
    wc = np.full((n_alt, W.shape[1]), np.nan)
    for i in range(n_alt):
        for r in range(1, n_alt + 1):
            b[i, r - 1] = np.mean(ranks[:, i] == r)
        first = ranks[:, i] == 1
        if first.sum() > 0:
            wc[i] = W[first].mean(axis=0)
    pc = np.zeros(n_alt)
    for i in range(n_alt):
        if np.isnan(wc[i]).any():
            continue
        Wfix = np.tile(wc[i], (X.shape[0], 1))
        pc[i] = np.mean(ranks_from_scores(additive(X, Wfix))[:, i] == 1)
    return b, wc, pc


def smaa_uniform(rng, X, m_draws=20):
    n, n_alt, n_crit = X.shape
    counts = np.zeros((n_alt, n_alt))
    wsum = np.zeros((n_alt, n_crit))
    wcnt = np.zeros(n_alt)
    for _ in range(m_draws):
        W = rng.dirichlet(np.ones(n_crit), size=n)
        ranks = ranks_from_scores(additive(X, W))
        for i in range(n_alt):
            for r in range(1, n_alt + 1):
                counts[i, r - 1] += np.sum(ranks[:, i] == r)
            first = ranks[:, i] == 1
            wsum[i] += W[first].sum(axis=0)
            wcnt[i] += first.sum()
    b = counts / (n * m_draws)
    wc = np.where(wcnt[:, None] > 0, wsum / np.maximum(wcnt[:, None], 1), np.nan)
    return b, wc


# ===========================================================================
# 7. Simulated decision problems (generalization study)
# ===========================================================================
def random_problem(rng, n_alt, n_crit, log_spread, offset):
    """x_ij = 10^U(-d_j, 0) + o * U(0,1) * mean_i, with d_j ~ U(0, log_spread)."""
    X = np.zeros((n_alt, n_crit))
    for j in range(n_crit):
        decades = rng.uniform(0.0, log_spread)
        vals = 10.0 ** rng.uniform(-decades, 0.0, size=n_alt)
        off = offset * rng.uniform(0.0, 1.0) * vals.mean()
        X[:, j] = vals + off
    return X


def random_validation(rng, n_problems=N_RANDOM, n_alt_choices=(4, 5, 6, 8, 10),
                      n_crit_choices=(3, 4, 5), log_spread_max=4.0, offset_max=5.0):
    recs = []
    for _ in range(n_problems):
        n_alt = int(rng.choice(n_alt_choices))
        n_crit = int(rng.choice(n_crit_choices))
        X = random_problem(rng, n_alt, n_crit, rng.uniform(0.2, log_spread_max),
                           rng.uniform(0.0, offset_max))[None, ...]
        W = rng.dirichlet(np.ones(n_crit), size=1)
        k = kappa(X)[0]
        nu = float(k.max() / (k.min() + EPS))
        dw = float(weight_shift(X, W)[0])
        r_add = ranks_from_scores(additive(X, W))
        r_wsmv = ranks_from_scores(additive_vector_norm(X, W))
        r_tv = ranks_from_scores(topsis(X, W, "vector"))
        r_tm = ranks_from_scores(topsis(X, W, "minmax"))
        t1, s1 = tau_rho(r_add, r_wsmv)
        t2, s2 = tau_rho(r_add, r_tv)
        t3, s3 = tau_rho(r_add, r_tm)
        recs.append({"n_alt": n_alt, "n_crit": n_crit,
                     "kappa_max": float(k.max()), "kappa_min": float(k.min()),
                     "nu": nu, "log_nu": np.log10(nu), "delta_w": dw,
                     "tau_wsm": float(t1[0]), "rho_wsm": float(s1[0]),
                     "identical_wsm": bool(np.all(r_add == r_wsmv)),
                     "tau_topsis": float(t2[0]), "rho_topsis": float(s2[0]),
                     "identical_topsis": bool(np.all(r_add == r_tv)),
                     "tau_topsis_minmax": float(t3[0]), "rho_topsis_minmax": float(s3[0]),
                     "identical_topsis_minmax": bool(np.all(r_add == r_tm))})
    return pd.DataFrame(recs)


# ===========================================================================
# 8. Self-tests of the propositions (run before anything is reported)
# ===========================================================================
def self_tests(rng, n_trials=2000):
    """Numerical checks of Proposition 1, Corollary 2 and Proposition 2 on
    random matrices with 3-10 alternatives and 2-6 criteria."""
    fails = {"prop1": 0, "cor2": 0, "prop2": 0}
    for _ in range(n_trials):
        n_alt = int(rng.integers(3, 11))
        n_crit = int(rng.integers(2, 7))
        X = random_problem(rng, n_alt, n_crit, rng.uniform(0.2, 4.0), rng.uniform(0, 5))[None]
        W = rng.dirichlet(np.ones(n_crit), size=1)
        Wt, _ = implied_weights(X, W)
        # Proposition 1: WSM-V ranking == additive ranking under implied weights
        if not np.array_equal(ranks_from_scores(additive_vector_norm(X, W)),
                              ranks_from_scores(additive(X, Wt))):
            fails["prop1"] += 1
        # Corollary 2: sign criterion == observed WSM-V inversion, every pair
        sa, sv = additive(X, W), additive_vector_norm(X, W)
        for i, j in itertools.combinations(range(n_alt), 2):
            _, _, pred = inversion_criterion(X, W, i, j)
            if pred[0] != observed_inversion(sa, sv, i, j)[0]:
                fails["cor2"] += 1
                break
        # Proposition 2: PROMETHEE II at theta = 1 == additive ranking
        if not np.array_equal(ranks_from_scores(promethee(X, W, 1.0)),
                              ranks_from_scores(additive(X, W))):
            fails["prop2"] += 1
    return fails


# ===========================================================================
# 9. Analyses
# ===========================================================================
def save(df, name):
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"  wrote results/{name:36s} {len(df):5d} rows")
    return df


def shapley_rows(scenario, L_n, L_a, L_na, loss_name):
    """Two-factor Shapley allocation of the joint loss L(NA)."""
    I = L_na - L_n - L_a
    phi_n = 0.5 * L_n + 0.5 * (L_na - L_a)
    phi_a = 0.5 * L_a + 0.5 * (L_na - L_n)
    return {"scenario": scenario, "loss": loss_name, "L_N": L_n, "L_A": L_a,
            "L_NA": L_na, "interaction": I, "phi_N": phi_n, "phi_A": phi_a,
            "share_N": phi_n / L_na if L_na > 0 else np.nan}


def run_all(make_figs=True):
    rng = np.random.default_rng(RNG_SEED)

    print("\n[0] Self-tests of the propositions")
    fails = self_tests(np.random.default_rng(RNG_SEED + 5))
    print(f"  {fails}")
    assert all(v == 0 for v in fails.values()), "a proposition self-test failed"

    print("\n[1] Calibration check")
    save(calibration_check(), "calibration_check.csv")

    print(f"\n[2] Baseline Monte Carlo, N = {N_ITER}, s = 10")
    W, CR = sample_ahp(rng, N_ITER)
    print(f"  AHP mean CR = {CR.mean():.4f}, P(CR < 0.1) = {np.mean(CR < 0.1):.4f}, "
          f"mean w = {np.round(W.mean(0), 4)}")
    store, posterior_rows, kappa_rows, score_rows = {}, [], [], []
    agree_rows, local_rows, inv_rows, check_rows, fact_rows, shap = [], [], [], [], [], []

    for scen in SCENARIOS:
        post = sample_posteriors(rng, scen, 10.0, N_ITER)
        X = build_decision_matrix(post)
        store[scen] = (X, W)
        for comp in COMPONENTS:
            p = post[comp]
            posterior_rows.append({"scenario": scen, "component": comp, "mean": p.mean(),
                                   "ci_lo": np.percentile(p, 2.5), "ci_hi": np.percentile(p, 97.5)})
        k = kappa(X)
        Wt, _ = implied_weights(X, W)
        dw, nu = weight_shift(X, W), kappa_spread(X)
        for j, cname in enumerate(CRITERION_NAMES):
            kappa_rows.append({"scenario": scen, "criterion": cname,
                               "kappa_mean": k[:, j].mean(),
                               "kappa_lo": np.percentile(k[:, j], 2.5),
                               "kappa_hi": np.percentile(k[:, j], 97.5),
                               "w_nominal": W[:, j].mean(), "w_implied": Wt[:, j].mean(),
                               "shift_pp": 100 * (Wt[:, j].mean() - W[:, j].mean()),
                               "delta_w_mean": dw.mean(), "nu_mean": nu.mean(),
                               "nu_lo": np.percentile(nu, 2.5), "nu_hi": np.percentile(nu, 97.5)})
        scores = {"VIKOR": vikor(X, W), "TOPSIS": topsis(X, W, "vector"),
                  "PROMETHEE": promethee(X, W, 1.0), "ADD": additive(X, W),
                  "WSM-V": additive_vector_norm(X, W), "TOPSIS-mm": topsis(X, W, "minmax")}
        for label, sc in scores.items():
            score_rows.append(summarize_scores(sc, label, scen))
        ranks = {k_: ranks_from_scores(v) for k_, v in scores.items()}
        main = {k_: ranks[k_] for k_ in ["VIKOR", "TOPSIS", "PROMETHEE", "ADD"]}
        agree_rows.append(pairwise_agreement(main, scen).assign(block="main"))
        local_rows.append(localisation(main, scen))
        cells = {"ADD": ranks["ADD"], "WSM-V": ranks["WSM-V"],
                 "TOPSIS": ranks["TOPSIS"], "TOPSIS-mm": ranks["TOPSIS-mm"]}
        agree_rows.append(pairwise_agreement(cells, scen).assign(block="factorial"))

        exact = np.all(ranks_from_scores(additive(X, Wt)) == ranks["WSM-V"], axis=1)
        check_rows.append({"scenario": scen, "prop1_exact_rate": float(exact.mean())})

        for i, j in itertools.combinations(range(4), 2):
            _, _, pred = inversion_criterion(X, W, i, j)
            obs_w = observed_inversion(scores["ADD"], scores["WSM-V"], i, j)
            obs_t = observed_inversion(scores["ADD"], scores["TOPSIS"], i, j)
            inv_rows.append({"scenario": scen, "pair": f"{COMPONENTS[i]}-{COMPONENTS[j]}",
                             "non_dominated_rate": float(non_dominated(X, i, j).mean()),
                             "predicted_rate": float(pred.mean()),
                             "observed_wsm_rate": float(obs_w.mean()),
                             "observed_topsis_rate": float(obs_t.mean()),
                             "accuracy_wsm": float(np.mean(pred == obs_w)),
                             "accuracy_topsis": float(np.mean(pred == obs_t)),
                             "n_pred": int(pred.sum()), "n_obs_topsis": int(obs_t.sum()),
                             "n_hit": int((pred & obs_t).sum()),
                             "recall_topsis": float(np.mean(pred[obs_t])) if obs_t.sum() else np.nan,
                             "precision_topsis": float(np.mean(obs_t[pred])) if pred.sum() else np.nan})

        # factorial cells, interaction and Shapley on both loss functions
        r_ref = ranks["ADD"]
        L_rate, L_tau = {}, {}
        for lab, key in [("N", "WSM-V"), ("A", "TOPSIS-mm"), ("NA", "TOPSIS")]:
            t, _ = tau_rho(ranks[key], r_ref)
            L_rate[lab] = float(np.mean(~np.all(ranks[key] == r_ref, axis=1)))
            L_tau[lab] = float(1 - t.mean())
            fact_rows.append({"scenario": scen, "cell": lab, "method": key,
                              "tau_b": 1 - L_tau[lab], "divergence_rate": L_rate[lab]})
        shap.append(shapley_rows(scen, L_rate["N"], L_rate["A"], L_rate["NA"], "identical_rate"))
        shap.append(shapley_rows(scen, L_tau["N"], L_tau["A"], L_tau["NA"], "kendall"))

    save(pd.DataFrame(posterior_rows), "posteriors.csv")
    save(pd.DataFrame(kappa_rows), "kappa_diagnostic.csv")
    save(pd.concat(score_rows, ignore_index=True), "method_scores.csv")
    save(pd.concat(agree_rows, ignore_index=True), "agreement.csv")
    save(pd.concat(local_rows, ignore_index=True), "localization.csv")
    inv = save(pd.DataFrame(inv_rows), "inversion_criterion.csv")
    save(pd.DataFrame(check_rows), "proposition1_check.csv")
    save(pd.DataFrame(fact_rows), "factorial_cells.csv")
    save(pd.DataFrame(shap), "factorial_shapley.csv")
    pooled = {"n_pred": int(inv.n_pred.sum()), "n_obs": int(inv.n_obs_topsis.sum()),
              "n_hit": int(inv.n_hit.sum())}
    pooled["precision"] = pooled["n_hit"] / pooled["n_pred"]
    pooled["recall"] = pooled["n_hit"] / pooled["n_obs"]
    save(pd.DataFrame([pooled]), "inversion_pooled.csv")

    print("\n[3] Sensor-Valve scatter (nominal)")
    X, Wb = store["Nominal"]
    i, j = COMPONENTS.index("Sensor"), COMPONENTS.index("Valve")
    a_sum, ak_sum, pred = inversion_criterion(X, Wb, i, j)
    obs_t = observed_inversion(additive(X, Wb), topsis(X, Wb, "vector"), i, j)
    save(pd.DataFrame({"a_sum": a_sum, "ak_sum": ak_sum, "predicted": pred,
                       "observed_topsis": obs_t}), "sensor_valve_scatter.csv")

    print("\n[4] PROMETHEE II threshold sweep")
    rows = []
    for scen, (X, Wb) in store.items():
        r_add = ranks_from_scores(additive(X, Wb))
        r_top = ranks_from_scores(topsis(X, Wb, "vector"))
        for th in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
            r_pr = ranks_from_scores(promethee(X, Wb, th))
            row = {"scenario": scen, "theta": th,
                   "tau_vs_additive": tau_rho(r_pr, r_add)[0].mean(),
                   "tau_vs_topsis": tau_rho(r_pr, r_top)[0].mean(),
                   "identical_vs_additive": float(np.mean(np.all(r_pr == r_add, axis=1)))}
            for ci, comp in enumerate(COMPONENTS):
                row["P_R1_" + comp] = float(np.mean(r_pr[:, ci] == 1))
            rows.append(row)
    save(pd.DataFrame(rows), "promethee_theta_sweep.csv")

    print("\n[5] PROMETHEE II criterion-specific thresholds (Config B)")
    rows = []
    for scen, (X, Wb) in store.items():
        r_add = ranks_from_scores(additive(X, Wb))
        for label, th in [("Config A", 1.0), ("Config B", [0.5, 1.0, 1.0])]:
            r_pr = ranks_from_scores(promethee(X, Wb, th))
            rows.append({"scenario": scen, "config": label,
                         "tau_b": tau_rho(r_pr, r_add)[0].mean(),
                         "top1_agree": float(np.mean(np.argmin(r_pr, 1) == np.argmin(r_add, 1))),
                         "identical": float(np.mean(np.all(r_pr == r_add, axis=1)))})
    save(pd.DataFrame(rows), "promethee_configB.csv")

    print("\n[6] SMAA-2 weight regimes")
    rows = []
    rng_u = np.random.default_rng(RNG_SEED + 1)
    for scen, (X, Wb) in store.items():
        b_ahp, wc_ahp, pc_ahp = smaa_indices(X, Wb)
        b_ent, wc_ent, _ = smaa_indices(X, entropy_weights(X))
        b_uni, wc_uni = smaa_uniform(rng_u, X, 20)
        for idx, comp in enumerate(COMPONENTS):
            rows.append({"scenario": scen, "component": comp,
                         "b1_ahp": b_ahp[idx, 0], "b1_entropy": b_ent[idx, 0],
                         "b1_uniform": b_uni[idx, 0], "pc_ahp": pc_ahp[idx],
                         "wc_ahp": np.round(wc_ahp[idx], 2).tolist(),
                         "wc_entropy": np.round(wc_ent[idx], 2).tolist(),
                         "wc_uniform": np.round(wc_uni[idx], 2).tolist()})
    save(pd.DataFrame(rows), "smaa_weight_regimes.csv")

    print("\n[7] Beta-concentration sweep")
    rows = []
    rng_s = np.random.default_rng(RNG_SEED + 2)
    for s in [5.0, 10.0, 20.0, 50.0]:
        Ws, _ = sample_ahp(rng_s, N_ITER)
        for scen in SCENARIOS:
            X = build_decision_matrix(sample_posteriors(rng_s, scen, s, N_ITER))
            rk = {"VIKOR": ranks_from_scores(vikor(X, Ws)),
                  "TOPSIS": ranks_from_scores(topsis(X, Ws, "vector")),
                  "PROMETHEE": ranks_from_scores(promethee(X, Ws, 1.0)),
                  "ADD": ranks_from_scores(additive(X, Ws))}
            ag = pairwise_agreement(rk, scen)
            rows.append({"s": s, "scenario": scen, "mean_tau_b": ag.tau_b.mean(),
                         "mean_identical": ag.identical.mean(),
                         "nu_mean": float(kappa_spread(X).mean()),
                         "delta_w_mean": float(weight_shift(X, Ws).mean())})
    save(pd.DataFrame(rows), "beta_sweep.csv")

    print(f"\n[8] Simulated decision problems, {N_RANDOM} problems")
    val = random_validation(np.random.default_rng(RNG_SEED + 3), N_RANDOM)
    save(val, "simulation_study.csv")

    rng_b = np.random.default_rng(RNG_SEED + 4)

    def binned(df, col, edges, labels, name):
        df = df.copy()
        df["bin"] = pd.cut(df[col], bins=edges, labels=labels)
        rows = []
        for lab, g in df.groupby("bin", observed=True):
            row = {"bin": lab, "n": len(g), f"{col}_median": g[col].median(),
                   "nu_median": g.nu.median(), "delta_w_mean": g.delta_w.mean()}
            for c in ["tau_wsm", "tau_topsis_minmax", "tau_topsis"]:
                v = g[c].to_numpy()
                row[c] = v.mean()
                if len(v) >= 2:
                    bs = np.array([rng_b.choice(v, len(v)).mean() for _ in range(N_BOOT)])
                    row[c + "_lo"], row[c + "_hi"] = np.percentile(bs, [2.5, 97.5])
                else:
                    row[c + "_lo"] = row[c + "_hi"] = np.nan
            for c in ["identical_wsm", "identical_topsis_minmax", "identical_topsis"]:
                row["disagree_" + c.split("_", 1)[1]] = 1 - g[c].mean()
            rows.append(row)
        return save(pd.DataFrame(rows), name)

    binned(val, "nu", [0.999, 1.05, 1.25, 1.75, 3.0, 6.0, np.inf],
           ["1.00-1.05", "1.05-1.25", "1.25-1.75", "1.75-3.0", "3.0-6.0", ">6.0"],
           "simulation_binned_nu.csv")
    binned(val, "delta_w", [-0.001, 0.02, 0.05, 0.10, 0.20, 0.30, 1.0],
           ["0-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", ">0.30"],
           "simulation_binned_deltaw.csv")

    print("\n[9] Scaling regressions (HC0 robust SE)")

    def ols(y, cols):
        Xd = np.column_stack([np.ones(len(val))] + [val[c].to_numpy(float) for c in cols])
        yv = np.asarray(y, float)
        beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        resid = yv - Xd @ beta
        r2 = 1 - resid.var() / yv.var()
        XtXi = np.linalg.pinv(Xd.T @ Xd)
        meat = (Xd * (resid ** 2)[:, None]).T @ Xd
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        return beta, se, r2

    fits = []
    for predictor in ["log_nu", "delta_w"]:
        for label, col in [("normalization only", "tau_wsm"),
                           ("aggregation only", "tau_topsis_minmax"),
                           ("both", "tau_topsis")]:
            beta, se, r2 = ols(val[col], [predictor, "n_alt", "n_crit"])
            sp = pd.Series(val[col]).corr(pd.Series(val[predictor]), method="spearman")
            fits.append({"predictor": predictor, "component": label, "slope": beta[1],
                         "se": se[1], "t": beta[1] / se[1], "coef_n_alt": beta[2],
                         "coef_n_crit": beta[3], "r2": r2, "spearman": sp,
                         "mean_tau": val[col].mean()})
    save(pd.DataFrame(fits), "scaling_regressions.csv")
    hi = val[val.nu > 6]
    save(pd.DataFrame([{"n": len(hi), "kappa_max_median": hi.kappa_max.median(),
                        "kappa_min_median": hi.kappa_min.median()}]), "high_nu_profile.csv")

    print("\n[10] Deterministic (posterior-mean) matrix")
    rows = []
    wbar = W.mean(axis=0)
    for scen in SCENARIOS:
        post = sample_posteriors(np.random.default_rng(RNG_SEED), scen, 10.0, N_ITER)
        Xd = np.zeros((1, 4, 3))
        for idx, comp in enumerate(COMPONENTS):
            Xd[0, idx] = [post[comp].mean(), CRITERIA[comp]["severity"], 1 - CRITERIA[comp]["detect"]]
        k = kappa(Xd)[0]
        Wt, _ = implied_weights(Xd, wbar[None, :])
        for j, cname in enumerate(CRITERION_NAMES):
            rows.append({"scenario": scen, "criterion": cname, "kappa": k[j],
                         "w_nominal": wbar[j], "w_implied": Wt[0, j],
                         "nu": k.max() / k.min(),
                         "delta_w": float(weight_shift(Xd, wbar[None, :])[0])})
    save(pd.DataFrame(rows), "kappa_deterministic.csv")

    print("\n[11] Severity compression")
    TOP = 0.90
    rows, shap2 = [], []
    Xbase, Wb = store["Nominal"]
    sev0 = np.array([CRITERIA[c]["severity"] for c in COMPONENTS])
    sbar = sev0.mean()
    for t in [1.0, 0.75, 0.5, 0.25, 0.10]:
        sev = (sbar + (1 - t) * (TOP - sbar)) + t * (sev0 - sbar)
        X = Xbase.copy()
        X[:, :, 1] = sev[None, :]
        r_add = ranks_from_scores(additive(X, Wb))
        r_n = ranks_from_scores(additive_vector_norm(X, Wb))
        r_a = ranks_from_scores(topsis(X, Wb, "minmax"))
        r_b = ranks_from_scores(topsis(X, Wb, "vector"))
        Ln = float(np.mean(~np.all(r_n == r_add, axis=1)))
        La = float(np.mean(~np.all(r_a == r_add, axis=1)))
        Lb = float(np.mean(~np.all(r_b == r_add, axis=1)))
        sr = shapley_rows(f"t={t}", Ln, La, Lb, "identical_rate")
        rows.append({"t": t, "severity_min": sev.min(), "severity_max": sev.max(),
                     "severity_range": sev.max() - sev.min(),
                     "kappa_severity": kappa(X)[:, 1].mean(),
                     "nu_mean": float(kappa_spread(X).mean()),
                     "delta_w_mean": float(weight_shift(X, Wb).mean()),
                     "div_normalization": Ln, "div_aggregation": La, "div_both": Lb,
                     "interaction": sr["interaction"], "phi_N": sr["phi_N"],
                     "phi_A": sr["phi_A"], "share_N": sr["share_N"]})
    save(pd.DataFrame(rows), "severity_compression.csv")

    print("\n[12] Worked example (3 alternatives, 2 criteria)")
    Xe = np.array([[0.020, 0.70], [0.005, 0.90], [0.001, 0.80]])
    we = np.array([0.5, 0.5])
    a = audit(Xe, we)
    u = minmax_normalize(Xe[None])[0]
    r = vector_normalize(Xe[None])[0]
    U_mm = (we * u).sum(1)
    U_v = (we * r).sum(1)
    U_imp = (a["w_implied"] * u).sum(1)
    rows = []
    for i, name in enumerate(["A", "B", "C"]):
        rows.append({"alt": name, "x1": Xe[i, 0], "x2": Xe[i, 1], "u1": u[i, 0], "u2": u[i, 1],
                     "r1": r[i, 0], "r2": r[i, 1], "U_minmax": U_mm[i], "U_vector": U_v[i],
                     "U_implied": U_imp[i]})
    ex = pd.DataFrame(rows)
    ex.attrs = {}
    save(ex, "worked_example.csv")
    Re = Xe.max(0) - Xe.min(0)
    aAB = we * (Xe[0] - Xe[1]) / Re
    save(pd.DataFrame([{"R1": Re[0], "R2": Re[1],
                        "norm1": np.sqrt((Xe[:, 0] ** 2).sum()), "norm2": np.sqrt((Xe[:, 1] ** 2).sum()),
                        "kappa1": a["kappa"][0], "kappa2": a["kappa"][1],
                        "w_imp1": a["w_implied"][0], "w_imp2": a["w_implied"][1],
                        "nu": a["nu"], "delta_w": a["delta_w"],
                        "a_AB_1": aAB[0], "a_AB_2": aAB[1], "sum_a_AB": aAB.sum(),
                        "sum_ka_AB": (a["kappa"] * aAB).sum(),
                        "warnings": str(a["warnings"])}]), "worked_example_audit.csv")
    assert a["warnings"] == [(0, 1)], a["warnings"]

    print("\n[13] LaTeX table bodies, spliced into manuscript.tex and supplementary.tex")
    write_tables()
    inline_tables()

    if make_figs:
        print("\n[14] Figures")
        make_figures()
    print("\nDone.")


# ===========================================================================
# 10. LaTeX table bodies (World Scientific idiom: rows only)
# ===========================================================================
def rd(name):
    return pd.read_csv(os.path.join(RES, name), keep_default_na=False, na_values=[""])


def emit(name, body):
    TABLE_BODIES[name.replace(".tex", "")] = body
    print(f"  built table body {name.replace('.tex', '')}")


def f4(x): return f"{x:.4f}"
def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def f1(x): return f"{x:.1f}"


def write_tables():
    # --- main text --------------------------------------------------------
    ex = rd("worked_example.csv")
    body = "".join(f"{r.alt} & {r.x1:.3f} & {r.x2:.2f} & {r.u1:.3f} & {r.u2:.3f} & "
                   f"{r.r1:.3f} & {r.r2:.3f} & {r.U_minmax:.3f} & {r.U_vector:.3f} & "
                   f"{r.U_implied:.3f} \\\\\n" for r in ex.itertuples())
    emit("worked_example.tex", body)

    post = rd("posteriors.csv")
    body = ""
    for comp in COMPONENTS:
        cells = []
        for scen in SCENARIOS:
            r = post[(post.component == comp) & (post.scenario == scen)].iloc[0]
            cells.append(f"{r['mean']:.4f} ({r.ci_lo:.4f}--{r.ci_hi:.4f})")
        body += f"{comp} & " + " & ".join(cells) + " \\\\\n"
    emit("posteriors.tex", body)

    kd = rd("kappa_diagnostic.csv")
    body = ""
    for scen in SCENARIOS:
        sub = kd[kd.scenario == scen].set_index("criterion").loc[CRITERION_NAMES]
        first = True
        for cname, r in sub.iterrows():
            lab = (f"{SCEN_SHORT[scen]} ($\\nu={r.nu_mean:.2f}$, $\\delta_w={r.delta_w_mean:.3f}$)"
                   if first else "")
            ci = f"({r.kappa_lo:.3f}--{r.kappa_hi:.3f})" if r.kappa_hi - r.kappa_lo > 1e-9 else "--"
            body += (f"{lab} & {cname} & {r.kappa_mean:.3f} & {ci} & {r.w_nominal:.3f} & "
                     f"{r.w_implied:.3f} & ${r.shift_pp:+.1f}$ \\\\\n")
            first = False
        body += "\\colrule\n" if scen != list(SCENARIOS)[-1] else ""
    emit("kappa.tex", body)

    inv = rd("inversion_criterion.csv")
    pooled = rd("inversion_pooled.csv").iloc[0]
    sub = inv[inv.scenario == "Nominal"].sort_values("observed_topsis_rate", ascending=False)
    body = "".join(f"{r.pair.replace('-', '--')} & {r.non_dominated_rate:.3f} & {r.predicted_rate:.4f} & "
                   f"{r.observed_topsis_rate:.4f} & {r.precision_topsis:.3f} & {r.recall_topsis:.3f} \\\\\n"
                   for r in sub.itertuples())
    body += (f"\\colrule\nPooled, all scenarios & -- & {int(pooled.n_pred)} cases & "
             f"{int(pooled.n_obs)} cases & {pooled.precision:.3f} & {pooled.recall:.3f} \\\\\n")
    emit("inversion.tex", body)

    ag = rd("agreement.csv")
    m = ag[ag.block == "main"]
    body = ""
    for pair in ["VIKOR--TOPSIS", "VIKOR--PROMETHEE", "VIKOR--ADD", "TOPSIS--PROMETHEE",
                 "TOPSIS--ADD", "PROMETHEE--ADD"]:
        t = [m[(m.pair == pair) & (m.scenario == s)].iloc[0] for s in SCENARIOS]
        body += (f"{pair} & " + " & ".join(f4(x.tau_b) for x in t) + " & " +
                 " & ".join(f4(x.identical) for x in t) + " \\\\\n")
    emit("agreement.tex", body)

    sh = rd("factorial_shapley.csv")
    body = ""
    for loss, lab, scale in [("identical_rate", "\\makecell[l]{Non-identical\\\\rankings (\\%)}", 100),
                             ("kendall", "\\makecell[l]{Kendall loss\\\\$1-\\bar{\\tau}_b$ ($\\times 10^{-2}$)}", 100)]:
        for k_, scen in enumerate(SCENARIOS):
            r = sh[(sh.loss == loss) & (sh.scenario == scen)].iloc[0]
            head = ("\\multirow{3}{*}{" + lab + "}") if k_ == 0 else ""
            body += (f"{head} & {SCEN_SHORT[scen]} & "
                     f"{r.L_N*scale:.2f} & {r.L_A*scale:.2f} & {r.L_NA*scale:.2f} & "
                     f"${r.interaction*scale:+.2f}$ & {r.phi_N*scale:.2f} & {r.phi_A*scale:.2f} & "
                     f"{100*r.share_N:.1f} \\\\\n")
        if loss == "identical_rate":
            body += "\\colrule\n"
    emit("factorial.tex", body)

    sc = rd("severity_compression.csv")
    body = "".join(f"{r.t:.2f} & {r.severity_min:.3f}--{r.severity_max:.3f} & {r.kappa_severity:.3f} & "
                   f"{r.nu_mean:.2f} & {r.delta_w_mean:.3f} & {100*r.div_normalization:.1f} & "
                   f"{100*r.div_aggregation:.1f} & {100*r.div_both:.1f} & {100*r.share_N:.0f} \\\\\n"
                   for r in sc.itertuples())
    emit("severity.tex", body)

    reg = rd("scaling_regressions.csv")
    body = ""
    for pred, lab in [("log_nu", "$\\log_{10}\\nu$"), ("delta_w", "$\\delta_w$")]:
        for k_, comp in enumerate(["normalization only", "aggregation only", "both"]):
            r = reg[(reg.predictor == pred) & (reg.component == comp)].iloc[0]
            head = ("\\multirow{3}{*}{" + lab + "}") if k_ == 0 else ""
            body += (f"{head} & {comp.capitalize()} & "
                     f"${r.slope:+.3f}$ & {r.se:.4f} & ${r.t:+.1f}$ & "
                     f"{r.r2:.3f} & ${r.spearman:+.3f}$ \\\\\n")
        if pred == "log_nu":
            body += "\\colrule\n"
    emit("scaling.tex", body)

    bn = rd("simulation_binned_nu.csv")
    body = "".join(f"{r.bin} & {int(r.n)} & {r.nu_median:.2f} & {r.delta_w_mean:.3f} & "
                   f"{r.tau_wsm:.3f} ({r.tau_wsm_lo:.3f}--{r.tau_wsm_hi:.3f}) & "
                   f"{r.tau_topsis_minmax:.3f} ({r.tau_topsis_minmax_lo:.3f}--{r.tau_topsis_minmax_hi:.3f}) & "
                   f"{r.tau_topsis:.3f} \\\\\n" for r in bn.itertuples())
    emit("binned_nu.tex", body)

    bd = rd("simulation_binned_deltaw.csv")
    body = "".join(f"{r.bin} & {int(r.n)} & {r.nu_median:.2f} & "
                   f"{r.tau_wsm:.3f} ({r.tau_wsm_lo:.3f}--{r.tau_wsm_hi:.3f}) & "
                   f"{r.tau_topsis_minmax:.3f} ({r.tau_topsis_minmax_lo:.3f}--{r.tau_topsis_minmax_hi:.3f}) & "
                   f"{r.tau_topsis:.3f} \\\\\n" for r in bd.itertuples())
    emit("binned_deltaw.tex", body)

    # --- supplementary ----------------------------------------------------
    cc = rd("calibration_check.csv")
    body = "".join(f"{r.component} & ${r['lambda']:.2e}$ & {r.c_calib:.4f} & ${r.implied_marginal:.2e}$ & "
                   f"{r.ratio:.6f} \\\\\n".replace("e-0", "\\times 10^{-").replace("$ &", "}$ &", 0)
                   for _, r in cc.iterrows())
    # cleaner scientific notation
    body = ""
    for _, r in cc.iterrows():
        def sci(x):
            e = int(np.floor(np.log10(x)))
            return f"${x/10**e:.2f}\\times 10^{{{e}}}$"
        body += f"{r.component} & {sci(r['lambda'])} & {r.c_calib:.4f} & {sci(r.implied_marginal)} & {r.ratio:.6f} \\\\\n"
    emit("calibration.tex", body)

    fa = ag[ag.block == "factorial"]
    body = ""
    for scen in SCENARIOS:
        first = True
        for pair in ["ADD--WSM-V", "ADD--TOPSIS-mm", "ADD--TOPSIS", "WSM-V--TOPSIS",
                     "WSM-V--TOPSIS-mm", "TOPSIS--TOPSIS-mm"]:
            r = fa[(fa.pair == pair) & (fa.scenario == scen)].iloc[0]
            body += (f"{SCEN_SHORT[scen] if first else ''} & {pair} & {f4(r.tau_b)} & {f4(r.rho)} & "
                     f"{f4(r.top1_agree)} & {f4(r.identical)} \\\\\n")
            first = False
        if scen != list(SCENARIOS)[-1]:
            body += "\\colrule\n"
    emit("agreement_full.tex", body)

    lo = rd("localization.csv")
    body = ""
    for scen in SCENARIOS:
        first = True
        for pair in ["VIKOR--TOPSIS", "VIKOR--ADD", "TOPSIS--ADD"]:
            r = lo[(lo.pair == pair) & (lo.scenario == scen)].iloc[0]
            vals = [r[c] for c in ["P-T", "P-S", "P-V", "T-S", "T-V", "S-V"]]
            mx = max(vals)
            cells = [(f"\\bf {v:.3f}" if v == mx else f"{v:.3f}") for v in vals]
            body += f"{SCEN_SHORT[scen] if first else ''} & {pair} & {r.f_dis:.4f} & " + " & ".join(cells) + " \\\\\n"
            first = False
        if scen != list(SCENARIOS)[-1]:
            body += "\\colrule\n"
    emit("localization.tex", body)

    bs = rd("beta_sweep.csv")
    body = ""
    for s in [5.0, 10.0, 20.0, 50.0]:
        first = True
        for scen in SCENARIOS:
            r = bs[(bs.s == s) & (bs.scenario == scen)].iloc[0]
            body += (f"{int(s) if first else ''} & {SCEN_SHORT[scen]} & {f4(r.mean_tau_b)} & "
                     f"{f4(r.mean_identical)} & {r.nu_mean:.3f} & {r.delta_w_mean:.4f} \\\\\n")
            first = False
    emit("beta_sweep.tex", body)

    cb = rd("promethee_configB.csv")
    body = ""
    for scen in SCENARIOS:
        for cfg in ["Config A", "Config B"]:
            r = cb[(cb.scenario == scen) & (cb.config == cfg)].iloc[0]
            body += (f"{SCEN_SHORT[scen] if cfg == 'Config A' else ''} & {cfg} & {f4(r.tau_b)} & "
                     f"{f4(r.top1_agree)} & {f4(r.identical)} \\\\\n")
    emit("configB.tex", body)

    sm = rd("smaa_weight_regimes.csv")
    body = ""
    for scen in SCENARIOS:
        first = True
        for comp in COMPONENTS:
            r = sm[(sm.scenario == scen) & (sm.component == comp)].iloc[0]
            wa = str(r.wc_ahp).replace("[", "(").replace("]", ")")
            wu = str(r.wc_uniform).replace("[", "(").replace("]", ")")
            body += (f"{SCEN_SHORT[scen] if first else ''} & {comp} & {r.b1_ahp:.3f} & {r.b1_entropy:.3f} & "
                     f"{r.b1_uniform:.3f} & {r.pc_ahp:.2f} & {wa} & {wu} \\\\\n")
            first = False
        if scen != list(SCENARIOS)[-1]:
            body += "\\colrule\n"
    emit("smaa.tex", body)

    th = rd("promethee_theta_sweep.csv")
    body = ""
    for t in sorted(th.theta.unique()):
        cells = []
        for scen in SCENARIOS:
            r = th[(th.theta == t) & (th.scenario == scen)].iloc[0]
            cells.append(f"{r.tau_vs_additive:.4f} & {r.tau_vs_topsis:.4f}")
        body += f"{t:.2f} & " + " & ".join(cells) + " \\\\\n"
    emit("theta_sweep.tex", body)

    sc_ = rd("method_scores.csv")
    body = ""
    for scen in SCENARIOS:
        for k_, comp in enumerate(COMPONENTS):
            body += ("\\multirow{4}{*}{" + SCEN_SHORT[scen] + "}" if k_ == 0 else "") + " & "
            cells = []
            for meth in ["VIKOR", "TOPSIS", "PROMETHEE", "ADD"]:
                r = sc_[(sc_.scenario == scen) & (sc_.method == meth) & (sc_.component == comp)].iloc[0]
                cells.append(f"{r.P_R1:.3f}")
            body += f"{comp} & " + " & ".join(cells) + " \\\\\n"
        if scen != list(SCENARIOS)[-1]:
            body += "\\colrule\n"
    emit("rank1.tex", body)


def inline_tables():
    """Splice every generated table body into paper/*.tex between the markers
    '%% BEGIN table:<name>' and '%% END table:<name>', so the documents stay
    self-contained while every number remains generated."""
    import re
    for doc in ["manuscript.tex", "supplementary.tex"]:
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            continue
        txt = open(path).read()

        def repl(m):
            name = m.group(1)
            body = TABLE_BODIES[name]
            return f"%% BEGIN table:{name}\n{body}%% END table:{name}"

        new = re.sub(r"%% BEGIN table:(\S+)\n.*?%% END table:\1", repl, txt, flags=re.S)
        if new != txt:
            open(path, "w").write(new)
            print(f"  refreshed table bodies in {doc}")
        else:
            print(f"  table bodies in {doc} already current")


# ===========================================================================
# 11. Figures (grayscale-safe: lightness + marker/linestyle/hatch encode series)
# ===========================================================================
def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    INK, MID, LIGHT = "#1a1a1a", "#8a8a8a", "#cfcfcf"
    ACC = "#1f5fa8"          # the one accent color
    SER = [ACC, MID, LIGHT]  # series 1 carries the accent; lightness still separates in grayscale
    HATCH = ["", "////", "...."]
    MK = ["o", "s", "^"]
    LS = ["-", (0, (4, 2)), (0, (1.5, 1.5))]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7,
                         "savefig.bbox": "tight", "savefig.dpi": 300, "figure.dpi": 150})

    def tidy(ax, axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis=axis)
        ax.grid(axis="x" if axis == "y" else "y", visible=False)

    def out(fig, name):
        fig.savefig(os.path.join(FIG, name + ".pdf"))
        plt.close(fig)
        print(f"  wrote {name}.pdf")

    scen_list = list(SCENARIOS)

    # Fig: mechanism ------------------------------------------------------
    kd = rd("kappa_diagnostic.csv")
    short = ["Likelihood", "Severity", "Detect. cost"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    xs = np.arange(3)
    wdt = 0.26
    for si, sc in enumerate(scen_list):
        sub = kd[kd.scenario == sc].set_index("criterion").loc[CRITERION_NAMES]
        v = sub.kappa_mean.to_numpy()
        lo = np.clip(v - sub.kappa_lo.to_numpy(), 0, None)
        hi = np.clip(sub.kappa_hi.to_numpy() - v, 0, None)
        ax.bar(xs + (si - 1) * wdt, v, wdt * 0.9, color=SER[si], hatch=HATCH[si],
               edgecolor=INK, linewidth=0.5, label=SCEN_SHORT[sc], zorder=3)
        ax.errorbar(xs + (si - 1) * wdt, v, yerr=[lo, hi], fmt="none", ecolor=INK,
                    elinewidth=0.7, capsize=2, zorder=4)
    ax.axhline(1, color=MID, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylim(0, 1.3); ax.set_ylabel(r"$\kappa_j = R_j / \|\mathbf{x}_j\|_2$")
    ax.set_title("(a) Reweighting factor by criterion", loc="left", fontsize=8)
    ax.legend(loc="upper right", ncol=1)
    tidy(ax)
    ax = axes[1]
    sub = kd[kd.scenario == "Nominal"].set_index("criterion").loc[CRITERION_NAMES]
    wn, wi = sub.w_nominal.to_numpy(), sub.w_implied.to_numpy()
    ax.bar(xs - 0.19, wn, 0.34, color=MID, label="Elicited (AHP)", zorder=3)
    ax.bar(xs + 0.19, wi, 0.34, color=ACC, hatch="////", edgecolor=INK, linewidth=0.5,
           label="Implied by vector norm.", zorder=3)
    for x, n, v in zip(xs, wn, wi):
        ax.text(x, max(n, v) + 0.03, f"{100*(v-n):+.1f} pp", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylim(0, 0.95); ax.set_ylabel("Criterion weight")
    ax.set_title("(b) Elicited vs. implied weights, nominal", loc="left", fontsize=8)
    ax.legend(loc="upper right")
    tidy(ax)
    fig.subplots_adjust(wspace=0.32)
    out(fig, "fig-mechanism")

    # Fig: inversion scatter ---------------------------------------------
    d = rd("sensor_valve_scatter.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    same = ~d.predicted
    hit = d.predicted & d.observed_topsis
    miss = d.predicted & ~d.observed_topsis
    ax.scatter(d.a_sum[same], d.ak_sum[same], s=2.5, c=LIGHT, linewidths=0, zorder=3, rasterized=True)
    ax.scatter(d.a_sum[hit], d.ak_sum[hit], s=10, c=ACC, linewidths=0, zorder=5)
    if miss.sum():
        ax.scatter(d.a_sum[miss], d.ak_sum[miss], s=14, facecolors="none", edgecolors=INK,
                   linewidths=0.6, zorder=5)
    lim = float(np.percentile(np.abs(np.r_[d.a_sum, d.ak_sum]), 99.5)) * 1.12
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    z = 0.13
    axi = ax.inset_axes([0.58, 0.06, 0.34, 0.34])
    axi.scatter(d.a_sum[same], d.ak_sum[same], s=2, c=LIGHT, linewidths=0, rasterized=True)
    axi.scatter(d.a_sum[hit], d.ak_sum[hit], s=7, c=ACC, linewidths=0)
    axi.axhline(0, color=INK, lw=0.5); axi.axvline(0, color=INK, lw=0.5)
    axi.set_xlim(-z, z / 2); axi.set_ylim(-z / 2, z)
    axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
    axi.set_title("origin detail", fontsize=6.5, loc="left", pad=2)
    ax.indicate_inset_zoom(axi, edgecolor=MID, linewidth=0.6)
    ax.set_xlabel(r"$\sum_j a_j$  (min-max order)")
    ax.set_ylabel(r"$\sum_j \kappa_j a_j$  (vector-normalized order)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=LIGHT, label="no inversion predicted"),
                       Line2D([], [], marker="o", ls="", ms=4, color=ACC, label="predicted and inverted by TOPSIS")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    tidy(ax, "both"); ax.grid(True)
    out(fig, "fig-inversion")

    # Fig: factorial + Shapley --------------------------------------------
    fc = rd("factorial_cells.csv")
    sh = rd("factorial_shapley.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    labels = ["Normalization\nonly", "Aggregation\nonly", "Both\n(TOPSIS)"]
    xs = np.arange(3)
    for si, sc in enumerate(scen_list):
        sub = fc[fc.scenario == sc].set_index("cell").loc[["N", "A", "NA"]]
        v = 100 * sub.divergence_rate.to_numpy()
        ax.bar(xs + (si - 1) * wdt, v, wdt * 0.9, color=SER[si], hatch=HATCH[si], edgecolor=INK,
               linewidth=0.5, label=SCEN_SHORT[sc], zorder=3)
        for x, vv in zip(xs + (si - 1) * wdt, v):
            ax.text(x, vv + 0.12, f"{vv:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Non-identical rankings (%)"); ax.set_ylim(0, 8.5)
    ax.set_title("(a) Conditional contrasts", loc="left", fontsize=8)
    ax.legend(loc="upper left"); tidy(ax)
    ax = axes[1]
    s2 = sh[sh.loss == "identical_rate"].set_index("scenario").loc[scen_list]
    ys = np.arange(3)
    ax.barh(ys, 100 * s2.phi_N, color=ACC, label=r"$\phi_N$ (normalization)", zorder=3)
    ax.barh(ys, 100 * s2.phi_A, left=100 * s2.phi_N, color=LIGHT, hatch="////", edgecolor=INK,
            linewidth=0.5, label=r"$\phi_A$ (aggregation)", zorder=3)
    for y, r in zip(ys, s2.itertuples()):
        ax.text(100 * r.L_NA + 0.15, y, f"{100*r.share_N:.0f}% / {100*(1-r.share_N):.0f}%",
                va="center", fontsize=6.5)
    ax.set_yticks(ys); ax.set_yticklabels([SCEN_SHORT[s] for s in scen_list], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 9.5)
    ax.set_xlabel("Shapley share of joint divergence (%)")
    ax.set_title("(b) Exact attribution", loc="left", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); tidy(ax, "x")
    fig.subplots_adjust(wspace=0.38)
    out(fig, "fig-factorial")

    # Fig: threshold sweep -------------------------------------------------
    th = rd("promethee_theta_sweep.csv")
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for si, sc in enumerate(scen_list):
        sub = th[th.scenario == sc].sort_values("theta")
        ax.plot(sub.theta, sub.tau_vs_additive, color=SER[si], marker=MK[si], ms=4, lw=1.4,
                ls="-", label=SCEN_SHORT[sc], zorder=4)
        ax.plot(sub.theta, sub.tau_vs_topsis, color=SER[si], marker=MK[si], ms=3.5, lw=1.1,
                ls=(0, (4, 2)), mfc="white", zorder=3)
    ax.axvline(1, color=MID, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel(r"Preference-threshold ratio $\theta = p_j / R_j$")
    ax.set_ylabel(r"Mean Kendall $\tau_b$")
    l1 = ax.legend(loc="lower right"); ax.add_artist(l1)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.3, label="vs. additive"),
                       Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)), label="vs. TOPSIS")],
              loc="lower left")
    tidy(ax)
    out(fig, "fig-threshold")

    # Fig: simulation study ------------------------------------------------
    bn = rd("simulation_binned_nu.csv")
    bn = bn[bn.n >= 20].reset_index(drop=True)
    x = np.arange(len(bn))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    series = [("tau_wsm", "Normalization only"), ("tau_topsis_minmax", "Aggregation only"),
              ("tau_topsis", "Both (TOPSIS)")]
    ax = axes[0]
    for si, (col, lab) in enumerate(series):
        ax.errorbar(x, bn[col], yerr=[bn[col] - bn[col + "_lo"], bn[col + "_hi"] - bn[col]],
                    color=SER[si], marker=MK[si], ms=4.5, lw=1.4, ls=LS[si], capsize=2,
                    elinewidth=0.7, label=lab, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(bn.bin, rotation=30, ha="right", fontsize=7)
    ax.set_xlabel(r"$\nu$ bin"); ax.set_ylabel(r"Mean Kendall $\tau_b$ vs. additive")
    ax.set_ylim(0.58, 1.02); ax.set_title("(a) Agreement by diagnostic bin", loc="left", fontsize=8)
    tidy(ax)
    ax = axes[1]
    for si, (col, lab) in enumerate(series):
        dcol = "disagree_" + {"tau_wsm": "wsm", "tau_topsis_minmax": "topsis_minmax", "tau_topsis": "topsis"}[col]
        ax.plot(x, 100 * bn[dcol], color=SER[si], marker=MK[si], ms=4.5, lw=1.4, ls=LS[si], label=lab, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(bn.bin, rotation=30, ha="right", fontsize=7)
    ax.set_xlabel(r"$\nu$ bin"); ax.set_ylabel("Non-identical rankings (%)")
    ax.set_ylim(0, 100); ax.set_title("(b) Rate of non-identical rankings", loc="left", fontsize=8)
    tidy(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.3)
    out(fig, "fig-simulation")

    # ---- diagram helpers ----------------------------------------------------
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

    def box(ax, xy, w, h, text, fc="white", ec=INK, lw=0.8, ls="-", fs=7.5, bold_first=False, tc=INK):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        lines = text.split("\n")
        if bold_first:
            ax.text(x, y + 0.13 * (len(lines) - 1), lines[0], ha="center", va="center", fontsize=fs,
                    fontweight="bold", color=tc, zorder=3)
            if len(lines) > 1:
                ax.text(x, y - 0.13, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs - 0.5,
                        color=tc, zorder=3, linespacing=1.3)
        else:
            ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3, linespacing=1.3)

    def arrow(ax, p, q, ls="-", color=INK, lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=1))

    def blank(ax, xlim, ylim):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)

    # Fig 1: audit architecture --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    blank(ax, (0, 12.8), (0, 5.2))
    box(ax, (1.55, 3.9), 2.7, 1.0, "Decision matrix $\\mathbf{X}$\norientations", fc="#f2f2f2")
    box(ax, (1.55, 2.2), 2.7, 1.0, "Weights $\\mathbf{w}$\ntolerances $\\epsilon_R,\\ \\epsilon_m$", fc="#f2f2f2")
    box(ax, (5.3, 3.05), 3.1, 1.2, "Normalization audit\nAlgorithm 1", fc="#e3ecf7", ec=ACC, lw=1.2, bold_first=True, tc=ACC)
    box(ax, (9.9, 3.05), 3.6, 1.55, "Report\n$\\boldsymbol{\\kappa}$, $\\tilde{\\mathbf{w}}$, $\\nu$, $\\delta_w$\nwarning pairs $\\mathcal{P}$, zero-range set $\\mathcal{Z}$",
        bold_first=True)
    box(ax, (9.9, 0.75), 3.6, 1.0, "Ranking method\n(additive, TOPSIS, VIKOR, ...)", fc="#f2f2f2")
    box(ax, (5.3, 0.75), 3.1, 1.0, "Sensitivity run under\nboth normalization rules", ls=(0, (3, 2)))
    arrow(ax, (2.9, 3.9), (3.75, 3.35)); arrow(ax, (2.9, 2.2), (3.75, 2.75))
    arrow(ax, (6.85, 3.05), (8.1, 3.05)); arrow(ax, (9.9, 2.27), (9.9, 1.25))
    ax.text(10.05, 1.75, "$\\delta_w$ small", fontsize=6.5, style="italic", color=MID, va="center")
    arrow(ax, (8.1, 2.5), (6.6, 1.25), ls=(0, (3, 2)), color=MID)
    ax.text(6.55, 2.05, "$\\delta_w$ material", fontsize=6.5, style="italic", color=MID, ha="right")
    arrow(ax, (6.85, 0.75), (8.1, 0.75), ls=(0, (3, 2)), color=MID)
    fig.subplots_adjust(0, 0, 1, 1)
    out(fig, "fig-architecture")

    # Fig 2: 2x2 design -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    blank(ax, (0, 10.4), (0, 5.4))
    ax.text(3.6, 5.05, "min-max normalization", ha="center", fontsize=7.5, style="italic")
    ax.text(7.9, 5.05, "vector normalization", ha="center", fontsize=7.5, style="italic")
    ax.text(0.45, 3.55, "weighted sum", ha="center", va="center", fontsize=7.5, style="italic", rotation=90)
    ax.text(0.45, 1.35, "ideal point\n(Euclidean)", ha="center", va="center", fontsize=7.5, style="italic", rotation=90)
    box(ax, (3.6, 3.55), 3.6, 1.35, "Additive / SMAA kernel\nreference, $L(\\emptyset)=0$", fc="#e3ecf7", ec=ACC, lw=1.2, bold_first=True, tc=ACC)
    box(ax, (7.9, 3.55), 3.6, 1.35, "WSM-V\nnormalization only, $L(N)$", fc="#f2f2f2", bold_first=True)
    box(ax, (3.6, 1.35), 3.6, 1.35, "TOPSIS (min-max)\naggregation only, $L(A)$", fc="#f2f2f2", bold_first=True)
    box(ax, (7.9, 1.35), 3.6, 1.35, "TOPSIS (vector)\nboth, $L(NA)$", bold_first=True)
    arrow(ax, (5.4, 3.55), (6.1, 3.55)); arrow(ax, (3.6, 2.87), (3.6, 2.03))
    arrow(ax, (5.15, 2.87), (6.35, 2.03), ls=(0, (3, 2)), color=MID)
    fig.subplots_adjust(0, 0, 1, 1)
    out(fig, "fig-design")

    # Fig S1: Bayesian network ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    blank(ax, (0, 12.4), (0, 4.6))
    roots = {"Power\nlevel": 1.2, "Temperature\nstress": 4.4, "Maintenance\nquality": 7.8, "Vibration\nlevel": 11.1}
    leaves = {"Sensor\nanomaly": 2.4, "Pump\nfailure": 5.4, "Turbine\nfailure": 8.4, "Valve\nstuck": 11.1}
    for name, x in roots.items():
        ax.add_patch(Ellipse((x, 3.6), 2.3, 1.2, fc="#f2f2f2", ec=INK, lw=0.8, zorder=2))
        ax.text(x, 3.6, name, ha="center", va="center", fontsize=7, zorder=3, linespacing=1.2)
    for name, x in leaves.items():
        box(ax, (x, 0.9), 2.1, 1.1, name, fc="#e3ecf7", ec=ACC, lw=1.0, fs=7)
    edges = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3)]
    rx, lx = list(roots.values()), list(leaves.values())
    for r, l in edges:
        arrow(ax, (rx[r], 3.0), (lx[l], 1.45), lw=0.7, color=MID)
    fig.subplots_adjust(0, 0, 1, 1)
    out(fig, "fig-bn")


if __name__ == "__main__":
    run_all(make_figs="--no-figs" not in sys.argv)
