# run_extreme_scenario_benchmark.py (已修复)
# from claude
# ============================================================
# Extreme / high-risk scenario inverse calibration benchmark.
#
# Motivation: 82.7% of samples in the dataset exceed the primary
# stress threshold (131 MPa). This script selects N_EXTREME cases
# with the highest simulated stress (> STRESS_FLOOR_MPa) and runs
# posterior inference on them using the Level 2 surrogate.
#
# Scientific purpose:
#   1. Show the surrogate-based MCMC correctly recovers parameters
#      responsible for extreme stress outcomes.
#   2. Demonstrate the posterior feasible region contracts toward
#      high-risk parameter combinations in these scenarios.
#   3. Provide a safety-critical validation complement to the
#      standard 20-case benchmark (which samples from the prior).
#
# Outputs:
#   paper_extreme_stress_posterior_summary.csv
#   paper_extreme_stress_parameter_recovery.csv
#   paper_extreme_stress_risk_assessment.csv
#   paper_extreme_stress_meta.json
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    CSV_PATH, OUT_DIR, INPUT_COLS, OUTPUT_COLS,
    PRIMARY_STRESS_OUTPUT, PRIMARY_AUXILIARY_OUTPUT,
    PRIMARY_STRESS_THRESHOLD, SEED,
    ITER2_IDX,
)
from run_phys_levels_main import (
    load_dataset, get_device, train_with_params, _to_numpy
)
from run_calibration_benchmark import (
    CALIBRATION_INPUT_COLS, OBS_COLS, PRIOR_TYPE,
    N_TOTAL, BURN_IN, THIN, OBS_NOISE_FRAC, PROPOSAL_SCALE,
    get_prior_stats, sample_trunc_gaussian_prior,
    log_prior_sub, reflect_to_bounds_sub,
    subset_input_array, expand_reduced_to_full,
    seed_all, ensure_dir, save_json,
    load_best_params, build_run_suffix,
)

# ============================================================
# Settings
# ============================================================

FINAL_LEVEL         = 2
STRESS_FLOOR_MPa    = 220.0   # only cases with stress > this threshold
N_EXTREME           = 10      # number of extreme cases to calibrate
CALIB_HOLDOUT_FRAC  = 0.15

# Stress bin labels for grouping output
STRESS_BINS = [220, 250, 300]   # edges in MPa; last bin is open


# ============================================================
# Data preparation
# ============================================================

def select_extreme_cases(df, stress_col, floor_mpa, n_cases, rng):
    """Select the N cases with highest stress, above floor_mpa."""
    df_ext = df[df[stress_col] > floor_mpa].copy().reset_index(drop=True)

    if len(df_ext) == 0:
        raise ValueError(f"No cases found above stress floor = {floor_mpa} MPa")

    if len(df_ext) < n_cases:
        print(f"  [WARN] Only {len(df_ext)} cases above {floor_mpa} MPa; using all.")
        n_cases = len(df_ext)

    # 先按 stress 从高到低排序
    df_ext = df_ext.sort_values(stress_col, ascending=False).reset_index(drop=True)

    # 从高应力池里随机抽 n_cases，避免只拿最顶端那几个
    picked_pos = rng.choice(len(df_ext), size=n_cases, replace=False)
    picked_pos = np.sort(picked_pos)

    selected = df_ext.iloc[picked_pos].reset_index(drop=True)
    remaining = df_ext.drop(index=picked_pos).reset_index(drop=True)
    return selected, remaining


# ============================================================
# Posterior inference (mirrors run_calibration_benchmark.py)
# ============================================================

def log_likelihood_sub(theta_sub, obs_vec, obs_sigma, model,
                        sx, sy, full_reference_row):
    theta_full = expand_reduced_to_full(
        theta_sub.reshape(1, -1), CALIBRATION_INPUT_COLS,
        full_reference_row, INPUT_COLS
    )

    x_s = sx.transform(theta_full)
    device = next(model.parameters()).device
    x_t = torch.tensor(x_s, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_s, logvar_s = model(x_t)

    mu_s_np = _to_numpy(mu_s)[0]
    mu_full = sy.inverse_transform(mu_s_np.reshape(1, -1))[0]

    obs_idx = [OUTPUT_COLS.index(c) for c in OBS_COLS]
    pred = mu_full[obs_idx]

    obs_vec = np.asarray(obs_vec, dtype=float)
    obs_sigma = np.asarray(obs_sigma, dtype=float)

    loglk = -0.5 * np.sum(((obs_vec - pred) / obs_sigma) ** 2)
    return float(loglk)


def log_posterior_sub(theta_sub, obs_vec, obs_sigma, prior_stats,
                       model, sx, sy, full_reference_row):
    lp = log_prior_sub(theta_sub, prior_stats)
    if not np.isfinite(lp):
        return -np.inf
    lk = log_likelihood_sub(theta_sub, obs_vec, obs_sigma,
                             model, sx, sy, full_reference_row)
    return lp + lk


def run_mcmc(theta_init, obs_vec, obs_sigma, prior_stats,
             model, sx, sy, full_reference_row, seed):
    rng = np.random.RandomState(seed)

    proposal_std = np.array(
        [prior_stats[c]["std"] * PROPOSAL_SCALE for c in CALIBRATION_INPUT_COLS],
        dtype=float
    )

    chain = np.zeros((N_TOTAL, len(CALIBRATION_INPUT_COLS)), dtype=float)
    theta_curr = np.asarray(theta_init, dtype=float).copy()

    lp_curr = log_posterior_sub(
        theta_curr, obs_vec, obs_sigma, prior_stats,
        model, sx, sy, full_reference_row
    )
    n_accept = 0

    for i in range(N_TOTAL):
        prop = theta_curr + rng.normal(
            loc=0.0, scale=proposal_std, size=len(CALIBRATION_INPUT_COLS)
        )
        prop = reflect_to_bounds_sub(prop, prior_stats)

        lp_prop = log_posterior_sub(
            prop, obs_vec, obs_sigma, prior_stats,
            model, sx, sy, full_reference_row
        )

        log_alpha = lp_prop - lp_curr
        if np.log(rng.uniform()) < log_alpha:
            theta_curr = prop
            lp_curr = lp_prop
            n_accept += 1

        chain[i] = theta_curr

    post_chain = chain[BURN_IN::THIN]
    return post_chain, n_accept / N_TOTAL

def build_iter_pair_indices(output_cols):
    idx_map = {c: i for i, c in enumerate(output_cols)}

    paired_names = [
        ("iteration1_avg_fuel_temp", "iteration2_avg_fuel_temp"),
        ("iteration1_max_fuel_temp", "iteration2_max_fuel_temp"),
        ("iteration1_max_monolith_temp", "iteration2_max_monolith_temp"),
        ("iteration1_max_global_stress", "iteration2_max_global_stress"),
        ("iteration1_monolith_new_temperature", "iteration2_monolith_new_temperature"),
        ("iteration1_Hcore_after", "iteration2_Hcore_after"),
        ("iteration1_wall2", "iteration2_wall2"),
    ]

    iter1_idx = [idx_map[a] for a, b in paired_names]
    iter2_idx = [idx_map[b] for a, b in paired_names]

    return paired_names, iter1_idx, iter2_idx

# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    rng = np.random.RandomState(SEED + 7)

    df = load_dataset()
    device = get_device()

    # ----- train/val/test split (matches main benchmark) -----
    X_full = df[INPUT_COLS].to_numpy(dtype=float)
    Y_full = df[OUTPUT_COLS].to_numpy(dtype=float)

    idx_all  = np.arange(len(df))
    idx_tr_val, idx_hold = train_test_split(
        idx_all, test_size=CALIB_HOLDOUT_FRAC, random_state=SEED
    )
    df_holdout = df.iloc[idx_hold].reset_index(drop=True)
    df_train   = df.iloc[idx_tr_val].reset_index(drop=True)

    X_tr_full = df_train[INPUT_COLS].to_numpy(dtype=float)
    Y_tr_full = df_train[OUTPUT_COLS].to_numpy(dtype=float)
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full, test_size=0.1765, random_state=SEED
    )

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s = sx.transform(X_tr)
    Xva_s = sx.transform(X_va)
    Ytr_s = sy.transform(Y_tr)
    Yva_s = sy.transform(Y_va)

    # ----- train surrogate -----
    best_params = load_best_params(FINAL_LEVEL)
    x_tr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
    x_va_t = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(Yva_s, dtype=torch.float32, device=device)

    # Current 15-output setting:
    # iteration1_keff has been removed, so only 7 iter1<->iter2 paired outputs remain.
    paired_names = [
        ("iteration1_avg_fuel_temp", "iteration2_avg_fuel_temp"),
        ("iteration1_max_fuel_temp", "iteration2_max_fuel_temp"),
        ("iteration1_max_monolith_temp", "iteration2_max_monolith_temp"),
        ("iteration1_max_global_stress", "iteration2_max_global_stress"),
        ("iteration1_monolith_new_temperature", "iteration2_monolith_new_temperature"),
        ("iteration1_Hcore_after", "iteration2_Hcore_after"),
        ("iteration1_wall2", "iteration2_wall2"),
    ]
    idx_map = {c: i for i, c in enumerate(OUTPUT_COLS)}
    iter1_pair_idx = [idx_map[a] for a, b in paired_names]
    iter2_pair_idx = [idx_map[b] for a, b in paired_names]

    delta_tr = Ytr_s[:, iter2_pair_idx] - Ytr_s[:, iter1_pair_idx]
    bias_dt = torch.tensor(delta_tr.mean(axis=0), dtype=torch.float32, device=device)

    model, _ = train_with_params(
        best_params, FINAL_LEVEL,
        x_tr_t, y_tr_t, x_va_t, y_va_t,
        Xtr_s, Ytr_s, bias_dt, device
    )
    model.eval()

    # ----- select extreme cases from holdout -----
    df_extreme, _ = select_extreme_cases(
        df_holdout, PRIMARY_STRESS_OUTPUT, STRESS_FLOOR_MPa, N_EXTREME, rng
    )
    print(f"Selected {len(df_extreme)} extreme-stress cases "
          f"(> {STRESS_FLOOR_MPa} MPa).")
    print(f"  Stress range: {df_extreme[PRIMARY_STRESS_OUTPUT].min():.1f} – "
          f"{df_extreme[PRIMARY_STRESS_OUTPUT].max():.1f} MPa")

    # ----- prior stats from training set -----
    split_like = {
        "X_tr": df_train[INPUT_COLS].to_numpy(dtype=float),
        "X_va": X_va,
    }
    prior_stats = get_prior_stats(split_like)

    # ----- run inverse for each extreme case -----
    summary_rows, recovery_rows, risk_rows = [], [], []
    obs_col_idx = [OUTPUT_COLS.index(c) for c in OBS_COLS]

    for case_i, row in df_extreme.iterrows():
        seed_case = SEED + 1000 + case_i
        seed_all(seed_case)

        true_x_full = row[INPUT_COLS].to_numpy(dtype=float)
        true_y_full = row[OUTPUT_COLS].to_numpy(dtype=float)
        true_stress = row[PRIMARY_STRESS_OUTPUT]

        obs_vec   = true_y_full[obs_col_idx]
        obs_sigma = np.abs(obs_vec) * OBS_NOISE_FRAC + 1e-6

        # Initial theta from prior mean
        theta_init = np.array([prior_stats[c]["mean"]
                                for c in CALIBRATION_INPUT_COLS])

        post_chain, accept_rate = run_mcmc(
            theta_init, obs_vec, obs_sigma, prior_stats,
            model, sx, sy, true_x_full, seed_case
        )

        # ----- posterior predictive stress distribution -----
        n_post = len(post_chain)
        post_x_full = expand_reduced_to_full(
            post_chain, CALIBRATION_INPUT_COLS, true_x_full, INPUT_COLS
        )
        x_post_s = sx.transform(post_x_full)
        x_post_t = torch.tensor(x_post_s, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_post_s, logvar_post_s = model(x_post_t)
        mu_post   = sy.inverse_transform(_to_numpy(mu_post_s))
        sig_post  = np.sqrt(np.exp(_to_numpy(logvar_post_s))) * sy.scale_

        stress_idx = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
        pred_stress_mu   = mu_post[:, stress_idx]
        pred_stress_sig  = sig_post[:, stress_idx]

        # Probability of exceeding threshold under posterior predictive
        z_thresh  = (PRIMARY_STRESS_THRESHOLD - pred_stress_mu) / (pred_stress_sig + 1e-9)
        from scipy.stats import norm
        prob_exceed_post = float(np.mean(1.0 - norm.cdf(z_thresh)))

        # Prior predictive exceedance (from prior samples)
        prior_samples = sample_trunc_gaussian_prior(
            n_post, prior_stats, CALIBRATION_INPUT_COLS,
            np.random.RandomState(seed_case + 1)
        )
        prior_x_full = expand_reduced_to_full(
            prior_samples, CALIBRATION_INPUT_COLS, true_x_full, INPUT_COLS
        )
        x_pr_s = sx.transform(prior_x_full)
        x_pr_t = torch.tensor(x_pr_s, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_pr_s, logvar_pr_s = model(x_pr_t)
        mu_pr  = sy.inverse_transform(_to_numpy(mu_pr_s))
        sig_pr = np.sqrt(np.exp(_to_numpy(logvar_pr_s))) * sy.scale_
        pred_stress_prior_mu  = mu_pr[:, stress_idx]
        pred_stress_prior_sig = sig_pr[:, stress_idx]
        z_prior = (PRIMARY_STRESS_THRESHOLD - pred_stress_prior_mu) / (pred_stress_prior_sig + 1e-9)
        prob_exceed_prior = float(np.mean(1.0 - norm.cdf(z_prior)))

        # ----- parameter recovery -----
        post_mean = post_chain.mean(axis=0)
        post_std  = post_chain.std(axis=0)
        prior_mean = np.array([prior_stats[c]["mean"] for c in CALIBRATION_INPUT_COLS])
        prior_std  = np.array([prior_stats[c]["std"]  for c in CALIBRATION_INPUT_COLS])
        true_x_sub = np.array([true_x_full[INPUT_COLS.index(c)]
                                for c in CALIBRATION_INPUT_COLS])
        contraction = 1.0 - post_std / (prior_std + 1e-30)
        in_90ci = np.abs(true_x_sub - post_mean) <= 1.645 * post_std

        # ----- summary row -----
        summary_rows.append({
            "case_id":           row.get("case_id", case_i),
            "true_stress_MPa":   float(true_stress),
            "stress_bin":        f">{STRESS_BINS[-1]}" if true_stress > STRESS_BINS[-1]
                                 else f"{STRESS_BINS[0]}-{STRESS_BINS[1]}"
                                      if true_stress < STRESS_BINS[1]
                                 else f"{STRESS_BINS[1]}-{STRESS_BINS[-1]}",
            "accept_rate":       float(accept_rate),
            "n_post_samples":    int(n_post),
            "pred_stress_post_mean": float(pred_stress_mu.mean()),
            "pred_stress_post_std":  float(pred_stress_mu.std()),
            "prob_exceed_prior":     prob_exceed_prior,
            "prob_exceed_post":      prob_exceed_post,
            "prob_exceed_delta":     prob_exceed_post - prob_exceed_prior,
            "contraction_mean":      float(contraction.mean()),
            "pct_params_in_90ci":    float(in_90ci.mean()),
        })

        # ----- per-parameter recovery -----
        for j, col in enumerate(CALIBRATION_INPUT_COLS):
            recovery_rows.append({
                "case_id":     row.get("case_id", case_i),
                "true_stress": float(true_stress),
                "param":       col,
                "true_val":    float(true_x_sub[j]),
                "prior_mean":  float(prior_mean[j]),
                "prior_std":   float(prior_std[j]),
                "post_mean":   float(post_mean[j]),
                "post_std":    float(post_std[j]),
                "contraction": float(contraction[j]),
                "in_90ci":     bool(in_90ci[j]),
            })

        # ----- risk assessment row -----
        risk_rows.append({
            "case_id":           row.get("case_id", case_i),
            "true_stress_MPa":   float(true_stress),
            "exceeds_threshold": bool(true_stress > PRIMARY_STRESS_THRESHOLD),
            "prob_exceed_prior": prob_exceed_prior,
            "prob_exceed_post":  prob_exceed_post,
            "risk_updated":      bool(prob_exceed_post > prob_exceed_prior),
        })

        print(f"  Case {case_i}: stress={true_stress:.1f} MPa, "
              f"P(exceed|prior)={prob_exceed_prior:.3f} -> "
              f"P(exceed|post)={prob_exceed_post:.3f}, "
              f"accept={accept_rate:.2f}")

    # ----- save outputs -----
    ensure_dir(OUT_DIR)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "paper_extreme_stress_posterior_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(recovery_rows).to_csv(
        os.path.join(OUT_DIR, "paper_extreme_stress_parameter_recovery.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(risk_rows).to_csv(
        os.path.join(OUT_DIR, "paper_extreme_stress_risk_assessment.csv"),
        index=False, encoding="utf-8-sig"
    )

    meta = {
        "stress_floor_MPa":          STRESS_FLOOR_MPa,
        "n_extreme_cases":           len(df_extreme),
        "stress_range":              [float(df_extreme[PRIMARY_STRESS_OUTPUT].min()),
                                      float(df_extreme[PRIMARY_STRESS_OUTPUT].max())],
        "primary_threshold_MPa":     PRIMARY_STRESS_THRESHOLD,
        "calibration_input_cols":    CALIBRATION_INPUT_COLS,
        "obs_cols":                  OBS_COLS,
        "mcmc_n_total":              N_TOTAL,
        "mcmc_burn_in":              BURN_IN,
        "mcmc_thin":                 THIN,
        "obs_noise_frac":            OBS_NOISE_FRAC,
    }
    save_json(meta, os.path.join(OUT_DIR, "paper_extreme_stress_meta.json"))

    print(f"\n[DONE] Extreme scenario benchmark complete.")
    print(f"  Mean P(exceed|post): "
          f"{pd.DataFrame(risk_rows)['prob_exceed_post'].mean():.3f}")
    print(f"  Mean contraction: "
          f"{pd.DataFrame(summary_rows)['contraction_mean'].mean():.3f}")
    print(f"  Saved to: {OUT_DIR}/paper_extreme_stress_*.csv")


if __name__ == "__main__":
    main()