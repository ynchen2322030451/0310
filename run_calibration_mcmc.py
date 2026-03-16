# run_calibration_mcmc.py
# ============================================================
# Bayesian calibration / inverse inference using a hold-out
# calibration set and a trained surrogate model.
#
# Main model: Level2
# Baseline optional: Level0
#
# Observations used:
#   - iteration2_max_global_stress
#   - iteration2_max_fuel_temp
#   - iteration2_max_monolith_temp
#   - iteration2_wall2
#   - iteration2_keff
# ============================================================

import os
import json
import math
import pickle
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    CSV_PATH,
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    ITER1_IDX,
    ITER2_IDX,
    PRIMARY_STRESS_THRESHOLD,
    THRESHOLD_SWEEP,
    SEED,
)
from run_phys_levels_main import (
    HeteroMLP,
    train_with_params,
    get_device,
    load_dataset,
)


# ============================================================
# User settings
# ============================================================

FINAL_LEVEL = 2                 # use 2 as final selected model
CALIB_HOLDOUT_FRAC = 0.10       # calibration pool, not used to train surrogate
CALIB_CASE_INDEX = 0            # which hold-out case to use as synthetic observation

OBS_COLS = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]

PRIOR_TYPE = "trunc_gaussian"   # "trunc_gaussian" or "uniform"

N_MCMC = 25000
BURN_IN = 5000
THIN = 10

OBS_NOISE_FRAC = 0.02           # fixed observation noise fraction of output std
PROPOSAL_SCALE = 0.15           # random walk step relative to input std

SAVE_POSTERIOR_PRED_SAMPLES = True
N_POST_PRED = 2000


# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_best_params(level: int):
    path = os.path.join(OUT_DIR, f"best_level{level}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing best params file: {path}\n"
            f"Please run `python run_phys_levels_main.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["best_params"]


def split_for_calibration(df):
    """
    Split full dataset into:
      - emulator set (for training/validation the surrogate)
      - calibration pool (never used in surrogate training)
    """
    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)

    X_emul, X_calib, Y_emul, Y_calib = train_test_split(
        X, Y, test_size=CALIB_HOLDOUT_FRAC, random_state=SEED
    )

    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_emul, Y_emul, test_size=0.1765, random_state=SEED
    )

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s = sx.transform(X_tr)
    Xva_s = sx.transform(X_va)
    Xcal_s = sx.transform(X_calib)

    Ytr_s = sy.transform(Y_tr)
    Yva_s = sy.transform(Y_va)
    Ycal_s = sy.transform(Y_calib)

    return {
        "X_tr": X_tr,
        "X_va": X_va,
        "X_cal": X_calib,
        "Y_tr": Y_tr,
        "Y_va": Y_va,
        "Y_cal": Y_calib,
        "Xtr_s": Xtr_s,
        "Xva_s": Xva_s,
        "Xcal_s": Xcal_s,
        "Ytr_s": Ytr_s,
        "Yva_s": Yva_s,
        "Ycal_s": Ycal_s,
        "sx": sx,
        "sy": sy,
    }


def build_pack(split, device):
    delta_tr = split["Ytr_s"][:, ITER2_IDX] - split["Ytr_s"][:, ITER1_IDX]
    bias_delta = delta_tr.mean(axis=0)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    x_tr = torch.tensor(split["Xtr_s"], dtype=torch.float32, device=device)
    y_tr = torch.tensor(split["Ytr_s"], dtype=torch.float32, device=device)
    x_va = torch.tensor(split["Xva_s"], dtype=torch.float32, device=device)
    y_va = torch.tensor(split["Yva_s"], dtype=torch.float32, device=device)

    # keep pack structure compatible with run_phys_levels_main.train_with_params
    pack = (
        x_tr, y_tr, x_va, y_va,
        split["Xtr_s"], split["Ytr_s"],
        None, None,
        bias_delta_t
    )
    return pack


@torch.no_grad()
def predict_single_x(model, sx, sy, x_raw, device):
    """
    x_raw: shape [D] in original input scale
    returns:
      mu_raw, sigma_raw for all outputs
    """
    x_s = sx.transform(x_raw.reshape(1, -1))
    x = torch.tensor(x_s, dtype=torch.float32, device=device)
    mu_s, logvar_s = model(x)

    mu_s = mu_s.detach().cpu().numpy()[0]
    logvar_s = logvar_s.detach().cpu().numpy()[0]

    mu_raw = sy.inverse_transform(mu_s.reshape(1, -1))[0]
    sigma_raw = np.sqrt(np.exp(logvar_s)) * sy.scale_

    return mu_raw, sigma_raw


def get_prior_stats(split):
    """
    Prior based on emulator training split only.
    """
    X_ref = split["X_tr"]
    stats = {}
    for i, c in enumerate(INPUT_COLS):
        col = X_ref[:, i]
        stats[c] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col) + 1e-12),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats


def log_prior(x, prior_stats, prior_type="trunc_gaussian"):
    """
    x in original input scale
    """
    lp = 0.0
    for i, c in enumerate(INPUT_COLS):
        xi = x[i]
        lo = prior_stats[c]["min"]
        hi = prior_stats[c]["max"]

        if xi < lo or xi > hi:
            return -np.inf

        if prior_type == "uniform":
            # constant inside bounds
            continue

        elif prior_type == "trunc_gaussian":
            mu = prior_stats[c]["mean"]
            sd = prior_stats[c]["std"]
            z = (xi - mu) / sd
            lp += -0.5 * z * z - math.log(sd) - 0.5 * math.log(2 * math.pi)

        else:
            raise ValueError(f"Unsupported PRIOR_TYPE: {prior_type}")

    return float(lp)


def log_likelihood(
    x,
    model,
    sx,
    sy,
    y_obs,
    obs_idx,
    obs_noise_sigma,
    device,
):
    mu_raw, sigma_raw = predict_single_x(model, sx, sy, x, device)

    ll = 0.0
    for k, j in enumerate(obs_idx):
        mu = mu_raw[j]
        sigma_model = max(float(sigma_raw[j]), 1e-12)
        sigma_obs = max(float(obs_noise_sigma[k]), 1e-12)

        sigma_total = math.sqrt(sigma_model**2 + sigma_obs**2)
        r = y_obs[k] - mu
        ll += -0.5 * (r / sigma_total) ** 2 - math.log(sigma_total) - 0.5 * math.log(2 * math.pi)

    return float(ll)


def log_posterior(
    x,
    prior_stats,
    model,
    sx,
    sy,
    y_obs,
    obs_idx,
    obs_noise_sigma,
    device,
):
    lp = log_prior(x, prior_stats, prior_type=PRIOR_TYPE)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(
        x=x,
        model=model,
        sx=sx,
        sy=sy,
        y_obs=y_obs,
        obs_idx=obs_idx,
        obs_noise_sigma=obs_noise_sigma,
        device=device,
    )
    return lp + ll


def reflect_to_bounds(x, prior_stats):
    """
    Simple reflective boundary handling
    """
    y = x.copy()
    for i, c in enumerate(INPUT_COLS):
        lo = prior_stats[c]["min"]
        hi = prior_stats[c]["max"]
        if y[i] < lo:
            y[i] = lo + (lo - y[i])
        if y[i] > hi:
            y[i] = hi - (y[i] - hi)
        y[i] = min(max(y[i], lo), hi)
    return y


def run_mh(
    x0,
    prior_stats,
    model,
    sx,
    sy,
    y_obs,
    obs_idx,
    obs_noise_sigma,
    device,
    n_steps=N_MCMC,
):
    x_curr = x0.copy()
    curr_lp = log_posterior(
        x_curr, prior_stats, model, sx, sy,
        y_obs, obs_idx, obs_noise_sigma, device
    )

    proposal_std = np.array([prior_stats[c]["std"] * PROPOSAL_SCALE for c in INPUT_COLS], dtype=float)
    proposal_std = np.maximum(proposal_std, 1e-12)

    chain = np.zeros((n_steps, len(INPUT_COLS)), dtype=float)
    logp_chain = np.zeros(n_steps, dtype=float)

    n_accept = 0

    for t in range(n_steps):
        prop = x_curr + np.random.normal(loc=0.0, scale=proposal_std, size=len(INPUT_COLS))
        prop = reflect_to_bounds(prop, prior_stats)

        prop_lp = log_posterior(
            prop, prior_stats, model, sx, sy,
            y_obs, obs_idx, obs_noise_sigma, device
        )

        log_alpha = prop_lp - curr_lp
        if np.log(np.random.rand()) < log_alpha:
            x_curr = prop
            curr_lp = prop_lp
            n_accept += 1

        chain[t] = x_curr
        logp_chain[t] = curr_lp

    accept_rate = n_accept / float(n_steps)
    return chain, logp_chain, accept_rate


def summarize_posterior(samples):
    rows = []
    for i, c in enumerate(INPUT_COLS):
        v = samples[:, i]
        rows.append({
            "parameter": c,
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "q05": float(np.quantile(v, 0.05)),
            "q25": float(np.quantile(v, 0.25)),
            "q50": float(np.quantile(v, 0.50)),
            "q75": float(np.quantile(v, 0.75)),
            "q95": float(np.quantile(v, 0.95)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        })
    return rows


def posterior_predictive(samples, model, sx, sy, device, n_keep=N_POST_PRED):
    n = min(n_keep, samples.shape[0])
    idx = np.random.choice(samples.shape[0], size=n, replace=False)
    sub = samples[idx]

    mus = []
    sigmas = []
    for x in sub:
        mu_raw, sigma_raw = predict_single_x(model, sx, sy, x, device)
        mus.append(mu_raw)
        sigmas.append(sigma_raw)

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # predictive samples
    y_pred = np.random.normal(loc=mus, scale=np.maximum(sigmas, 1e-12))

    return mus, sigmas, y_pred


def compute_feasible_region(samples, model, sx, sy, device):
    stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")
    rows = []

    mus = []
    for x in samples:
        mu_raw, _ = predict_single_x(model, sx, sy, x, device)
        mus.append(mu_raw)
    mus = np.asarray(mus)

    stress_mu = mus[:, stress_idx]

    for thr in THRESHOLD_SWEEP:
        mask = stress_mu <= thr
        feasible = samples[mask]
        rows.append({
            "threshold_MPa": float(thr),
            "n_posterior_samples": int(samples.shape[0]),
            "n_feasible": int(feasible.shape[0]),
            "feasible_fraction": float(np.mean(mask)),
        })

        if feasible.shape[0] > 0:
            summ = summarize_posterior(feasible)
            pd.DataFrame(summ).to_csv(
                os.path.join(OUT_DIR, f"calibration_feasible_region_summary_thr{int(thr)}.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    return rows


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(OUT_DIR)
    device = get_device()

    df = load_dataset()
    split = split_for_calibration(df)

    # train surrogate only on emulator split
    best_params = load_best_params(FINAL_LEVEL)
    pack = build_pack(split, device)

    x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, _, _, bias_delta_t = pack

    model, mono_pairs = train_with_params(
        best_params=best_params,
        level=FINAL_LEVEL,
        x_tr=x_tr,
        y_tr=y_tr,
        x_va=x_va,
        y_va=y_va,
        Xtr_np=Xtr_np,
        Ytr_np=Ytr_np,
        bias_delta_t=bias_delta_t,
        device=device,
    )
    # choose one calibration case from hold-out pool
    x_true = split["X_cal"][CALIB_CASE_INDEX]
    y_true = split["Y_cal"][CALIB_CASE_INDEX]

    obs_idx = [OUTPUT_COLS.index(c) for c in OBS_COLS]
    y_obs = y_true[obs_idx].copy()

    # fixed observation noise based on emulator-training output std
    y_train_std = np.std(split["Y_tr"][:, obs_idx], axis=0) + 1e-12
    obs_noise_sigma = OBS_NOISE_FRAC * y_train_std

    prior_stats = get_prior_stats(split)

    # initialize at prior mean
    x0 = np.array([prior_stats[c]["mean"] for c in INPUT_COLS], dtype=float)

    chain, logp_chain, accept_rate = run_mh(
        x0=x0,
        prior_stats=prior_stats,
        model=model,
        sx=split["sx"],
        sy=split["sy"],
        y_obs=y_obs,
        obs_idx=obs_idx,
        obs_noise_sigma=obs_noise_sigma,
        device=device,
        n_steps=N_MCMC,
    )

    post = chain[BURN_IN::THIN]
    logp_post = logp_chain[BURN_IN::THIN]

    # posterior summary
    post_summary = summarize_posterior(post)
    pd.DataFrame(post_summary).to_csv(
        os.path.join(OUT_DIR, "calibration_posterior_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(post, columns=INPUT_COLS).to_csv(
        os.path.join(OUT_DIR, "calibration_posterior_samples.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"log_posterior": logp_post}).to_csv(
        os.path.join(OUT_DIR, "calibration_logposterior_trace.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # posterior predictive
    mus, sigmas, y_pred = posterior_predictive(
        samples=post,
        model=model,
        sx=split["sx"],
        sy=split["sy"],
        device=device,
        n_keep=N_POST_PRED,
    )

    pd.DataFrame(mus, columns=OUTPUT_COLS).to_csv(
        os.path.join(OUT_DIR, "calibration_posterior_predictive_mu.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(sigmas, columns=OUTPUT_COLS).to_csv(
        os.path.join(OUT_DIR, "calibration_posterior_predictive_sigma.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(y_pred, columns=OUTPUT_COLS).to_csv(
        os.path.join(OUT_DIR, "calibration_posterior_predictive_samples.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # compare posterior predictive to observation
    obs_compare_rows = []
    for k, c in enumerate(OBS_COLS):
        j = OUTPUT_COLS.index(c)
        pred_mean = float(np.mean(y_pred[:, j]))
        pred_std = float(np.std(y_pred[:, j]))
        obs_compare_rows.append({
            "observable": c,
            "y_obs": float(y_obs[k]),
            "posterior_pred_mean": pred_mean,
            "posterior_pred_std": pred_std,
            "abs_error_mean": abs(pred_mean - float(y_obs[k])),
        })

    pd.DataFrame(obs_compare_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_observation_fit.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # feasible region under stress thresholds
    feasible_rows = compute_feasible_region(
        samples=post,
        model=model,
        sx=split["sx"],
        sy=split["sy"],
        device=device,
    )
    pd.DataFrame(feasible_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_feasible_region_overview.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # metadata
    meta = {
        "final_level": FINAL_LEVEL,
        "calibration_holdout_frac": CALIB_HOLDOUT_FRAC,
        "calibration_case_index": CALIB_CASE_INDEX,
        "obs_cols": OBS_COLS,
        "prior_type": PRIOR_TYPE,
        "n_mcmc": N_MCMC,
        "burn_in": BURN_IN,
        "thin": THIN,
        "obs_noise_frac": OBS_NOISE_FRAC,
        "proposal_scale": PROPOSAL_SCALE,
        "accept_rate": accept_rate,
        "x_true": {c: float(v) for c, v in zip(INPUT_COLS, x_true)},
        "y_obs": {c: float(v) for c, v in zip(OBS_COLS, y_obs)},
    }
    save_json(meta, os.path.join(OUT_DIR, "calibration_run_meta.json"))

    print("[DONE] Calibration / MCMC completed.")
    print(f"Acceptance rate = {accept_rate:.3f}")


if __name__ == "__main__":
    main()