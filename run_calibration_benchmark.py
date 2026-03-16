# run_calibration_benchmark.py
# ============================================================
# Repeated synthetic calibration benchmark for inverse UQ
#
# Goal:
#   1) reserve an independent calibration pool
#   2) retrain a clean surrogate (Level2 by default) on emulator-only data
#   3) run repeated synthetic-truth Bayesian calibration benchmarks
#
# Outputs:
#   - calibration_benchmark_case_summary.csv
#   - calibration_benchmark_parameter_recovery.csv
#   - calibration_benchmark_observation_fit.csv
#   - calibration_benchmark_meta.json
# ============================================================

import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    ITER1_IDX,
    ITER2_IDX,
    THRESHOLD_SWEEP,
    SEED,
)
from run_phys_levels_main import (
    load_dataset,
    get_device,
    train_with_params,
)


# ============================================================
# User settings
# ============================================================

FINAL_LEVEL = 2
CALIB_HOLDOUT_FRAC = 0.15        # independent calibration pool
N_CASES = 20                     # repeated synthetic truths
CASE_SELECTION = "stress_stratified"   # "random" or "stress_stratified"

OBS_COLS = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]

PRIOR_TYPE = "trunc_gaussian"    # "trunc_gaussian" or "uniform"

N_MCMC = 8000
BURN_IN = 2000
THIN = 5

OBS_NOISE_FRAC = 0.02            # fixed observation noise = frac * training std
PROPOSAL_SCALE = 0.15            # RW proposal std = frac * prior std

SAVE_PER_CASE_POSTERIOR = False  # set True if you want large output files


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


def split_for_benchmark(df):
    """
    Split full dataset into:
      - emulator set: used to train surrogate
      - calibration pool: used only for synthetic truths / inverse benchmark
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


def build_training_args(split, device):
    delta_tr = split["Ytr_s"][:, ITER2_IDX] - split["Ytr_s"][:, ITER1_IDX]
    bias_delta = delta_tr.mean(axis=0)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    x_tr = torch.tensor(split["Xtr_s"], dtype=torch.float32, device=device)
    y_tr = torch.tensor(split["Ytr_s"], dtype=torch.float32, device=device)
    x_va = torch.tensor(split["Xva_s"], dtype=torch.float32, device=device)
    y_va = torch.tensor(split["Yva_s"], dtype=torch.float32, device=device)

    return {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "x_va": x_va,
        "y_va": y_va,
        "Xtr_np": split["Xtr_s"],
        "Ytr_np": split["Ytr_s"],
        "bias_delta_t": bias_delta_t,
        "device": device,
    }


@torch.no_grad()
def predict_single_x(model, sx, sy, x_raw, device):
    x_s = sx.transform(x_raw.reshape(1, -1))
    x = torch.tensor(x_s, dtype=torch.float32, device=device)
    mu_s, logvar_s = model(x)

    mu_s = mu_s.detach().cpu().numpy()[0]
    logvar_s = logvar_s.detach().cpu().numpy()[0]

    mu_raw = sy.inverse_transform(mu_s.reshape(1, -1))[0]
    sigma_raw = np.sqrt(np.exp(logvar_s)) * sy.scale_

    return mu_raw, sigma_raw


def get_prior_stats(split):
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
    lp = 0.0
    for i, c in enumerate(INPUT_COLS):
        xi = x[i]
        lo = prior_stats[c]["min"]
        hi = prior_stats[c]["max"]

        if xi < lo or xi > hi:
            return -np.inf

        if prior_type == "uniform":
            continue

        elif prior_type == "trunc_gaussian":
            mu = prior_stats[c]["mean"]
            sd = prior_stats[c]["std"]
            z = (xi - mu) / sd
            lp += -0.5 * z * z - math.log(sd) - 0.5 * math.log(2 * math.pi)
        else:
            raise ValueError(f"Unsupported PRIOR_TYPE: {prior_type}")

    return float(lp)


def log_likelihood(x, model, sx, sy, y_obs, obs_idx, obs_noise_sigma, device):
    mu_raw, sigma_raw = predict_single_x(model, sx, sy, x, device)

    ll = 0.0
    for k, j in enumerate(obs_idx):
        mu = float(mu_raw[j])
        sigma_model = max(float(sigma_raw[j]), 1e-12)
        sigma_obs = max(float(obs_noise_sigma[k]), 1e-12)

        sigma_total = math.sqrt(sigma_model**2 + sigma_obs**2)
        r = y_obs[k] - mu
        ll += -0.5 * (r / sigma_total) ** 2 - math.log(sigma_total) - 0.5 * math.log(2 * math.pi)

    return float(ll)


def log_posterior(x, prior_stats, model, sx, sy, y_obs, obs_idx, obs_noise_sigma, device):
    lp = log_prior(x, prior_stats, prior_type=PRIOR_TYPE)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(x, model, sx, sy, y_obs, obs_idx, obs_noise_sigma, device)
    return lp + ll


def reflect_to_bounds(x, prior_stats):
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


def run_mh(x0, prior_stats, model, sx, sy, y_obs, obs_idx, obs_noise_sigma, device):
    x_curr = x0.copy()
    curr_lp = log_posterior(
        x_curr, prior_stats, model, sx, sy,
        y_obs, obs_idx, obs_noise_sigma, device
    )

    proposal_std = np.array(
        [prior_stats[c]["std"] * PROPOSAL_SCALE for c in INPUT_COLS],
        dtype=float
    )
    proposal_std = np.maximum(proposal_std, 1e-12)

    chain = np.zeros((N_MCMC, len(INPUT_COLS)), dtype=float)
    logp_chain = np.zeros(N_MCMC, dtype=float)

    n_accept = 0

    for t in range(N_MCMC):
        prop = x_curr + np.random.normal(0.0, proposal_std, size=len(INPUT_COLS))
        prop = reflect_to_bounds(prop, prior_stats)

        prop_lp = log_posterior(
            prop, prior_stats, model, sx, sy,
            y_obs, obs_idx, obs_noise_sigma, device
        )

        if np.log(np.random.rand()) < (prop_lp - curr_lp):
            x_curr = prop
            curr_lp = prop_lp
            n_accept += 1

        chain[t] = x_curr
        logp_chain[t] = curr_lp

    accept_rate = n_accept / float(N_MCMC)
    return chain, logp_chain, accept_rate


def posterior_predictive(samples, model, sx, sy, device):
    mus = []
    sigmas = []

    for x in samples:
        mu_raw, sigma_raw = predict_single_x(model, sx, sy, x, device)
        mus.append(mu_raw)
        sigmas.append(sigma_raw)

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    y_pred = np.random.normal(loc=mus, scale=np.maximum(sigmas, 1e-12))
    return mus, sigmas, y_pred


def summarize_case_posterior(samples):
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


def choose_case_indices(split):
    """
    Select multiple synthetic truths from calibration pool.
    Default: stress-stratified selection.
    """
    n_pool = split["X_cal"].shape[0]
    if N_CASES > n_pool:
        raise ValueError(f"N_CASES={N_CASES} exceeds calibration pool size {n_pool}")

    if CASE_SELECTION == "random":
        idx = np.random.choice(n_pool, size=N_CASES, replace=False)
        return np.sort(idx)

    elif CASE_SELECTION == "stress_stratified":
        stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")
        stress = split["Y_cal"][:, stress_idx]

        order = np.argsort(stress)
        bins = np.array_split(order, N_CASES)

        chosen = []
        for b in bins:
            if len(b) == 0:
                continue
            chosen.append(np.random.choice(b))
        return np.array(chosen, dtype=int)

    else:
        raise ValueError(f"Unsupported CASE_SELECTION: {CASE_SELECTION}")


def compute_feasible_fraction(samples, model, sx, sy, device):
    stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")

    mus = []
    for x in samples:
        mu_raw, _ = predict_single_x(model, sx, sy, x, device)
        mus.append(mu_raw)
    mus = np.asarray(mus)

    stress_mu = mus[:, stress_idx]

    rows = []
    for thr in THRESHOLD_SWEEP:
        mask = stress_mu <= thr
        rows.append({
            "threshold_MPa": float(thr),
            "n_posterior_samples": int(samples.shape[0]),
            "n_feasible": int(np.sum(mask)),
            "feasible_fraction": float(np.mean(mask)),
        })
    return rows


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(OUT_DIR)
    device = get_device()

    # 1) data split
    df = load_dataset()
    split = split_for_benchmark(df)

    # 2) train clean surrogate only on emulator set
    best_params = load_best_params(FINAL_LEVEL)
    args = build_training_args(split, device)

    model, mono_pairs = train_with_params(
        best_params=best_params,
        level=FINAL_LEVEL,
        x_tr=args["x_tr"],
        y_tr=args["y_tr"],
        x_va=args["x_va"],
        y_va=args["y_va"],
        Xtr_np=args["Xtr_np"],
        Ytr_np=args["Ytr_np"],
        bias_delta_t=args["bias_delta_t"],
        device=args["device"],
    )

    # 3) benchmark setup
    prior_stats = get_prior_stats(split)
    obs_idx = [OUTPUT_COLS.index(c) for c in OBS_COLS]

    y_train_std = np.std(split["Y_tr"][:, obs_idx], axis=0) + 1e-12
    obs_noise_sigma = OBS_NOISE_FRAC * y_train_std

    case_indices = choose_case_indices(split)

    case_summary_rows = []
    param_recovery_rows = []
    obs_fit_rows = []

    for bench_id, case_idx in enumerate(case_indices):
        x_true = split["X_cal"][case_idx]
        y_true = split["Y_cal"][case_idx]
        y_obs = y_true[obs_idx].copy()

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
        )

        post = chain[BURN_IN::THIN]
        logp_post = logp_chain[BURN_IN::THIN]

        # posterior summary for parameters
        for i, c in enumerate(INPUT_COLS):
            v = post[:, i]
            q05 = float(np.quantile(v, 0.05))
            q25 = float(np.quantile(v, 0.25))
            q50 = float(np.quantile(v, 0.50))
            q75 = float(np.quantile(v, 0.75))
            q95 = float(np.quantile(v, 0.95))
            true_val = float(x_true[i])

            param_recovery_rows.append({
                "benchmark_case_id": int(bench_id),
                "pool_case_index": int(case_idx),
                "parameter": c,
                "true_value": true_val,
                "posterior_mean": float(np.mean(v)),
                "posterior_std": float(np.std(v)),
                "abs_error_mean": abs(float(np.mean(v)) - true_val),
                "covered_90": bool((true_val >= q05) and (true_val <= q95)),
                "covered_50": bool((true_val >= q25) and (true_val <= q75)),
                "width_90": q95 - q05,
                "width_50": q75 - q25,
                "q05": q05,
                "q50": q50,
                "q95": q95,
            })

        # posterior predictive
        mus, sigmas, y_pred = posterior_predictive(post, model, split["sx"], split["sy"], device)

        for k, c in enumerate(OBS_COLS):
            j = OUTPUT_COLS.index(c)
            pred_mean = float(np.mean(y_pred[:, j]))
            pred_std = float(np.std(y_pred[:, j]))
            q05 = float(np.quantile(y_pred[:, j], 0.05))
            q95 = float(np.quantile(y_pred[:, j], 0.95))
            q025 = float(np.quantile(y_pred[:, j], 0.025))
            q975 = float(np.quantile(y_pred[:, j], 0.975))
            obs_val = float(y_obs[k])

            obs_fit_rows.append({
                "benchmark_case_id": int(bench_id),
                "pool_case_index": int(case_idx),
                "observable": c,
                "y_obs": obs_val,
                "posterior_pred_mean": pred_mean,
                "posterior_pred_std": pred_std,
                "abs_error_mean": abs(pred_mean - obs_val),
                "covered_90": bool((obs_val >= q05) and (obs_val <= q95)),
                "covered_95": bool((obs_val >= q025) and (obs_val <= q975)),
                "width_90": q95 - q05,
                "width_95": q975 - q025,
            })

        # feasible fractions under thresholds
        feasible_rows = compute_feasible_fraction(post, model, split["sx"], split["sy"], device)
        feasible_map = {r["threshold_MPa"]: r["feasible_fraction"] for r in feasible_rows}

        # case-level summary
        case_summary_rows.append({
            "benchmark_case_id": int(bench_id),
            "pool_case_index": int(case_idx),
            "accept_rate": float(accept_rate),
            "n_post_samples": int(post.shape[0]),
            "obs_stress": float(y_obs[OBS_COLS.index("iteration2_max_global_stress")]),
            "obs_keff": float(y_obs[OBS_COLS.index("iteration2_keff")]),
            "mean_abs_obs_fit_error": float(np.mean([
                r["abs_error_mean"] for r in obs_fit_rows if r["benchmark_case_id"] == int(bench_id)
            ])),
            "obs_coverage90_mean": float(np.mean([
                float(r["covered_90"]) for r in obs_fit_rows if r["benchmark_case_id"] == int(bench_id)
            ])),
            "feasible_fraction_110": float(feasible_map.get(110.0, np.nan)),
            "feasible_fraction_120": float(feasible_map.get(120.0, np.nan)),
            "feasible_fraction_131": float(feasible_map.get(131.0, np.nan)),
        })

        # optional save
        if SAVE_PER_CASE_POSTERIOR:
            pd.DataFrame(post, columns=INPUT_COLS).to_csv(
                os.path.join(OUT_DIR, f"benchmark_case{bench_id:03d}_posterior_samples.csv"),
                index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(y_pred, columns=OUTPUT_COLS).to_csv(
                os.path.join(OUT_DIR, f"benchmark_case{bench_id:03d}_posterior_predictive.csv"),
                index=False, encoding="utf-8-sig"
            )

        print(f"[OK] Finished benchmark case {bench_id+1}/{len(case_indices)}")

    # 4) save outputs
    pd.DataFrame(case_summary_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_benchmark_case_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(param_recovery_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_benchmark_parameter_recovery.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(obs_fit_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_benchmark_observation_fit.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 5) aggregate summaries for paper
    # parameter-level aggregate
    df_param = pd.DataFrame(param_recovery_rows)
    df_param_agg = df_param.groupby("parameter").agg(
        mean_abs_error=("abs_error_mean", "mean"),
        mean_width90=("width_90", "mean"),
        coverage90=("covered_90", "mean"),
        coverage50=("covered_50", "mean"),
    ).reset_index()
    df_param_agg.to_csv(
        os.path.join(OUT_DIR, "calibration_benchmark_parameter_recovery_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    # observable-level aggregate
    df_obs = pd.DataFrame(obs_fit_rows)
    df_obs_agg = df_obs.groupby("observable").agg(
        mean_abs_error=("abs_error_mean", "mean"),
        mean_width90=("width_90", "mean"),
        coverage90=("covered_90", "mean"),
        coverage95=("covered_95", "mean"),
    ).reset_index()
    df_obs_agg.to_csv(
        os.path.join(OUT_DIR, "calibration_benchmark_observation_fit_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    meta = {
        "final_level": FINAL_LEVEL,
        "calibration_holdout_frac": CALIB_HOLDOUT_FRAC,
        "n_cases": N_CASES,
        "case_selection": CASE_SELECTION,
        "obs_cols": OBS_COLS,
        "prior_type": PRIOR_TYPE,
        "n_mcmc": N_MCMC,
        "burn_in": BURN_IN,
        "thin": THIN,
        "obs_noise_frac": OBS_NOISE_FRAC,
        "proposal_scale": PROPOSAL_SCALE,
        "save_per_case_posterior": SAVE_PER_CASE_POSTERIOR,
    }
    save_json(meta, os.path.join(OUT_DIR, "calibration_benchmark_meta.json"))

    print("[DONE] Repeated synthetic calibration benchmark completed.")


if __name__ == "__main__":
    main()