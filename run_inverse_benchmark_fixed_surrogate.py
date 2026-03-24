# run_inverse_benchmark_fixed_surrogate.py
# ============================================================
# Fixed-surrogate repeated synthetic inverse benchmark
#
# Main principles:
#   1) split full dataset once and save the split
#   2) calibration pool is strictly separated from emulator pool
#   3) load fixed pretrained surrogate (checkpoint + scalers)
#   4) run repeated synthetic inverse benchmark on calibration pool
#
# Supports:
#   - full inverse
#   - reduced_maintext inverse
#
# Main outputs:
#   - inverse_split_meta.json
#   - inverse_emulator_pool.csv
#   - inverse_calibration_pool.csv
#   - inverse_case_indices_<run_tag>.csv
#   - inverse_benchmark_case_summary_<run_tag>.csv
#   - inverse_benchmark_parameter_recovery_<run_tag>.csv
#   - inverse_benchmark_parameter_recovery_summary_<run_tag>.csv
#   - inverse_benchmark_observation_fit_<run_tag>.csv
#   - inverse_benchmark_observation_fit_summary_<run_tag>.csv
#   - inverse_benchmark_meta_<run_tag>.json
#   - inverse_diagnostics_summary_<run_tag>.json
#
# Optional per-case outputs:
#   - benchmark_caseXXX_prior_samples_<run_tag>.csv
#   - benchmark_caseXXX_posterior_samples_<run_tag>.csv
#   - benchmark_caseXXX_posterior_predictive_<run_tag>.csv
#   - benchmark_caseXXX_full_chain_<run_tag>.csv
# ============================================================

import os
import json
import math
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    THRESHOLD_SWEEP,
    SEED,
)
from run_phys_levels_main import (
    load_dataset,
    get_device,
    HeteroMLP,
)

# ============================================================
# User settings
# ============================================================

FINAL_LEVEL = 2
RUN_TAG = "reduced_maintext"   # "full" or "reduced_maintext"

# split
CALIB_HOLDOUT_FRAC = 0.15
SAVE_SPLIT_FILES = True
RESAVE_SPLIT_IF_EXISTS = False

# repeated benchmark
N_CASES = 50
CASE_SELECTION = "stress_stratified"   # "random" or "stress_stratified"

# observations used in inverse likelihood
OBS_COLS = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]

# prior / MCMC
PRIOR_TYPE = "trunc_gaussian"   # "trunc_gaussian" or "uniform"
N_TOTAL = 8000
BURN_IN = 2000
THIN = 5
OBS_NOISE_FRAC = 0.02
PROPOSAL_SCALE = 0.15

# optional exports
SAVE_PRIOR_SAMPLES = True
N_PRIOR_EXPORT = 5000
SAVE_PER_CASE_POSTERIOR = True
SAVE_FULL_CHAIN_FOR_REP_CASES = True
REPRESENTATIVE_CASE_FOR_TRACE_MODE = "closest_to_threshold"  # "first" or "closest_to_threshold"
PRIMARY_STRESS_THRESHOLD = 131.0
REP_CASE_STRESS_WINDOW = 5.0

# reduced set frozen for main text
if RUN_TAG == "full":
    CALIBRATION_INPUT_COLS = INPUT_COLS
elif RUN_TAG == "reduced_maintext":
    CALIBRATION_INPUT_COLS = ["E_intercept", "alpha_base", "alpha_slope", "nu"]
else:
    raise ValueError("RUN_TAG must be 'full' or 'reduced_maintext'")

# file paths
ROOT_OUT = Path(OUT_DIR)
SPLIT_META_PATH = ROOT_OUT / "inverse_split_meta.json"
EMULATOR_POOL_PATH = ROOT_OUT / "inverse_emulator_pool.csv"
CALIB_POOL_PATH = ROOT_OUT / "inverse_calibration_pool.csv"


# ============================================================
# basic utils
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


def build_run_suffix():
    return f"_{RUN_TAG}"


def subset_input_array(x_full: np.ndarray, full_cols, sub_cols):
    idx = [full_cols.index(c) for c in sub_cols]
    return x_full[:, idx]


def expand_reduced_to_full(x_reduced: np.ndarray, reduced_cols, full_reference_row: np.ndarray, full_cols):
    out = np.tile(full_reference_row.reshape(1, -1), (x_reduced.shape[0], 1))
    for j, c in enumerate(reduced_cols):
        full_j = full_cols.index(c)
        out[:, full_j] = x_reduced[:, j]
    return out


# ============================================================
# split management
# ============================================================

def make_pool_dataframe(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    df_x = pd.DataFrame(X, columns=INPUT_COLS)
    df_y = pd.DataFrame(Y, columns=OUTPUT_COLS)
    return pd.concat([df_x, df_y], axis=1)


def create_and_save_split(df_full: pd.DataFrame):
    X = df_full[INPUT_COLS].to_numpy(dtype=float)
    Y = df_full[OUTPUT_COLS].to_numpy(dtype=float)

    X_emul, X_calib, Y_emul, Y_calib = train_test_split(
        X, Y,
        test_size=CALIB_HOLDOUT_FRAC,
        random_state=SEED
    )

    df_emul = make_pool_dataframe(X_emul, Y_emul)
    df_calib = make_pool_dataframe(X_calib, Y_calib)

    if SAVE_SPLIT_FILES:
        df_emul.to_csv(EMULATOR_POOL_PATH, index=False, encoding="utf-8-sig")
        df_calib.to_csv(CALIB_POOL_PATH, index=False, encoding="utf-8-sig")

        split_meta = {
            "seed": SEED,
            "calib_holdout_frac": CALIB_HOLDOUT_FRAC,
            "n_total": int(len(df_full)),
            "n_emulator_pool": int(len(df_emul)),
            "n_calibration_pool": int(len(df_calib)),
            "input_cols": INPUT_COLS,
            "output_cols": OUTPUT_COLS,
        }
        save_json(split_meta, SPLIT_META_PATH)

    return df_emul, df_calib


def load_or_create_split(df_full: pd.DataFrame):
    if (
        not RESAVE_SPLIT_IF_EXISTS
        and EMULATOR_POOL_PATH.exists()
        and CALIB_POOL_PATH.exists()
        and SPLIT_META_PATH.exists()
    ):
        df_emul = pd.read_csv(EMULATOR_POOL_PATH)
        df_calib = pd.read_csv(CALIB_POOL_PATH)
        return df_emul, df_calib

    return create_and_save_split(df_full)


# ============================================================
# fixed surrogate loading
# ============================================================

def load_fixed_surrogate(level: int, device):
    ckpt_path = ROOT_OUT / f"checkpoint_level{level}.pt"
    scaler_path = ROOT_OUT / f"scalers_level{level}.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scalers: {scaler_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    sx = scalers["sx"]
    sy = scalers["sy"]

    bp = ckpt["best_params"]
    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(bp["width"]),
        depth=int(bp["depth"]),
        dropout=float(bp["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    return model, sx, sy, bp


@torch.no_grad()
def predict_single_x_full(model, sx, sy, x_full_raw, device):
    x_s = sx.transform(x_full_raw.reshape(1, -1))
    x = torch.tensor(x_s, dtype=torch.float32, device=device)
    mu_s, logvar_s = model(x)

    mu_s = mu_s.detach().cpu().numpy()[0]
    logvar_s = logvar_s.detach().cpu().numpy()[0]

    mu_raw = sy.inverse_transform(mu_s.reshape(1, -1))[0]
    sigma_raw = np.sqrt(np.exp(logvar_s)) * sy.scale_
    return mu_raw, sigma_raw


# ============================================================
# prior helpers
# ============================================================

def get_prior_stats_from_emulator_pool(df_emul: pd.DataFrame):
    X_ref_full = df_emul[INPUT_COLS].to_numpy(dtype=float)
    X_ref_sub = subset_input_array(X_ref_full, INPUT_COLS, CALIBRATION_INPUT_COLS)

    stats = {}
    for i, c in enumerate(CALIBRATION_INPUT_COLS):
        col = X_ref_sub[:, i]
        stats[c] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col) + 1e-12),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats


def sample_uniform_prior(n: int, bounds_dict: dict, cols: list, rng: np.random.RandomState):
    arr = np.zeros((n, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        lo, hi = bounds_dict[c]
        arr[:, j] = rng.uniform(lo, hi, size=n)
    return arr


def sample_trunc_gaussian_prior(n: int, prior_stats: dict, cols: list, rng: np.random.RandomState):
    arr = np.zeros((n, len(cols)), dtype=float)

    for j, c in enumerate(cols):
        mu = float(prior_stats[c]["mean"])
        sd = float(prior_stats[c]["std"])
        lo = float(prior_stats[c]["min"])
        hi = float(prior_stats[c]["max"])

        vals = []
        batch = max(1000, n)

        while len(vals) < n:
            cand = rng.normal(loc=mu, scale=sd, size=batch)
            cand = cand[(cand >= lo) & (cand <= hi)]
            if cand.size > 0:
                vals.extend(cand.tolist())

        arr[:, j] = np.asarray(vals[:n], dtype=float)

    return arr


def log_prior_sub(theta_sub, prior_stats, prior_type="trunc_gaussian"):
    lp = 0.0
    for i, c in enumerate(CALIBRATION_INPUT_COLS):
        xi = theta_sub[i]
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


def reflect_to_bounds_sub(theta_sub, prior_stats):
    y = theta_sub.copy()
    for i, c in enumerate(CALIBRATION_INPUT_COLS):
        lo = prior_stats[c]["min"]
        hi = prior_stats[c]["max"]
        if y[i] < lo:
            y[i] = lo + (lo - y[i])
        if y[i] > hi:
            y[i] = hi - (y[i] - hi)
        y[i] = min(max(y[i], lo), hi)
    return y


# ============================================================
# case selection
# ============================================================

def choose_case_indices(df_calib: pd.DataFrame):
    n_pool = len(df_calib)
    if N_CASES > n_pool:
        raise ValueError(f"N_CASES={N_CASES} exceeds calibration pool size {n_pool}")

    y_cal = df_calib[OUTPUT_COLS].to_numpy(dtype=float)

    if CASE_SELECTION == "random":
        idx = np.random.choice(n_pool, size=N_CASES, replace=False)
        idx = np.sort(idx)

    elif CASE_SELECTION == "stress_stratified":
        stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")
        stress = y_cal[:, stress_idx]
        order = np.argsort(stress)
        bins = np.array_split(order, N_CASES)

        chosen = []
        for b in bins:
            if len(b) == 0:
                continue
            chosen.append(np.random.choice(b))
        idx = np.asarray(chosen, dtype=int)

    else:
        raise ValueError(f"Unsupported CASE_SELECTION: {CASE_SELECTION}")

    return idx


# ============================================================
# likelihood / posterior
# ============================================================

def log_likelihood_sub(theta_sub, model, sx, sy, x_ref_full, y_obs, obs_idx, obs_noise_sigma, device):
    x_full = expand_reduced_to_full(
        theta_sub.reshape(1, -1),
        reduced_cols=CALIBRATION_INPUT_COLS,
        full_reference_row=x_ref_full,
        full_cols=INPUT_COLS
    )[0]

    mu_raw, sigma_raw = predict_single_x_full(model, sx, sy, x_full, device)

    ll = 0.0
    for k, j in enumerate(obs_idx):
        mu = float(mu_raw[j])
        sigma_model = max(float(sigma_raw[j]), 1e-12)
        sigma_obs = max(float(obs_noise_sigma[k]), 1e-12)
        sigma_total = math.sqrt(sigma_model**2 + sigma_obs**2)

        r = y_obs[k] - mu
        ll += -0.5 * (r / sigma_total) ** 2 - math.log(sigma_total) - 0.5 * math.log(2 * math.pi)

    return float(ll)


def log_posterior_sub(theta_sub, prior_stats, model, sx, sy, x_ref_full, y_obs, obs_idx, obs_noise_sigma, device):
    lp = log_prior_sub(theta_sub, prior_stats, prior_type=PRIOR_TYPE)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_sub(theta_sub, model, sx, sy, x_ref_full, y_obs, obs_idx, obs_noise_sigma, device)
    return lp + ll


def run_mh_sub(x0_sub, prior_stats, model, sx, sy, x_ref_full, y_obs, obs_idx, obs_noise_sigma, device):
    theta_curr = x0_sub.copy()
    curr_lp = log_posterior_sub(
        theta_curr, prior_stats, model, sx, sy,
        x_ref_full, y_obs, obs_idx, obs_noise_sigma, device
    )

    proposal_std = np.array(
        [prior_stats[c]["std"] * PROPOSAL_SCALE for c in CALIBRATION_INPUT_COLS],
        dtype=float
    )
    proposal_std = np.maximum(proposal_std, 1e-12)

    chain = np.zeros((N_TOTAL, len(CALIBRATION_INPUT_COLS)), dtype=float)
    logp_chain = np.zeros(N_TOTAL, dtype=float)

    n_accept = 0

    for t in range(N_TOTAL):
        prop = theta_curr + np.random.normal(0.0, proposal_std, size=len(CALIBRATION_INPUT_COLS))
        prop = reflect_to_bounds_sub(prop, prior_stats)

        prop_lp = log_posterior_sub(
            prop, prior_stats, model, sx, sy,
            x_ref_full, y_obs, obs_idx, obs_noise_sigma, device
        )

        if np.log(np.random.rand()) < (prop_lp - curr_lp):
            theta_curr = prop
            curr_lp = prop_lp
            n_accept += 1

        chain[t] = theta_curr
        logp_chain[t] = curr_lp

    accept_rate = n_accept / float(N_TOTAL)
    return chain, logp_chain, accept_rate


def posterior_predictive_from_subspace(samples_sub, model, sx, sy, x_ref_full, device):
    mus = []
    sigmas = []

    X_full = expand_reduced_to_full(
        samples_sub,
        reduced_cols=CALIBRATION_INPUT_COLS,
        full_reference_row=x_ref_full,
        full_cols=INPUT_COLS
    )

    for x_full in X_full:
        mu_raw, sigma_raw = predict_single_x_full(model, sx, sy, x_full, device)
        mus.append(mu_raw)
        sigmas.append(sigma_raw)

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    y_pred = np.random.normal(loc=mus, scale=np.maximum(sigmas, 1e-12))
    return mus, sigmas, y_pred


def compute_feasible_fraction(samples_sub, model, sx, sy, x_ref_full, device):
    stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")

    X_full = expand_reduced_to_full(
        samples_sub,
        reduced_cols=CALIBRATION_INPUT_COLS,
        full_reference_row=x_ref_full,
        full_cols=INPUT_COLS
    )

    mus = []
    for x_full in X_full:
        mu_raw, _ = predict_single_x_full(model, sx, sy, x_full, device)
        mus.append(mu_raw)
    mus = np.asarray(mus)

    stress_mu = mus[:, stress_idx]

    rows = []
    for thr in THRESHOLD_SWEEP:
        mask = stress_mu <= thr
        rows.append({
            "threshold_MPa": float(thr),
            "n_posterior_samples": int(samples_sub.shape[0]),
            "n_feasible": int(np.sum(mask)),
            "feasible_fraction": float(np.mean(mask)),
        })
    return rows


# ============================================================
# main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(OUT_DIR)
    device = get_device()
    run_suffix = build_run_suffix()

    # 1) load full dataset and create saved split
    df_full = load_dataset()
    df_emul, df_calib = load_or_create_split(df_full)

    # 2) load fixed surrogate
    model, sx, sy, best_params = load_fixed_surrogate(FINAL_LEVEL, device)

    # 3) prior / likelihood setup
    prior_stats = get_prior_stats_from_emulator_pool(df_emul)
    obs_idx = [OUTPUT_COLS.index(c) for c in OBS_COLS]

    y_emul = df_emul[OUTPUT_COLS].to_numpy(dtype=float)
    y_train_std = np.std(y_emul[:, obs_idx], axis=0) + 1e-12
    obs_noise_sigma = OBS_NOISE_FRAC * y_train_std

    # 4) choose benchmark cases from calibration pool
    case_indices = choose_case_indices(df_calib)
    df_case_idx = pd.DataFrame({
        "benchmark_case_id": np.arange(len(case_indices)),
        "pool_case_index": case_indices
    })
    df_case_idx.to_csv(
        ROOT_OUT / f"inverse_case_indices{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    X_cal = df_calib[INPUT_COLS].to_numpy(dtype=float)
    Y_cal = df_calib[OUTPUT_COLS].to_numpy(dtype=float)
    X_emul = df_emul[INPUT_COLS].to_numpy(dtype=float)

    case_summary_rows = []
    param_recovery_rows = []
    obs_fit_rows = []

    # fixed reference point for non-calibrated parameters
    x_ref_full_global = np.mean(X_emul, axis=0)

    for bench_id, case_idx in enumerate(case_indices):
        x_true_full = X_cal[case_idx]
        y_true = Y_cal[case_idx]
        y_obs = y_true[obs_idx].copy()

        x_true_sub = subset_input_array(
            x_true_full.reshape(1, -1), INPUT_COLS, CALIBRATION_INPUT_COLS
        )[0]

        # main-text choice:
        # non-calibrated parameters are fixed at emulator-pool reference point
        x_ref_full = x_ref_full_global.copy()

        # initialize at prior mean
        x0_sub = np.array([prior_stats[c]["mean"] for c in CALIBRATION_INPUT_COLS], dtype=float)

        # prior export
        if SAVE_PRIOR_SAMPLES:
            rng_prior = np.random.RandomState(SEED + 10000 + bench_id)

            if PRIOR_TYPE == "uniform":
                prior_sub = sample_uniform_prior(
                    n=N_PRIOR_EXPORT,
                    bounds_dict={c: (prior_stats[c]["min"], prior_stats[c]["max"]) for c in CALIBRATION_INPUT_COLS},
                    cols=CALIBRATION_INPUT_COLS,
                    rng=rng_prior
                )
            elif PRIOR_TYPE == "trunc_gaussian":
                prior_sub = sample_trunc_gaussian_prior(
                    n=N_PRIOR_EXPORT,
                    prior_stats=prior_stats,
                    cols=CALIBRATION_INPUT_COLS,
                    rng=rng_prior
                )
            else:
                raise ValueError(f"Unsupported PRIOR_TYPE for prior export: {PRIOR_TYPE}")

            pd.DataFrame(prior_sub, columns=CALIBRATION_INPUT_COLS).to_csv(
                ROOT_OUT / f"benchmark_case{bench_id:03d}_prior_samples{run_suffix}.csv",
                index=False, encoding="utf-8-sig"
            )

        # MCMC
        chain, logp_chain, accept_rate = run_mh_sub(
            x0_sub=x0_sub,
            prior_stats=prior_stats,
            model=model,
            sx=sx,
            sy=sy,
            x_ref_full=x_ref_full,
            y_obs=y_obs,
            obs_idx=obs_idx,
            obs_noise_sigma=obs_noise_sigma,
            device=device,
        )

        post = chain[BURN_IN::THIN]

        # parameter recovery
        for i, c in enumerate(CALIBRATION_INPUT_COLS):
            v = post[:, i]
            q05 = float(np.quantile(v, 0.05))
            q25 = float(np.quantile(v, 0.25))
            q50 = float(np.quantile(v, 0.50))
            q75 = float(np.quantile(v, 0.75))
            q95 = float(np.quantile(v, 0.95))
            true_val = float(x_true_sub[i])

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
        mus, sigmas, y_pred = posterior_predictive_from_subspace(
            post, model, sx, sy, x_ref_full, device
        )

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

        # feasible fractions
        feasible_rows = compute_feasible_fraction(
            post, model, sx, sy, x_ref_full, device
        )
        feasible_map = {r["threshold_MPa"]: r["feasible_fraction"] for r in feasible_rows}

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

        # representative full chain
        save_this_chain = False
        if SAVE_FULL_CHAIN_FOR_REP_CASES:
            if REPRESENTATIVE_CASE_FOR_TRACE_MODE == "first" and bench_id == 0:
                save_this_chain = True
            elif REPRESENTATIVE_CASE_FOR_TRACE_MODE == "closest_to_threshold":
                if abs(y_obs[OBS_COLS.index("iteration2_max_global_stress")] - PRIMARY_STRESS_THRESHOLD) <= REP_CASE_STRESS_WINDOW:
                    save_this_chain = True

        if save_this_chain:
            pd.DataFrame(chain, columns=CALIBRATION_INPUT_COLS).to_csv(
                ROOT_OUT / f"benchmark_case{bench_id:03d}_full_chain{run_suffix}.csv",
                index=False, encoding="utf-8-sig"
            )

        if SAVE_PER_CASE_POSTERIOR:
            pd.DataFrame(post, columns=CALIBRATION_INPUT_COLS).to_csv(
                ROOT_OUT / f"benchmark_case{bench_id:03d}_posterior_samples{run_suffix}.csv",
                index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(y_pred, columns=OUTPUT_COLS).to_csv(
                ROOT_OUT / f"benchmark_case{bench_id:03d}_posterior_predictive{run_suffix}.csv",
                index=False, encoding="utf-8-sig"
            )

        print(f"[OK] Finished benchmark case {bench_id + 1}/{len(case_indices)}")

    # 5) save outputs
    df_case = pd.DataFrame(case_summary_rows)
    df_param = pd.DataFrame(param_recovery_rows)
    df_obs = pd.DataFrame(obs_fit_rows)

    df_case.to_csv(
        ROOT_OUT / f"inverse_benchmark_case_summary{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )
    df_param.to_csv(
        ROOT_OUT / f"inverse_benchmark_parameter_recovery{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )
    df_obs.to_csv(
        ROOT_OUT / f"inverse_benchmark_observation_fit{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    df_param_agg = df_param.groupby("parameter").agg(
        mean_abs_error=("abs_error_mean", "mean"),
        mean_width90=("width_90", "mean"),
        coverage90=("covered_90", "mean"),
        coverage50=("covered_50", "mean"),
    ).reset_index()
    df_param_agg.to_csv(
        ROOT_OUT / f"inverse_benchmark_parameter_recovery_summary{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    df_obs_agg = df_obs.groupby("observable").agg(
        mean_abs_error=("abs_error_mean", "mean"),
        mean_width90=("width_90", "mean"),
        coverage90=("covered_90", "mean"),
        coverage95=("covered_95", "mean"),
    ).reset_index()
    df_obs_agg.to_csv(
        ROOT_OUT / f"inverse_benchmark_observation_fit_summary{run_suffix}.csv",
        index=False, encoding="utf-8-sig"
    )

    diag = {
        "run_tag": RUN_TAG,
        "final_level": FINAL_LEVEL,
        "fixed_surrogate": True,
        "n_benchmark_cases": int(len(df_case)),
        "mean_accept_rate": float(df_case["accept_rate"].mean()),
        "mean_obs_fit_error": float(df_case["mean_abs_obs_fit_error"].mean()),
        "mean_obs_coverage90": float(df_case["obs_coverage90_mean"].mean()),
        "mean_feasible_fraction_110": float(df_case["feasible_fraction_110"].mean()),
        "mean_feasible_fraction_120": float(df_case["feasible_fraction_120"].mean()),
        "mean_feasible_fraction_131": float(df_case["feasible_fraction_131"].mean()),
    }
    save_json(diag, ROOT_OUT / f"inverse_diagnostics_summary{run_suffix}.json")

    meta = {
        "final_level": FINAL_LEVEL,
        "fixed_surrogate": True,
        "run_tag": RUN_TAG,
        "calibration_input_cols": CALIBRATION_INPUT_COLS,
        "calib_holdout_frac": CALIB_HOLDOUT_FRAC,
        "n_cases": N_CASES,
        "case_selection": CASE_SELECTION,
        "obs_cols": OBS_COLS,
        "prior_type": PRIOR_TYPE,
        "n_mcmc": N_TOTAL,
        "burn_in": BURN_IN,
        "thin": THIN,
        "obs_noise_frac": OBS_NOISE_FRAC,
        "proposal_scale": PROPOSAL_SCALE,
        "reference_strategy_for_noncalibrated_params": "emulator_pool_mean",
        "checkpoint_level": FINAL_LEVEL,
        "save_per_case_posterior": SAVE_PER_CASE_POSTERIOR,
    }
    save_json(meta, ROOT_OUT / f"inverse_benchmark_meta{run_suffix}.json")

    print(f"[DONE] Fixed-surrogate inverse benchmark completed ({RUN_TAG}).")


if __name__ == "__main__":
    main()