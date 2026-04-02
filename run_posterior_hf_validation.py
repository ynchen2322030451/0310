# run_posterior_hf_validation.py
# ============================================================
# Posterior-informed parameter validation against existing
# high-fidelity (HF) data via nearest-neighbor proxy matching.
#
# IMPORTANT SCIENTIFIC NOTE
# -------------------------
# This script does NOT rerun the high-fidelity solver at the
# posterior mean. Instead, it:
#
#   HF test case (the "held-out experiment")
#     -> MCMC calibrates 4 surrogate input parameters
#     -> posterior mean in 8D input space
#     -> nearest HF neighbor(s) searched from existing HF library
#        (default: train+val only, excluding all test points)
#     -> compare errors of:
#          (a) global prior mean predictor
#          (b) conditional prior predictor
#          (c) posterior mean predictor
#          (d) nearest-HF-neighbor proxy truth
#
# Therefore this should be described in the paper as:
#   "posterior-informed HF proxy validation"
# or
#   "nearest-neighbor high-fidelity consistency check"
#
# Model:  8 inputs -> 15 outputs (HeteroMLP).
# MCMC calibrates 4 params:
#   E_intercept, alpha_base, alpha_slope, nu
# Remaining 4 params:
#   E_slope, SS316_T_ref, SS316_k_ref, SS316_alpha
# are either fixed at training-set mean (global prior) or retained
# from the case truth (conditional prior baseline).
#
# Outputs (written to OUT_DIR, tagged by RUN_TAG):
#   paper_posterior_hf_validation_summary_<RUN_TAG>.csv
#   paper_posterior_hf_validation_per_output_<RUN_TAG>.csv
#   paper_posterior_hf_validation_meta_<RUN_TAG>.json
# ============================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

from paper_experiment_config import (
    CSV_PATH,
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    FIXED_CKPT_PATH,
    FIXED_SCALER_PATH,
    FIXED_SPLIT_DIR,
    SEED,
)
from run_phys_levels_main import (
    load_dataset,
    get_device,
    _to_numpy,
)

# ============================================================
# Settings
# ============================================================

RUN_TAG = "reduced_maintext"

CALIBRATION_INPUT_COLS = ["E_intercept", "alpha_base", "alpha_slope", "nu"]
FIXED_INPUT_COLS = [c for c in INPUT_COLS if c not in CALIBRATION_INPUT_COLS]

N_NEIGHBORS = 3
N_CASES = 20

# Use only train+val HF library for neighbor search, not test set.
USE_TRAINVAL_HF_LIBRARY_ONLY = True

BENCHMARK_DIR = os.path.join(OUT_DIR, "benchmark_case")

SUMMARY_CSV = os.path.join(
    OUT_DIR, f"paper_posterior_hf_validation_summary_{RUN_TAG}.csv"
)
PER_OUTPUT_CSV = os.path.join(
    OUT_DIR, f"paper_posterior_hf_validation_per_output_{RUN_TAG}.csv"
)
META_JSON = os.path.join(
    OUT_DIR, f"paper_posterior_hf_validation_meta_{RUN_TAG}.json"
)


# ============================================================
# Helpers
# ============================================================

def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def _safe_read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# Consistency checks
# ============================================================

def check_split_consistency(df_full: pd.DataFrame):
    split_meta_path = os.path.join(FIXED_SPLIT_DIR, "split_meta.json")
    train_idx_path = os.path.join(FIXED_SPLIT_DIR, "train_indices.csv")
    val_idx_path   = os.path.join(FIXED_SPLIT_DIR, "val_indices.csv")
    test_idx_path  = os.path.join(FIXED_SPLIT_DIR, "test_indices.csv")

    check_required_file(split_meta_path, "split_meta.json")
    check_required_file(train_idx_path, "train_indices.csv")
    check_required_file(val_idx_path,   "val_indices.csv")
    check_required_file(test_idx_path,  "test_indices.csv")

    split_meta = _safe_read_json(split_meta_path)

    idx_train = pd.read_csv(train_idx_path)["index"].to_numpy(dtype=int)
    idx_val   = pd.read_csv(val_idx_path)["index"].to_numpy(dtype=int)
    idx_test  = pd.read_csv(test_idx_path)["index"].to_numpy(dtype=int)

    # strict consistency checks
    n_total_meta = int(split_meta.get("n_total", -1))
    if n_total_meta != len(df_full):
        raise ValueError(
            f"Split/data mismatch: split_meta n_total={n_total_meta}, "
            f"but current dataset has len(df_full)={len(df_full)}.\n"
            f"CSV_PATH={CSV_PATH}\nFIXED_SPLIT_DIR={FIXED_SPLIT_DIR}"
        )

    csv_meta = split_meta.get("csv_path", None)
    if csv_meta is not None:
        if os.path.basename(_norm_path(csv_meta)) != os.path.basename(_norm_path(CSV_PATH)):
            raise ValueError(
                "Split/data mismatch: split_meta csv_path basename differs from current CSV_PATH.\n"
                f"split_meta csv_path = {csv_meta}\n"
                f"current CSV_PATH    = {CSV_PATH}"
            )

    all_idx = np.concatenate([idx_train, idx_val, idx_test])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Frozen split indices contain duplicates across train/val/test.")
    if all_idx.min() < 0 or all_idx.max() >= len(df_full):
        raise ValueError("Frozen split indices are out of bounds for current dataset.")

    return split_meta, idx_train, idx_val, idx_test


# ============================================================
# Load frozen surrogate
# ============================================================

def load_surrogate(device):
    from run_phys_levels_main import HeteroMLP

    check_required_file(FIXED_CKPT_PATH, "Fixed checkpoint")
    check_required_file(FIXED_SCALER_PATH, "Fixed scalers")

    ckpt = torch.load(FIXED_CKPT_PATH, map_location=device)
    bp = ckpt["best_params"]

    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(bp["width"]),
        depth=int(bp["depth"]),
        dropout=float(bp["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(FIXED_SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers["sx"], scalers["sy"], ckpt


# ============================================================
# Surrogate forward
# ============================================================

def surrogate_predict(x_raw, model, sx, sy, device):
    """
    x_raw: (N, 8) numpy
    returns:
      mu    : (N, 15)
      sigma : (N, 15)
    """
    x_s = sx.transform(x_raw)
    x_t = torch.tensor(x_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_s, logvar_s = model(x_t)
        var_s = torch.exp(logvar_s)
    mu = sy.inverse_transform(_to_numpy(mu_s))
    sigma = np.sqrt(_to_numpy(var_s)) * sy.scale_
    return mu, sigma


# ============================================================
# Parameter vectors
# ============================================================

def build_global_prior_mean_vector(sx):
    """8D global prior center = training-set mean for all inputs."""
    return sx.mean_.copy()


def build_conditional_prior_vector(x_true, global_prior_mean):
    """
    More informative baseline:
      - calibrated 4 params set to prior mean
      - non-calibrated 4 params kept at case truth
    """
    x = x_true.copy()
    for c in CALIBRATION_INPUT_COLS:
        j = INPUT_COLS.index(c)
        x[j] = global_prior_mean[j]
    return x


def build_posterior_mean_vector(case_i, global_prior_mean):
    """
    Posterior mean for the 4 calibrated params.
    Remaining 4 params fixed at global training mean.
    """
    post_file = os.path.join(
        BENCHMARK_DIR,
        f"benchmark_case{case_i:03d}_posterior_samples_{RUN_TAG}.csv"
    )
    if not os.path.exists(post_file):
        return None, post_file

    post_df = pd.read_csv(post_file)

    theta = global_prior_mean.copy()
    for col in CALIBRATION_INPUT_COLS:
        if col not in post_df.columns:
            raise ValueError(
                f"Posterior file missing required column '{col}': {post_file}"
            )
        theta[INPUT_COLS.index(col)] = float(post_df[col].mean())
    return theta, post_file


# ============================================================
# Neighbor search
# ============================================================

def find_nearest_hf_neighbors(theta_full, X_pool, sx, k=3):
    """
    Search nearest HF rows in standardised 8D input space.
    Returns:
      local_indices_in_pool, distances
    """
    X_s = sx.transform(X_pool)
    q_s = sx.transform(theta_full.reshape(1, -1))[0]
    dists = np.linalg.norm(X_s - q_s, axis=1)
    order = np.argsort(dists)
    return order[:k], dists[order[:k]]


# ============================================================
# Case loading
# ============================================================

def load_case_true_x_y(case_i, df_full, idx_test):
    """
    Strict mapping:
      benchmark case_i  <->  frozen test set position case_i
    """
    if case_i >= len(idx_test):
        raise IndexError(
            f"Case index {case_i} out of range for frozen test set of length {len(idx_test)}."
        )
    global_idx = int(idx_test[case_i])
    row = df_full.iloc[global_idx]
    x_true = row[INPUT_COLS].to_numpy(dtype=float)
    y_true = row[OUTPUT_COLS].to_numpy(dtype=float)
    return x_true, y_true, global_idx


# ============================================================
# Error metrics
# ============================================================

def prediction_error(y_pred, y_true):
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / (np.abs(y_true) + 1e-30)
    return abs_err, rel_err


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(SEED)
    device = get_device()

    # --- Load dataset ---
    df_full = load_dataset()
    X_full = df_full[INPUT_COLS].to_numpy(dtype=float)
    Y_full = df_full[OUTPUT_COLS].to_numpy(dtype=float)

    # --- Frozen split consistency ---
    split_meta, idx_train, idx_val, idx_test = check_split_consistency(df_full)

    # --- Load surrogate ---
    model, sx, sy, ckpt = load_surrogate(device)
    global_prior_mean = build_global_prior_mean_vector(sx)

    # --- Candidate HF library for neighbor search ---
    if USE_TRAINVAL_HF_LIBRARY_ONLY:
        hf_pool_indices = np.concatenate([idx_train, idx_val])
        hf_pool_name = "train+val"
    else:
        hf_pool_indices = np.arange(len(df_full))
        hf_pool_name = "all"

    X_pool = X_full[hf_pool_indices]
    Y_pool = Y_full[hf_pool_indices]

    print(f"[INFO] Dataset            : n={len(df_full)}")
    print(f"[INFO] Frozen split       : train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print(f"[INFO] Surrogate ckpt     : {FIXED_CKPT_PATH}")
    print(f"[INFO] Scalers            : {FIXED_SCALER_PATH}")
    print(f"[INFO] Benchmark dir      : {BENCHMARK_DIR}")
    print(f"[INFO] HF neighbor pool   : {hf_pool_name}  (n={len(hf_pool_indices)})")
    print(f"[INFO] Calibration params : {CALIBRATION_INPUT_COLS}")
    print(f"[INFO] Fixed params       : {FIXED_INPUT_COLS}")
    print(f"[INFO] Cases to process   : {N_CASES}  (RUN_TAG={RUN_TAG})\n")

    # --- Output indices ---
    stress_idx = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    keff_idx = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)
    pri_idx = [OUTPUT_COLS.index(c) for c in PRIMARY_OUTPUTS]

    # --- Global prior prediction: same for all cases ---
    mu_global_prior, sigma_global_prior = surrogate_predict(
        global_prior_mean.reshape(1, -1), model, sx, sy, device
    )
    mu_global_prior = mu_global_prior[0]
    sigma_global_prior = sigma_global_prior[0]

    summary_rows = []
    per_output_rows = []
    n_skipped = 0
    posterior_files_used = []

    for case_i in range(N_CASES):
        # 1) ground-truth HF test case
        x_true, y_true, global_idx = load_case_true_x_y(case_i, df_full, idx_test)

        # 2) conditional prior baseline
        theta_conditional_prior = build_conditional_prior_vector(
            x_true, global_prior_mean
        )
        mu_cond_prior, sigma_cond_prior = surrogate_predict(
            theta_conditional_prior.reshape(1, -1), model, sx, sy, device
        )
        mu_cond_prior = mu_cond_prior[0]
        sigma_cond_prior = sigma_cond_prior[0]

        # 3) posterior mean predictor
        theta_post, post_file = build_posterior_mean_vector(case_i, global_prior_mean)
        if theta_post is None:
            print(f"  [SKIP] Case {case_i:02d}: missing posterior file -> {post_file}")
            n_skipped += 1
            continue
        posterior_files_used.append(post_file)

        mu_post, sigma_post = surrogate_predict(
            theta_post.reshape(1, -1), model, sx, sy, device
        )
        mu_post = mu_post[0]
        sigma_post = sigma_post[0]

        # 4) nearest HF proxy neighbor
        nbr_local_idx, nbr_dists = find_nearest_hf_neighbors(
            theta_post, X_pool, sx, k=N_NEIGHBORS
        )
        nbr_global_idx = int(hf_pool_indices[nbr_local_idx[0]])
        nbr_x = X_pool[nbr_local_idx[0]]
        nbr_y = Y_pool[nbr_local_idx[0]]

        # 5) errors
        err_global_abs, _ = prediction_error(mu_global_prior, y_true)
        err_cond_abs, _   = prediction_error(mu_cond_prior, y_true)
        err_post_abs, _   = prediction_error(mu_post, y_true)
        err_nbr_abs, _    = prediction_error(nbr_y, y_true)

        pri_mae_global = float(np.mean([err_global_abs[i] for i in pri_idx]))
        pri_mae_cond   = float(np.mean([err_cond_abs[i]   for i in pri_idx]))
        pri_mae_post   = float(np.mean([err_post_abs[i]   for i in pri_idx]))
        pri_mae_nbr    = float(np.mean([err_nbr_abs[i]    for i in pri_idx]))

        improvement_vs_global = pri_mae_global - pri_mae_post
        improvement_vs_cond   = pri_mae_cond - pri_mae_post

        # 6) summary row
        summary_rows.append({
            "case_i": case_i,
            "global_idx": global_idx,

            "true_stress_MPa": float(y_true[stress_idx]),
            "true_keff": float(y_true[keff_idx]),

            # posterior mean params
            **{
                f"post_mean_{c}": float(theta_post[INPUT_COLS.index(c)])
                for c in CALIBRATION_INPUT_COLS
            },

            # primary MAE comparisons
            "pri_mae_global_prior": pri_mae_global,
            "pri_mae_conditional_prior": pri_mae_cond,
            "pri_mae_post": pri_mae_post,
            "pri_mae_nbr": pri_mae_nbr,

            "improvement_vs_global_prior": improvement_vs_global,
            "improvement_vs_conditional_prior": improvement_vs_cond,

            # stress-specific
            "stress_err_global_prior": float(err_global_abs[stress_idx]),
            "stress_err_conditional_prior": float(err_cond_abs[stress_idx]),
            "stress_err_post": float(err_post_abs[stress_idx]),
            "stress_err_nbr": float(err_nbr_abs[stress_idx]),

            "stress_pred_global_prior": float(mu_global_prior[stress_idx]),
            "stress_pred_conditional_prior": float(mu_cond_prior[stress_idx]),
            "stress_pred_post": float(mu_post[stress_idx]),
            "stress_nbr_true": float(nbr_y[stress_idx]),

            # neighbor info
            "nbr_global_idx": nbr_global_idx,
            "nbr_dist_scaled": float(nbr_dists[0]),
        })

        # 7) per-output rows (primary outputs only)
        for c in PRIMARY_OUTPUTS:
            j = OUTPUT_COLS.index(c)
            per_output_rows.append({
                "case_i": case_i,
                "output": c,
                "true_val": float(y_true[j]),

                "pred_global_prior": float(mu_global_prior[j]),
                "pred_conditional_prior": float(mu_cond_prior[j]),
                "pred_post": float(mu_post[j]),
                "pred_nbr": float(nbr_y[j]),

                "err_global_prior": float(err_global_abs[j]),
                "err_conditional_prior": float(err_cond_abs[j]),
                "err_post": float(err_post_abs[j]),
                "err_nbr": float(err_nbr_abs[j]),

                "sigma_global_prior": float(sigma_global_prior[j]),
                "sigma_conditional_prior": float(sigma_cond_prior[j]),
                "sigma_post": float(sigma_post[j]),
            })

        print(
            f"  Case {case_i:02d}: "
            f"stress={y_true[stress_idx]:.1f} MPa  "
            f"global={pri_mae_global:.3f}  "
            f"cond={pri_mae_cond:.3f}  "
            f"post={pri_mae_post:.3f}  "
            f"nbr_dist={nbr_dists[0]:.3f}  "
            f"improve(global)={improvement_vs_global > 0}  "
            f"improve(cond)={improvement_vs_cond > 0}"
        )

    if not summary_rows:
        raise RuntimeError(
            f"No cases processed. Check posterior files in {BENCHMARK_DIR} "
            f"for RUN_TAG={RUN_TAG}."
        )

    # ---- Save outputs ----
    df_summary = pd.DataFrame(summary_rows)
    df_per_out = pd.DataFrame(per_output_rows)

    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    df_per_out.to_csv(PER_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ---- Aggregate stats ----
    n_processed = len(df_summary)
    n_improved_global = int((df_summary["improvement_vs_global_prior"] > 0).sum())
    n_improved_cond = int((df_summary["improvement_vs_conditional_prior"] > 0).sum())

    meta = {
        "scientific_note": (
            "This is nearest-neighbor HF proxy validation using the existing HF dataset, "
            "not an actual high-fidelity rerun at the posterior mean."
        ),
        "run_tag": RUN_TAG,
        "n_cases_attempted": N_CASES,
        "n_cases_processed": n_processed,
        "n_cases_skipped": n_skipped,

        "calibration_params": CALIBRATION_INPUT_COLS,
        "fixed_params": FIXED_INPUT_COLS,

        "neighbor_pool": hf_pool_name,
        "use_trainval_hf_library_only": USE_TRAINVAL_HF_LIBRARY_ONLY,
        "n_neighbors": N_NEIGHBORS,

        "n_improved_vs_global_prior": n_improved_global,
        "pct_improved_vs_global_prior": float(n_improved_global / n_processed * 100.0),
        "n_improved_vs_conditional_prior": n_improved_cond,
        "pct_improved_vs_conditional_prior": float(n_improved_cond / n_processed * 100.0),

        "mean_pri_mae_global_prior": float(df_summary["pri_mae_global_prior"].mean()),
        "mean_pri_mae_conditional_prior": float(df_summary["pri_mae_conditional_prior"].mean()),
        "mean_pri_mae_post": float(df_summary["pri_mae_post"].mean()),
        "mean_pri_mae_nbr": float(df_summary["pri_mae_nbr"].mean()),

        "mean_improvement_vs_global_prior": float(df_summary["improvement_vs_global_prior"].mean()),
        "mean_improvement_vs_conditional_prior": float(df_summary["improvement_vs_conditional_prior"].mean()),

        "mean_stress_err_global_prior": float(df_summary["stress_err_global_prior"].mean()),
        "mean_stress_err_conditional_prior": float(df_summary["stress_err_conditional_prior"].mean()),
        "mean_stress_err_post": float(df_summary["stress_err_post"].mean()),
        "mean_stress_err_nbr": float(df_summary["stress_err_nbr"].mean()),

        "mean_nbr_dist_scaled": float(df_summary["nbr_dist_scaled"].mean()),

        # provenance
        "csv_path": _norm_path(CSV_PATH),
        "fixed_ckpt_path": _norm_path(FIXED_CKPT_PATH),
        "fixed_scaler_path": _norm_path(FIXED_SCALER_PATH),
        "fixed_split_dir": _norm_path(FIXED_SPLIT_DIR),
        "benchmark_dir": _norm_path(BENCHMARK_DIR),
        "summary_csv": _norm_path(SUMMARY_CSV),
        "per_output_csv": _norm_path(PER_OUTPUT_CSV),
        "posterior_files_used": posterior_files_used,
        "split_meta": split_meta,
        "checkpoint_level": ckpt.get("level", None),
        "checkpoint_tag": ckpt.get("tag", None),
    }

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- Print summary ----
    print(f"\n{'=' * 72}")
    print("  POSTERIOR HF PROXY VALIDATION COMPLETE")
    print(f"  Cases processed                 : {n_processed} / {N_CASES}")
    print(f"  Improved vs global prior        : {n_improved_global}/{n_processed} "
          f"({meta['pct_improved_vs_global_prior']:.0f}%)")
    print(f"  Improved vs conditional prior   : {n_improved_cond}/{n_processed} "
          f"({meta['pct_improved_vs_conditional_prior']:.0f}%)")
    print(f"  Mean primary MAE")
    print(f"    global prior      : {meta['mean_pri_mae_global_prior']:.3f}")
    print(f"    conditional prior : {meta['mean_pri_mae_conditional_prior']:.3f}")
    print(f"    posterior mean    : {meta['mean_pri_mae_post']:.3f}")
    print(f"    HF neighbor proxy : {meta['mean_pri_mae_nbr']:.3f}")
    print(f"  Mean stress MAE")
    print(f"    global prior      : {meta['mean_stress_err_global_prior']:.2f} MPa")
    print(f"    conditional prior : {meta['mean_stress_err_conditional_prior']:.2f} MPa")
    print(f"    posterior mean    : {meta['mean_stress_err_post']:.2f} MPa")
    print(f"    HF neighbor proxy : {meta['mean_stress_err_nbr']:.2f} MPa")
    print(f"  Mean neighbor dist (std-space)  : {meta['mean_nbr_dist_scaled']:.3f}")
    print(f"{'=' * 72}")
    print(f"Saved:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {PER_OUTPUT_CSV}")
    print(f"  {META_JSON}")


if __name__ == "__main__":
    main()