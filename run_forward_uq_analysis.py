# run_forward_uq_analysis.py
# ============================================================
# Forward UQ analysis for paper
# Final comparison: Level0 vs Level2
# Main outputs:
#   - iteration2_max_global_stress
#   - iteration2_keff
#   - iteration2_max_fuel_temp
#   - iteration2_max_monolith_temp
#   - iteration2_wall2
# ============================================================

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
import torch

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    THRESHOLD_SWEEP,
    SEED,
)
from run_phys_levels_main import HeteroMLP, get_device


# -----------------------------
# Settings
# -----------------------------
FORWARD_LEVELS = [0, 2]
N_SAMPLES = 20000
DRAW_PREDICTIVE_SAMPLES = True   # True: sample y ~ N(mu, sigma^2); False: use mu only


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_and_scalers(level: int):
    ckpt_path = os.path.join(OUT_DIR, f"checkpoint_level{level}.pt")
    scaler_path = os.path.join(OUT_DIR, f"scalers_level{level}.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}\n"
            f"Please run `python run_phys_levels_main.py` first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing scaler file: {scaler_path}\n"
            f"Please run `python run_phys_levels_main.py` first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return ckpt, scalers["sx"], scalers["sy"]


def build_model_from_ckpt(ckpt, device):
    best_params = ckpt["best_params"]
    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(best_params["width"]),
        depth=int(best_params["depth"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def sample_uniform_inputs_from_meta(n: int):
    meta = load_json(os.path.join(OUT_DIR, "meta_stats.json"))
    input_stats = meta["input_stats"]

    X = np.zeros((n, len(INPUT_COLS)), dtype=float)
    for i, c in enumerate(INPUT_COLS):
        lo = float(input_stats[c]["min"])
        hi = float(input_stats[c]["max"])
        X[:, i] = np.random.uniform(lo, hi, size=n)
    return X, input_stats


@torch.no_grad()
def predict_distribution(model, sx, sy, X_raw, device):
    Xs = sx.transform(X_raw)
    x = torch.tensor(Xs, dtype=torch.float32, device=device)
    mu_s, logvar_s = model(x)

    mu_s = mu_s.detach().cpu().numpy()
    logvar_s = logvar_s.detach().cpu().numpy()
    sigma_s = np.sqrt(np.exp(logvar_s))

    # inverse transform
    mu = sy.inverse_transform(mu_s)
    sigma = sigma_s * sy.scale_

    return mu, sigma


def compute_output_distribution_summary(samples, output_names):
    rows = []
    for j, name in enumerate(output_names):
        v = samples[:, j]
        rows.append({
            "output": name,
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


def compute_primary_distribution_summary(samples):
    rows = []
    for name in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(name)
        v = samples[:, j]
        rows.append({
            "output": name,
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "q05": float(np.quantile(v, 0.05)),
            "q25": float(np.quantile(v, 0.25)),
            "q50": float(np.quantile(v, 0.50)),
            "q75": float(np.quantile(v, 0.75)),
            "q95": float(np.quantile(v, 0.95)),
        })
    return rows


def compute_failure_probability(stress_samples, thresholds):
    rows = []
    for thr in thresholds:
        rows.append({
            "threshold_MPa": float(thr),
            "p_fail": float(np.mean(stress_samples > thr)),
        })
    return rows


def compute_joint_response_table(samples, n_keep=5000):
    idx_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    idx_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    n = min(n_keep, samples.shape[0])
    sel = np.random.choice(samples.shape[0], size=n, replace=False)
    sub = samples[sel]

    return pd.DataFrame({
        "iteration2_max_global_stress": sub[:, idx_stress],
        "iteration2_keff": sub[:, idx_keff],
    })


def compute_cvr(samples, input_stats):
    """
    CVR = mean output CV / mean input CV
    Using the same definition style as the CovidSim paper:
    mean over selected outputs / mean over uncertain inputs.
    """
    input_cvs = []
    for c in INPUT_COLS:
        mu = float(input_stats[c]["mean"])
        sd = float(input_stats[c]["std"])
        if abs(mu) < 1e-12:
            continue
        input_cvs.append(sd / abs(mu))
    mean_input_cv = float(np.mean(input_cvs))

    rows = []
    for name in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(name)
        v = samples[:, j]
        mu = float(np.mean(v))
        sd = float(np.std(v))
        out_cv = sd / abs(mu) if abs(mu) > 1e-12 else np.nan
        cvr = out_cv / mean_input_cv if mean_input_cv > 1e-12 else np.nan
        rows.append({
            "output": name,
            "output_mean": mu,
            "output_std": sd,
            "output_cv": float(out_cv),
            "mean_input_cv": mean_input_cv,
            "CVR": float(cvr),
        })

    # overall over primary outputs
    primary_cvs = [r["output_cv"] for r in rows if np.isfinite(r["output_cv"])]
    overall_cvr = float(np.mean(primary_cvs) / mean_input_cv) if mean_input_cv > 1e-12 else np.nan

    return rows, {
        "mean_input_cv": mean_input_cv,
        "mean_primary_output_cv": float(np.mean(primary_cvs)),
        "overall_primary_CVR": overall_cvr,
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_dir(OUT_DIR)
    device = get_device()

    # 1) sample uncertain inputs
    X_mc, input_stats = sample_uniform_inputs_from_meta(N_SAMPLES)

    # save sampled inputs once
    pd.DataFrame(X_mc, columns=INPUT_COLS).to_csv(
        os.path.join(OUT_DIR, "forward_uq_input_samples.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary_rows = []

    for level in FORWARD_LEVELS:
        ckpt, sx, sy = load_checkpoint_and_scalers(level)
        model = build_model_from_ckpt(ckpt, device)

        # 2) predictive distribution
        mu, sigma = predict_distribution(model, sx, sy, X_mc, device)

        # choose what to propagate downstream
        if DRAW_PREDICTIVE_SAMPLES:
            y_samples = np.random.normal(loc=mu, scale=np.maximum(sigma, 1e-12))
        else:
            y_samples = mu.copy()

        # 3) distribution summaries
        all_summary = compute_output_distribution_summary(y_samples, OUTPUT_COLS)
        primary_summary = compute_primary_distribution_summary(y_samples)

        pd.DataFrame(all_summary).to_csv(
            os.path.join(OUT_DIR, f"forward_uq_all_outputs_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame(primary_summary).to_csv(
            os.path.join(OUT_DIR, f"forward_uq_primary_outputs_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 4) failure probability under thresholds
        stress_idx = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
        stress_samples = y_samples[:, stress_idx]
        fail_rows = compute_failure_probability(stress_samples, THRESHOLD_SWEEP)

        pd.DataFrame(fail_rows).to_csv(
            os.path.join(OUT_DIR, f"forward_uq_failure_prob_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 5) joint response: stress vs keff
        joint_df = compute_joint_response_table(y_samples, n_keep=5000)
        joint_df.to_csv(
            os.path.join(OUT_DIR, f"forward_uq_joint_stress_keff_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 6) CVR uncertainty amplification
        cvr_rows, cvr_overall = compute_cvr(y_samples, input_stats)

        pd.DataFrame(cvr_rows).to_csv(
            os.path.join(OUT_DIR, f"forward_uq_cvr_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        save_json(
            cvr_overall,
            os.path.join(OUT_DIR, f"forward_uq_cvr_summary_level{level}.json")
        )

        # 7) save a compact npz for reuse
        np.savez_compressed(
            os.path.join(OUT_DIR, f"forward_uq_samples_level{level}.npz"),
            X=X_mc,
            mu=mu,
            sigma=sigma,
            y=y_samples,
        )

        # 8) global summary row for paper
        stress_q95 = float(np.quantile(stress_samples, 0.95))
        keff_idx = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)
        keff_samples = y_samples[:, keff_idx]

        fail_map = {r["threshold_MPa"]: r["p_fail"] for r in fail_rows}

        summary_rows.append({
            "level": level,
            "n_samples": int(N_SAMPLES),
            "draw_predictive_samples": bool(DRAW_PREDICTIVE_SAMPLES),

            "stress_mean": float(np.mean(stress_samples)),
            "stress_std": float(np.std(stress_samples)),
            "stress_q95": stress_q95,

            "keff_mean": float(np.mean(keff_samples)),
            "keff_std": float(np.std(keff_samples)),

            "p_fail_110": float(fail_map.get(110.0, np.nan)),
            "p_fail_120": float(fail_map.get(120.0, np.nan)),
            "p_fail_131": float(fail_map.get(131.0, np.nan)),

            "overall_primary_CVR": float(cvr_overall["overall_primary_CVR"]),
        })

        print(f"[OK] Forward UQ finished for level {level}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "paper_forward_uq_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("[DONE] Forward UQ analysis completed.")


if __name__ == "__main__":
    main()