# run_sobol_analysis.py
# ============================================================
# Sobol analysis with repeated estimation + CI for paper
# Main-text focus:
#   - Level0 vs Level2
#   - iteration2_max_global_stress
#   - iteration2_keff
# ============================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    PAPER_LEVELS,
    SEED,
)
from run_phys_levels_main import HeteroMLP, get_device

# -----------------------------
# Sobol settings
# -----------------------------
SOBOL_OUTPUTS = [
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,   # keff
]
N_BASE = 512
N_REPEATS = 20
CI_Z = 1.96

# Parameter sampling range:
# use training/statistical support from dataset meta
META_STATS_PATH = os.path.join(OUT_DIR, "meta_stats.json")


def load_checkpoint_and_scalers(level: int):
    ckpt_path = os.path.join(OUT_DIR, f"checkpoint_level{level}.pt")
    scaler_path = os.path.join(OUT_DIR, f"scalers_level{level}.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scalers: {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return ckpt, scalers["sx"], scalers["sy"]


def build_model_from_ckpt(ckpt, device):
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
    return model


def load_input_bounds():
    if not os.path.exists(META_STATS_PATH):
        raise FileNotFoundError(f"Missing meta stats: {META_STATS_PATH}")
    with open(META_STATS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    st = meta["input_stats"]

    bounds = []
    for c in INPUT_COLS:
        lo = float(st[c]["min"])
        hi = float(st[c]["max"])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid bound for {c}: ({lo}, {hi})")
        bounds.append((lo, hi))
    return bounds


def sample_sobol_matrices(n, bounds, rng):
    d = len(bounds)
    A = np.zeros((n, d), dtype=float)
    B = np.zeros((n, d), dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        A[:, j] = rng.uniform(lo, hi, size=n)
        B[:, j] = rng.uniform(lo, hi, size=n)
    return A, B


@torch.no_grad()
def predict_mu_original(model, sx, sy, x_np, device):
    xs = sx.transform(x_np)
    xt = torch.tensor(xs, dtype=torch.float32, device=device)
    mu_s, _ = model(xt)
    mu = sy.inverse_transform(mu_s.detach().cpu().numpy())
    return mu


def jansen_indices_from_predictions(YA, YB, YABi):
    """
    YA, YB, YABi: shape [N]
    Jansen estimators:
      ST_i = mean((YA - YABi)^2) / (2 Var(Y))
      S1_i = 1 - mean((YB - YABi)^2) / (2 Var(Y))
    """
    VY = np.var(np.concatenate([YA, YB]), ddof=1)
    if VY <= 1e-15:
        return 0.0, 0.0

    ST = np.mean((YA - YABi) ** 2) / (2.0 * VY)
    S1 = 1.0 - np.mean((YB - YABi) ** 2) / (2.0 * VY)
    return float(S1), float(ST)


def repeated_sobol_for_output(model, sx, sy, out_idx, bounds, device, base_seed):
    s1_all = []
    st_all = []

    d = len(bounds)

    for r in range(N_REPEATS):
        rng = np.random.RandomState(base_seed + 1000 * r + out_idx)

        A, B = sample_sobol_matrices(N_BASE, bounds, rng)

        YA = predict_mu_original(model, sx, sy, A, device)[:, out_idx]
        YB = predict_mu_original(model, sx, sy, B, device)[:, out_idx]

        s1_r = []
        st_r = []

        for j in range(d):
            ABj = A.copy()
            ABj[:, j] = B[:, j]
            YABj = predict_mu_original(model, sx, sy, ABj, device)[:, out_idx]

            S1, ST = jansen_indices_from_predictions(YA, YB, YABj)
            s1_r.append(S1)
            st_r.append(ST)

        s1_all.append(s1_r)
        st_all.append(st_r)

    s1_all = np.asarray(s1_all, dtype=float)   # [R, D]
    st_all = np.asarray(st_all, dtype=float)   # [R, D]
    return s1_all, st_all


def summarize_repeated_indices(arr):
    """
    arr: [R, D]
    returns mean/std/ci_low/ci_high
    """
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    ci_half = CI_Z * std / np.sqrt(max(arr.shape[0], 1))
    lo = mean - ci_half
    hi = mean + ci_half
    return mean, std, lo, hi


def main():
    device = get_device()
    bounds = load_input_bounds()

    rows = []

    # freeze main-text levels to 0 vs 2 even if PAPER_LEVELS changed elsewhere
    levels_for_sa = [lv for lv in PAPER_LEVELS if lv in [0, 2]]

    for level in levels_for_sa:
        print(f"\n[INFO] Sobol repeated estimation for level {level}")
        ckpt, sx, sy = load_checkpoint_and_scalers(level)
        model = build_model_from_ckpt(ckpt, device)

        for out_name in SOBOL_OUTPUTS:
            out_idx = OUTPUT_COLS.index(out_name)

            s1_rep, st_rep = repeated_sobol_for_output(
                model=model,
                sx=sx,
                sy=sy,
                out_idx=out_idx,
                bounds=bounds,
                device=device,
                base_seed=SEED + 100 * level,
            )

            s1_mean, s1_std, s1_lo, s1_hi = summarize_repeated_indices(s1_rep)
            st_mean, st_std, st_lo, st_hi = summarize_repeated_indices(st_rep)

            for j, inp in enumerate(INPUT_COLS):
                rows.append({
                    "output": out_name,
                    "input": inp,
                    "level": level,

                    "S1_raw_mean": float(s1_mean[j]),
                    "S1_raw_std": float(s1_std[j]),
                    "S1_ci_low": float(s1_lo[j]),
                    "S1_ci_high": float(s1_hi[j]),

                    # plotting-safe value
                    "S1_plot": float(max(0.0, s1_mean[j])),

                    "ST_mean": float(st_mean[j]),
                    "ST_std": float(st_std[j]),
                    "ST_ci_low": float(st_lo[j]),
                    "ST_ci_high": float(st_hi[j]),
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "paper_sobol_results_with_ci.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved repeated Sobol with CI: {out_csv}")

    # keep backward-compatible simplified export if needed
    df_simple = df.rename(columns={"S1_plot": "S1", "ST_mean": "ST"})[
        ["output", "input", "S1", "ST", "level"]
    ]
    out_csv_simple = os.path.join(OUT_DIR, "paper_sobol_results.csv")
    df_simple.to_csv(out_csv_simple, index=False, encoding="utf-8-sig")
    print(f"[DONE] Updated simplified Sobol file: {out_csv_simple}")


if __name__ == "__main__":
    main()