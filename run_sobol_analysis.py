# run_sobol_analysis.py
# ============================================================
# Sobol analysis with repeated estimation + CI for paper
#
# Exports:
#   1) paper_sobol_results_with_ci.csv
#      -> main-text focused (iter2 stress + iter2 keff)
#   2) paper_sobol_results_with_ci_all_iters.csv
#      -> iter1 vs iter2 comparison file
#   3) paper_sobol_results.csv
#      -> simplified backward-compatible main-text file
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
# 主文仍然只关注 iter2 stress + iter2 keff
MAIN_TEXT_OUTPUTS = [
    PRIMARY_STRESS_OUTPUT,      # iteration2_max_global_stress
    PRIMARY_AUXILIARY_OUTPUT,   # iteration2_keff
]

# 新增：用于 iter1 vs iter2 对比的输出集合
COMPARE_OUTPUTS = [
    "iteration1_keff",
    "iteration2_keff",
    "iteration1_avg_fuel_temp",
    "iteration2_avg_fuel_temp",
    "iteration1_max_fuel_temp",
    "iteration2_max_fuel_temp",
    "iteration1_max_monolith_temp",
    "iteration2_max_monolith_temp",
    "iteration1_max_global_stress",
    "iteration2_max_global_stress",
    "iteration1_wall2",
    "iteration2_wall2",
]

N_BASE = 512
N_REPEATS = 20
CI_Z = 1.96

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


@torch.no_grad()
def predict_mu(model, sx, sy, x_raw, device):
    xs = sx.transform(x_raw)
    xt = torch.tensor(xs, dtype=torch.float32, device=device)
    mu_s, _ = model(xt)
    mu_s = mu_s.detach().cpu().numpy()
    mu = sy.inverse_transform(mu_s)
    return mu


def sample_A_B(bounds, n_base, seed):
    rng = np.random.default_rng(seed)
    d = len(bounds)

    A = np.zeros((n_base, d), dtype=float)
    B = np.zeros((n_base, d), dtype=float)

    for j, (lo, hi) in enumerate(bounds):
        A[:, j] = rng.uniform(lo, hi, size=n_base)
        B[:, j] = rng.uniform(lo, hi, size=n_base)

    return A, B


def build_ABj(A, B, j):
    ABj = A.copy()
    ABj[:, j] = B[:, j]
    return ABj


def sobol_jansen_indices_from_evals(fA, fB, fAB):
    """
    Jansen estimators
    fA, fB: (N,)
    fAB: list of length d, each item shape (N,)
    """
    N = len(fA)
    d = len(fAB)

    var_y = np.var(np.concatenate([fA, fB]), ddof=1)
    var_y = max(var_y, 1e-16)

    ST = np.zeros(d)
    S1 = np.zeros(d)

    for j in range(d):
        ST[j] = 0.5 * np.mean((fA - fAB[j]) ** 2) / var_y
        S1[j] = 1.0 - 0.5 * np.mean((fB - fAB[j]) ** 2) / var_y

    return S1, ST


def repeated_sobol_for_output(model, sx, sy, out_idx, bounds, device, base_seed):
    d = len(bounds)
    s1_rep = []
    st_rep = []

    for r in range(N_REPEATS):
        seed = base_seed + r
        A, B = sample_A_B(bounds, N_BASE, seed)

        muA = predict_mu(model, sx, sy, A, device)[:, out_idx]
        muB = predict_mu(model, sx, sy, B, device)[:, out_idx]

        fAB = []
        for j in range(d):
            ABj = build_ABj(A, B, j)
            muABj = predict_mu(model, sx, sy, ABj, device)[:, out_idx]
            fAB.append(muABj)

        s1, st = sobol_jansen_indices_from_evals(muA, muB, fAB)
        s1_rep.append(s1)
        st_rep.append(st)

    return np.asarray(s1_rep), np.asarray(st_rep)


def summarize_repeated_indices(arr):
    """
    arr shape: (n_repeat, d)
    """
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    lo = mean - CI_Z * std / np.sqrt(arr.shape[0])
    hi = mean + CI_Z * std / np.sqrt(arr.shape[0])
    return mean, std, lo, hi


def compute_rows_for_outputs(model, sx, sy, bounds, device, level, outputs_to_run):
    rows = []

    for out_name in outputs_to_run:
        out_idx = OUTPUT_COLS.index(out_name)

        s1_rep, st_rep = repeated_sobol_for_output(
            model=model,
            sx=sx,
            sy=sy,
            out_idx=out_idx,
            bounds=bounds,
            device=device,
            base_seed=SEED + 100 * level + 10000 * out_idx,
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

                "S1_plot": float(max(0.0, s1_mean[j])),

                "ST_mean": float(st_mean[j]),
                "ST_std": float(st_std[j]),
                "ST_ci_low": float(st_lo[j]),
                "ST_ci_high": float(st_hi[j]),
            })

    return rows


def main():
    device = get_device()
    bounds = load_input_bounds()

    # freeze main-text levels to 0 vs 2
    levels_for_sa = [lv for lv in PAPER_LEVELS if lv in [0, 2]]

    rows_main = []
    rows_all_iters = []

    for level in levels_for_sa:
        print(f"\n[INFO] Sobol repeated estimation for level {level}")
        ckpt, sx, sy = load_checkpoint_and_scalers(level)
        model = build_model_from_ckpt(ckpt, device)

        # 1) 主文 iter2-only
        rows_main.extend(
            compute_rows_for_outputs(
                model=model,
                sx=sx,
                sy=sy,
                bounds=bounds,
                device=device,
                level=level,
                outputs_to_run=MAIN_TEXT_OUTPUTS,
            )
        )

        # 2) iter1/iter2 对比版
        rows_all_iters.extend(
            compute_rows_for_outputs(
                model=model,
                sx=sx,
                sy=sy,
                bounds=bounds,
                device=device,
                level=level,
                outputs_to_run=COMPARE_OUTPUTS,
            )
        )

    # 主文文件：保持兼容
    df_main = pd.DataFrame(rows_main)
    out_csv_main = os.path.join(OUT_DIR, "paper_sobol_results_with_ci.csv")
    df_main.to_csv(out_csv_main, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved repeated Sobol with CI (main-text): {out_csv_main}")

    df_simple = df_main.rename(columns={"S1_plot": "S1", "ST_mean": "ST"})[
        ["output", "input", "S1", "ST", "level"]
    ]
    out_csv_simple = os.path.join(OUT_DIR, "paper_sobol_results.csv")
    df_simple.to_csv(out_csv_simple, index=False, encoding="utf-8-sig")
    print(f"[DONE] Updated simplified Sobol file: {out_csv_simple}")

    # 新文件：iter1/iter2 对比专用
    df_all = pd.DataFrame(rows_all_iters)
    out_csv_all = os.path.join(OUT_DIR, "paper_sobol_results_with_ci_all_iters.csv")
    df_all.to_csv(out_csv_all, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved repeated Sobol with CI (all iters): {out_csv_all}")


if __name__ == "__main__":
    main()