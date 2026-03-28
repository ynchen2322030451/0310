# run_sobol_analysis.py
# ============================================================
# Sobol analysis with repeated estimation + CI
#
# Main features:
#   1) level 2 is forced to use fixed surrogate artifacts
#   2) level 0 uses legacy baseline artifacts
#   3) supports iter1 + iter2 outputs
#   4) exports CI-ready table for paper
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
    PAPER_LEVELS,
    SEED,
    FIXED_CKPT_PATH,
    FIXED_SCALER_PATH,
)
from run_phys_levels_main import HeteroMLP, get_device


# ============================================================
# User settings
# ============================================================

SOBOL_OUTPUTS = [
    "iteration1_avg_fuel_temp",
    "iteration1_max_fuel_temp",
    "iteration1_max_monolith_temp",
    "iteration1_max_global_stress",
    "iteration1_wall2",
    # iteration1_keff 可导出，但不建议主解释
    "iteration1_keff",

    "iteration2_keff",
    "iteration2_avg_fuel_temp",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_max_global_stress",
    "iteration2_wall2",
]

LEVELS_TO_RUN = [lv for lv in PAPER_LEVELS if lv in [0, 2]]

N_BASE = 512*32
N_REPEATS = 50
CI_Z = 1.96

META_STATS_CANDIDATES = [
    os.path.join(OUT_DIR, "fixed_surrogate_fixed_level2", "meta_stats.json"),
    os.path.join(OUT_DIR, "meta_stats.json"),
]


# ============================================================
# Helpers
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_existing_meta_stats():
    for p in META_STATS_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find meta_stats.json in any candidate path:\n" + "\n".join(META_STATS_CANDIDATES)
    )


def load_input_bounds():
    meta_path = find_existing_meta_stats()
    meta = load_json(meta_path)
    if "input_stats" not in meta:
        raise ValueError(f"Missing 'input_stats' in {meta_path}")

    bounds = []
    for c in INPUT_COLS:
        st = meta["input_stats"][c]
        lo = float(st["min"])
        hi = float(st["max"])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid bound for {c}: ({lo}, {hi})")
        bounds.append((lo, hi))
    return bounds


def get_artifact_dir(level: int) -> str:
    mapping = {
        0: os.path.join(OUT_DIR, "fixed_surrogate_fixed_base"),
        2: os.path.join(OUT_DIR, "fixed_surrogate_fixed_level2"),
    }
    if level not in mapping:
        raise ValueError(f"Unsupported level for fixed surrogate: {level}")
    return mapping[level]


def load_checkpoint_and_scalers(level: int):
    art_dir = get_artifact_dir(level)
    ckpt_path = os.path.join(art_dir, f"checkpoint_level{level}.pt")
    scaler_path = os.path.join(art_dir, f"scalers_level{level}.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

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

    s1_all = np.asarray(s1_all, dtype=float)
    st_all = np.asarray(st_all, dtype=float)
    return s1_all, st_all


def summarize_repeated_indices(arr):
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

    for level in LEVELS_TO_RUN:
        print(f"\n[INFO] Sobol repeated estimation for level {level}")
        ckpt, sx, sy = load_checkpoint_and_scalers(level)
        model = build_model_from_ckpt(ckpt, device)

        for out_name in SOBOL_OUTPUTS:
            if out_name not in OUTPUT_COLS:
                print(f"[WARN] Output not found in OUTPUT_COLS: {out_name}")
                continue

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
                    "S1_plot": float(max(0.0, s1_mean[j])),

                    "ST_mean": float(st_mean[j]),
                    "ST_std": float(st_std[j]),
                    "ST_ci_low": float(st_lo[j]),
                    "ST_ci_high": float(st_hi[j]),
                })

    df = pd.DataFrame(rows)

    out_csv_all = os.path.join(OUT_DIR, "paper_sobol_results_with_ci_all_iters.csv")
    df.to_csv(out_csv_all, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {out_csv_all}")

    # backward-compatible simplified export
    df_simple = df.rename(columns={"S1_plot": "S1", "ST_mean": "ST"})[
        ["output", "input", "S1", "ST", "level"]
    ]
    out_csv_simple = os.path.join(OUT_DIR, "paper_sobol_results.csv")
    df_simple.to_csv(out_csv_simple, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {out_csv_simple}")


if __name__ == "__main__":
    main()