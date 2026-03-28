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
    FIXED_CKPT_PATH,
    FIXED_SCALER_PATH,
)
from run_phys_levels_main import HeteroMLP, get_device


# ============================================================
# User settings
# ============================================================

RUN_TAG = "reduced"   # "reduced" or "full"
CASE_ID = 3

PARAM_X = "E_intercept"
PARAM_Y = "alpha_base"

THRESHOLDS = [131.0, 120.0]

# grid resolution for 2D map
NX = 120
NY = 120

# if reduced inverse
CALIBRATION_INPUT_COLS = ["E_intercept", "alpha_base", "alpha_slope", "SS316_k_ref"]

# use level2 surrogate
LEVEL = 2

OUT_PREFIX = os.path.join(OUT_DIR, f"paper_2d_case{CASE_ID:03d}_{RUN_TAG}")


# ============================================================
# Utilities
# ============================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


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

    return ckpt, scalers["sx"], scalers["sy"], ckpt_path, scaler_path


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


@torch.no_grad()
def predict_mu_sigma_raw(model, sx, sy, x_full_raw, device):
    xs = sx.transform(x_full_raw)
    xt = torch.tensor(xs, dtype=torch.float32, device=device)
    mu_s, logvar_s = model(xt)

    mu_s = mu_s.detach().cpu().numpy()
    logvar_s = logvar_s.detach().cpu().numpy()

    mu = sy.inverse_transform(mu_s)
    sigma = np.sqrt(np.exp(logvar_s)) * sy.scale_
    return mu, sigma


def expand_reduced_to_full(x_reduced: np.ndarray, reduced_cols, full_reference_row: np.ndarray, full_cols):
    out = np.tile(full_reference_row.reshape(1, -1), (x_reduced.shape[0], 1))
    for j, c in enumerate(reduced_cols):
        full_j = full_cols.index(c)
        out[:, full_j] = x_reduced[:, j]
    return out


def load_case_files(case_id: int, run_tag: str):
    suf = f"_{run_tag}"
    prior_path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_prior_samples{suf}.csv")
    post_path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_posterior_samples{suf}.csv")

    require_file(prior_path)
    require_file(post_path)

    df_prior = pd.read_csv(prior_path)
    df_post = pd.read_csv(post_path)
    return df_prior, df_post


def load_reference_row():
    """
    Use training mean from meta_stats? Here use midpoint from meta bounds as a stable fallback.
    Better: use calibration benchmark x_ref_full if available.
    """
    meta_path = os.path.join(OUT_DIR, "meta_stats.json")
    require_file(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    x_ref = []
    for c in INPUT_COLS:
        lo = float(meta["input_stats"][c]["min"])
        hi = float(meta["input_stats"][c]["max"])
        x_ref.append(0.5 * (lo + hi))
    return np.asarray(x_ref, dtype=float)


def build_2d_grid(df_ref: pd.DataFrame, xcol: str, ycol: str, nx: int, ny: int, pad_frac: float = 0.05):
    x = df_ref[xcol].to_numpy(dtype=float)
    y = df_ref[ycol].to_numpy(dtype=float)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    dx = (xmax - xmin) * pad_frac if xmax > xmin else 1.0
    dy = (ymax - ymin) * pad_frac if ymax > ymin else 1.0

    xv = np.linspace(xmin - dx, xmax + dx, nx)
    yv = np.linspace(ymin - dy, ymax + dy, ny)

    XX, YY = np.meshgrid(xv, yv)
    return xv, yv, XX, YY


def main():
    device = get_device()

    # 1) load surrogate
    ckpt, sx, sy = load_checkpoint_and_scalers(LEVEL)
    model = build_model_from_ckpt(ckpt, device)

    # 2) load prior/posterior samples for representative case
    df_prior, df_post = load_case_files(CASE_ID, RUN_TAG)

    if RUN_TAG == "reduced":
        reduced_cols = CALIBRATION_INPUT_COLS
    else:
        reduced_cols = INPUT_COLS

    if PARAM_X not in reduced_cols or PARAM_Y not in reduced_cols:
        raise ValueError(f"{PARAM_X} or {PARAM_Y} not found in calibration columns: {reduced_cols}")

    # 3) export 2D point clouds directly for plotting
    df_prior_2d = df_prior[[PARAM_X, PARAM_Y]].copy()
    df_post_2d = df_post[[PARAM_X, PARAM_Y]].copy()

    df_prior_2d.to_csv(f"{OUT_PREFIX}_prior_points.csv", index=False, encoding="utf-8-sig")
    df_post_2d.to_csv(f"{OUT_PREFIX}_posterior_points.csv", index=False, encoding="utf-8-sig")

    # 4) evaluate feasible posterior subsets
    x_ref_full = load_reference_row()

    stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")

    X_post_red = df_post[reduced_cols].to_numpy(dtype=float)
    if RUN_TAG == "reduced":
        X_post_full = expand_reduced_to_full(X_post_red, reduced_cols, x_ref_full, INPUT_COLS)
    else:
        X_post_full = X_post_red.copy()

    mu_post, sigma_post = predict_mu_sigma_raw(model, sx, sy, X_post_full, device)
    stress_mean = mu_post[:, stress_idx]

    for thr in THRESHOLDS:
        mask = stress_mean <= thr
        df_feas = df_post.loc[mask, [PARAM_X, PARAM_Y]].copy()
        df_feas.to_csv(
            f"{OUT_PREFIX}_feasible_points_thr{int(thr)}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 5) build 2D grid map
    grid_ref = df_post if len(df_post) > 0 else df_prior
    xv, yv, XX, YY = build_2d_grid(grid_ref, PARAM_X, PARAM_Y, NX, NY)

    other_vals = {}
    for c in reduced_cols:
        if c not in [PARAM_X, PARAM_Y]:
            other_vals[c] = float(df_post[c].median())

    rows = []
    grid_red = np.zeros((XX.size, len(reduced_cols)), dtype=float)

    for j, c in enumerate(reduced_cols):
        if c == PARAM_X:
            grid_red[:, j] = XX.ravel()
        elif c == PARAM_Y:
            grid_red[:, j] = YY.ravel()
        else:
            grid_red[:, j] = other_vals[c]

    if RUN_TAG == "reduced":
        grid_full = expand_reduced_to_full(grid_red, reduced_cols, x_ref_full, INPUT_COLS)
    else:
        grid_full = grid_red.copy()

    mu_grid, sigma_grid = predict_mu_sigma_raw(model, sx, sy, grid_full, device)
    stress_mu_grid = mu_grid[:, stress_idx]
    stress_sigma_grid = sigma_grid[:, stress_idx]

    for i in range(grid_red.shape[0]):
        row = {
            PARAM_X: float(grid_red[i, reduced_cols.index(PARAM_X)]),
            PARAM_Y: float(grid_red[i, reduced_cols.index(PARAM_Y)]),
            "stress_mean": float(stress_mu_grid[i]),
            "stress_sigma": float(stress_sigma_grid[i]),
        }
        for thr in THRESHOLDS:
            row[f"feasible_mean_thr{int(thr)}"] = float(stress_mu_grid[i] <= thr)
        rows.append(row)

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(f"{OUT_PREFIX}_grid_map.csv", index=False, encoding="utf-8-sig")

    # 6) save small metadata
    meta = {
        "run_tag": RUN_TAG,
        "case_id": CASE_ID,
        "level": LEVEL,
        "param_x": PARAM_X,
        "param_y": PARAM_Y,
        "thresholds": THRESHOLDS,
        "grid_shape": [NY, NX],
        "other_reduced_params_fixed_to": other_vals,
        "note": "Grid map uses posterior median values for reduced parameters not shown in the 2D plane.",
    }
    with open(f"{OUT_PREFIX}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[DONE] Exported 2D inverse plotting data:")
    print(f" - {OUT_PREFIX}_prior_points.csv")
    print(f" - {OUT_PREFIX}_posterior_points.csv")
    for thr in THRESHOLDS:
        print(f" - {OUT_PREFIX}_feasible_points_thr{int(thr)}.csv")
    print(f" - {OUT_PREFIX}_grid_map.csv")
    print(f" - {OUT_PREFIX}_meta.json")


if __name__ == "__main__":
    main()