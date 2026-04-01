# inspect_main_run_results.py
# ============================================================
# Inspect results produced by run_phys_levels_main.py
#
# What it does:
#   1) reload dataset and reconstruct the SAME random split
#   2) load checkpoint/scalers from OUT_DIR root
#   3) evaluate a chosen level on test set
#   4) export:
#        - overall summary
#        - per-output metrics
#        - per-input/per-output diagnostics
#        - quantile-bin error tables
#
# Usage:
#   python inspect_main_run_results.py
#   or modify LEVEL below
# ============================================================

import os
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

from paper_experiment_config import (
    CSV_PATH,
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    SEED,
)
from run_phys_levels_main import (
    HeteroMLP,
    load_dataset,
    split_and_scale,
    get_device,
    gaussian_nll,
    compute_basic_metrics,
    compute_prob_metrics_gaussian,
)

# ============================================================
# User settings
# ============================================================

LEVEL = 2          # 改成 0 / 1 / 2 / 4 都可以
N_INPUT_BINS = 5   # 每个输入分成几个分位数箱
OUT_SUBDIR = Path(OUT_DIR) / f"inspect_level{LEVEL}"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return np.nan

    try:
        if method == "pearson":
            return float(pearsonr(x, y)[0])
        elif method == "spearman":
            return float(spearmanr(x, y)[0])
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception:
        return np.nan


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


def load_artifacts(level: int):
    ckpt_path = Path(OUT_DIR) / f"checkpoint_level{level}.pt"
    scaler_path = Path(OUT_DIR) / f"scalers_level{level}.pkl"
    metrics_path = Path(OUT_DIR) / f"metrics_level{level}.json"
    sanity_path = Path(OUT_DIR) / f"sanity_level{level}.csv"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scalers: {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    metrics_json = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)

    sanity_df = pd.read_csv(sanity_path) if sanity_path.exists() else None

    return ckpt, scalers["sx"], scalers["sy"], metrics_json, sanity_df


@torch.no_grad()
def predict_raw(model, sx, sy, X_raw, device):
    X_s = sx.transform(X_raw)
    xt = torch.tensor(X_s, dtype=torch.float32, device=device)

    mu_s, logvar_s = model(xt)

    mu_s_np = mu_s.detach().cpu().numpy()
    logvar_s_np = logvar_s.detach().cpu().numpy()

    mu_raw = sy.inverse_transform(mu_s_np)
    sigma_raw = np.sqrt(np.exp(logvar_s_np)) * sy.scale_

    return mu_raw, sigma_raw, mu_s, logvar_s


def rmse_vec(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae_vec(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)


def make_bin_labels(series, n_bins=5):
    q = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(series, q)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # 去重，避免常值或近常值输入导致 qcut 失败
    bins = np.unique(bins)
    if len(bins) < 3:
        return pd.Series(["all"] * len(series)), ["all"]

    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if i == 0:
            labels.append(f"(-inf, {right:.4g}]")
        elif i == len(bins) - 2:
            labels.append(f"({left:.4g}, +inf)")
        else:
            labels.append(f"({left:.4g}, {right:.4g}]")

    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return binned.astype(str), labels


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUT_SUBDIR)
    device = get_device()
    print(f"[INFO] device = {device}")

    # --------------------------------------------------------
    # 1) reconstruct SAME random split used by run_phys_levels_main.py
    # --------------------------------------------------------
    df = load_dataset()
    split = split_and_scale(df)

    X_te = split["X_te"]
    Y_te = split["Y_te"]

    # --------------------------------------------------------
    # 2) load artifacts
    # --------------------------------------------------------
    ckpt, sx, sy, old_metrics, old_sanity = load_artifacts(LEVEL)
    model = build_model_from_ckpt(ckpt, device)

    # --------------------------------------------------------
    # 3) predict
    # --------------------------------------------------------
    mu_raw, sigma_raw, mu_s_t, logvar_s_t = predict_raw(model, sx, sy, X_te, device)

    y_true = Y_te.copy()
    y_pred = mu_raw.copy()
    sigma = sigma_raw.copy()

    # standardized y_te for NLL
    y_te_s = sy.transform(Y_te)
    y_te_s_t = torch.tensor(y_te_s, dtype=torch.float32, device=device)

    test_nll = float(gaussian_nll(y_te_s_t, mu_s_t, logvar_s_t).item())
    basic = compute_basic_metrics(y_true, y_pred)
    prob = compute_prob_metrics_gaussian(y_true, y_pred, sigma, alpha=0.10)

    # --------------------------------------------------------
    # 4) overall summary
    # --------------------------------------------------------
    primary_idx = [OUTPUT_COLS.index(c) for c in PRIMARY_OUTPUTS]

    overall = {
        "level": LEVEL,
        "n_test": int(len(X_te)),
        "test_nll_standardized": test_nll,
        "all_outputs": {
            "MAE_mean": float(np.mean(basic["MAE"])),
            "RMSE_mean": float(np.mean(basic["RMSE"])),
            "R2_mean": float(np.mean(basic["R2"])),
            "PICP90_mean": float(np.mean(prob["PICP"])),
            "MPIW90_mean": float(np.mean(prob["MPIW"])),
            "CRPS_mean": float(np.mean(prob["CRPS"])),
        },
        "primary_outputs": {
            "MAE_mean": float(np.mean(basic["MAE"][primary_idx])),
            "RMSE_mean": float(np.mean(basic["RMSE"][primary_idx])),
            "R2_mean": float(np.mean(basic["R2"][primary_idx])),
            "PICP90_mean": float(np.mean(prob["PICP"][primary_idx])),
            "MPIW90_mean": float(np.mean(prob["MPIW"][primary_idx])),
            "CRPS_mean": float(np.mean(prob["CRPS"][primary_idx])),
        },
    }

    with open(OUT_SUBDIR / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 5) per-output metrics
    # --------------------------------------------------------
    per_output_rows = []
    abs_err = np.abs(y_pred - y_true)
    sq_err = (y_pred - y_true) ** 2

    for j, out in enumerate(OUTPUT_COLS):
        per_output_rows.append({
            "level": LEVEL,
            "output": out,
            "iter": "iter1" if j < 8 else "iter2",
            "group": "primary" if out in PRIMARY_OUTPUTS else "secondary",
            "MAE": float(basic["MAE"][j]),
            "RMSE": float(basic["RMSE"][j]),
            "R2": float(basic["R2"][j]),
            "PICP90": float(prob["PICP"][j]),
            "MPIW90": float(prob["MPIW"][j]),
            "CRPS": float(prob["CRPS"][j]),
            "y_true_mean": float(np.mean(y_true[:, j])),
            "y_true_std": float(np.std(y_true[:, j])),
            "y_pred_mean": float(np.mean(y_pred[:, j])),
            "y_pred_std": float(np.std(y_pred[:, j])),
            "sigma_mean": float(np.mean(sigma[:, j])),
            "sigma_std": float(np.std(sigma[:, j])),
        })

    per_output_df = pd.DataFrame(per_output_rows)
    per_output_df.to_csv(OUT_SUBDIR / "per_output_metrics.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 6) per-input × per-output diagnostics
    # --------------------------------------------------------
    diag_rows = []
    for i, inp in enumerate(INPUT_COLS):
        x = X_te[:, i]
        for j, out in enumerate(OUTPUT_COLS):
            yt = y_true[:, j]
            yp = y_pred[:, j]
            ae = np.abs(yp - yt)
            se = (yp - yt) ** 2

            diag_rows.append({
                "level": LEVEL,
                "input": inp,
                "output": out,
                "iter": "iter1" if j < 8 else "iter2",
                "group": "primary" if out in PRIMARY_OUTPUTS else "secondary",

                # 输入与真值/预测的关联
                "pearson_input_true": safe_corr(x, yt, "pearson"),
                "spearman_input_true": safe_corr(x, yt, "spearman"),
                "pearson_input_pred": safe_corr(x, yp, "pearson"),
                "spearman_input_pred": safe_corr(x, yp, "spearman"),

                # 输入与误差大小的关联
                "pearson_input_abs_error": safe_corr(x, ae, "pearson"),
                "spearman_input_abs_error": safe_corr(x, ae, "spearman"),
                "pearson_input_sq_error": safe_corr(x, se, "pearson"),
                "spearman_input_sq_error": safe_corr(x, se, "spearman"),

                # 输入与预测不确定性的关联
                "pearson_input_sigma": safe_corr(x, sigma[:, j], "pearson"),
                "spearman_input_sigma": safe_corr(x, sigma[:, j], "spearman"),
            })

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_SUBDIR / "input_output_diagnostics.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 7) input-bin local error tables
    # --------------------------------------------------------
    bin_rows = []

    for i, inp in enumerate(INPUT_COLS):
        x = X_te[:, i]
        bin_series, labels = make_bin_labels(x, n_bins=N_INPUT_BINS)

        tmp = pd.DataFrame({"bin": bin_series})

        for j, out in enumerate(OUTPUT_COLS):
            tmp[f"{out}__y_true"] = y_true[:, j]
            tmp[f"{out}__y_pred"] = y_pred[:, j]
            tmp[f"{out}__sigma"] = sigma[:, j]

        for bin_name, sub in tmp.groupby("bin"):
            if len(sub) == 0:
                continue

            idx = sub.index.to_numpy()

            for j, out in enumerate(OUTPUT_COLS):
                yt = y_true[idx, j]
                yp = y_pred[idx, j]
                sg = sigma[idx, j]

                lo = yp - 1.6448536269514722 * sg
                hi = yp + 1.6448536269514722 * sg
                picp = np.mean((yt >= lo) & (yt <= hi))

                bin_rows.append({
                    "level": LEVEL,
                    "input": inp,
                    "input_bin": bin_name,
                    "n_in_bin": int(len(idx)),
                    "output": out,
                    "iter": "iter1" if j < 8 else "iter2",
                    "group": "primary" if out in PRIMARY_OUTPUTS else "secondary",
                    "bin_x_mean": float(np.mean(x[idx])),
                    "bin_x_min": float(np.min(x[idx])),
                    "bin_x_max": float(np.max(x[idx])),
                    "MAE_bin": float(np.mean(np.abs(yp - yt))),
                    "RMSE_bin": float(np.sqrt(np.mean((yp - yt) ** 2))),
                    "y_true_mean_bin": float(np.mean(yt)),
                    "y_pred_mean_bin": float(np.mean(yp)),
                    "sigma_mean_bin": float(np.mean(sg)),
                    "PICP90_bin": float(picp),
                })

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(OUT_SUBDIR / "input_bin_local_error_table.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 8) focus tables for quick reading
    # --------------------------------------------------------
    focus_outputs = PRIMARY_OUTPUTS
    focus_bin_df = bin_df[bin_df["output"].isin(focus_outputs)].copy()
    focus_bin_df.to_csv(OUT_SUBDIR / "input_bin_local_error_primary_only.csv", index=False, encoding="utf-8-sig")

    focus_diag_df = diag_df[diag_df["output"].isin(focus_outputs)].copy()
    focus_diag_df.to_csv(OUT_SUBDIR / "input_output_diagnostics_primary_only.csv", index=False, encoding="utf-8-sig")

    # 每个 primary 输出下，按“输入与绝对误差的 Spearman”排序
    ranked_rows = []
    for out in focus_outputs:
        sub = focus_diag_df[focus_diag_df["output"] == out].copy()
        sub["abs_rank"] = sub["spearman_input_abs_error"].abs().rank(ascending=False, method="first")
        sub = sub.sort_values("abs_rank")
        ranked_rows.append(sub)

    if ranked_rows:
        ranked_df = pd.concat(ranked_rows, ignore_index=True)
        ranked_df.to_csv(OUT_SUBDIR / "primary_outputs_error_sensitivity_ranked.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] done. Results saved to: {OUT_SUBDIR}")
    print("Files:")
    print(" - overall_summary.json")
    print(" - per_output_metrics.csv")
    print(" - input_output_diagnostics.csv")
    print(" - input_output_diagnostics_primary_only.csv")
    print(" - input_bin_local_error_table.csv")
    print(" - input_bin_local_error_primary_only.csv")
    print(" - primary_outputs_error_sensitivity_ranked.csv")


if __name__ == "__main__":
    main()