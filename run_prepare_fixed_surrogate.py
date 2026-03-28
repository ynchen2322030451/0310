# run_prepare_fixed_surrogate.py
# ============================================================
# Prepare one fixed surrogate for all downstream experiments
#
# This script ONLY does:
#   1) freeze train/val/test split
#   2) run Optuna hyperparameter search
#   3) train one final surrogate
#   4) save checkpoint / scalers / split / metrics / predictions
#
# Downstream scripts should ONLY load the saved artifacts,
# and must NOT retrain a new neural network.
# ============================================================

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    CSV_PATH,
    OUT_DIR,
    SEED,
    TRIALS,
    INPUT_COLS,
    OUTPUT_COLS,
    ITER1_IDX,
    ITER2_IDX,
    PRIMARY_OUTPUTS,
)
from run_phys_levels_main import (
    load_dataset,
    seed_all,
    get_device,
    ensure_dir,
    summarize,
    print_stats,
    objective_factory,
    train_with_params,
    gaussian_nll,
    compute_basic_metrics,
    compute_prob_metrics_gaussian,
    eval_inequality_violation,
    _to_numpy,
)

# ============================================================
# User settings
# ============================================================

LEVEL = 2                      # fixed surrogate level for downstream experiments
TEST_FRAC = 0.15
VAL_FRAC_WITHIN_TRAIN = 0.1765  # gives ~70/15/15 overall
REMAKE_SPLIT = False           # True = overwrite frozen split
RUN_TAG = "fixed_level2"

# Optional: if you later want baseline fixed surrogate too, set LEVEL = 0 once and rerun.


# ============================================================
# Paths
# ============================================================

ROOT_OUT = Path(OUT_DIR)
SPLIT_DIR = ROOT_OUT / "fixed_split"
ART_DIR = ROOT_OUT / f"fixed_surrogate_{RUN_TAG}"

SPLIT_META_JSON = SPLIT_DIR / "split_meta.json"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
TRAIN_IDX_CSV = SPLIT_DIR / "train_indices.csv"
VAL_IDX_CSV = SPLIT_DIR / "val_indices.csv"
TEST_IDX_CSV = SPLIT_DIR / "test_indices.csv"

BEST_JSON = ART_DIR / f"best_level{LEVEL}.json"
CKPT_PT = ART_DIR / f"checkpoint_level{LEVEL}.pt"
SCALERS_PKL = ART_DIR / f"scalers_level{LEVEL}.pkl"
META_STATS_JSON = ART_DIR / "meta_stats.json"
METRICS_JSON = ART_DIR / f"metrics_level{LEVEL}.json"
PER_DIM_CSV = ART_DIR / f"paper_metrics_per_dim_level{LEVEL}.csv"
TEST_PRED_JSON = ART_DIR / f"test_predictions_level{LEVEL}.json"
SANITY_CSV = ART_DIR / f"sanity_level{LEVEL}.csv"


# ============================================================
# Split helpers
# ============================================================

def save_split_indices(idx_train, idx_val, idx_test):
    pd.DataFrame({"index": idx_train}).to_csv(TRAIN_IDX_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame({"index": idx_val}).to_csv(VAL_IDX_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame({"index": idx_test}).to_csv(TEST_IDX_CSV, index=False, encoding="utf-8-sig")


def make_split_once(df: pd.DataFrame):
    idx_all = np.arange(len(df))

    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
    )

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_FRAC_WITHIN_TRAIN,
        random_state=SEED,
        shuffle=True,
    )

    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    idx_test = np.sort(idx_test)

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    save_split_indices(idx_train, idx_val, idx_test)

    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    split_meta = {
        "csv_path": CSV_PATH,
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac_within_train": VAL_FRAC_WITHIN_TRAIN,
        "n_total": int(len(df)),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
    }
    with open(SPLIT_META_JSON, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)

    return df_train, df_val, df_test, idx_train, idx_val, idx_test


def load_frozen_split():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    idx_train = pd.read_csv(TRAIN_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_val = pd.read_csv(VAL_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_test = pd.read_csv(TEST_IDX_CSV)["index"].to_numpy(dtype=int)

    return df_train, df_val, df_test, idx_train, idx_val, idx_test


def get_or_make_split(df: pd.DataFrame):
    if (
        (not REMAKE_SPLIT)
        and TRAIN_CSV.exists()
        and VAL_CSV.exists()
        and TEST_CSV.exists()
        and TRAIN_IDX_CSV.exists()
        and VAL_IDX_CSV.exists()
        and TEST_IDX_CSV.exists()
        and SPLIT_META_JSON.exists()
    ):
        print("[INFO] Loading frozen split from disk.")
        return load_frozen_split()

    print("[INFO] Creating new frozen split.")
    return make_split_once(df)


# ============================================================
# Prepare tensors + scalers
# ============================================================

def build_scaled_pack(df_train, df_val, df_test, device):
    X_tr = df_train[INPUT_COLS].to_numpy(dtype=float)
    Y_tr = df_train[OUTPUT_COLS].to_numpy(dtype=float)

    X_va = df_val[INPUT_COLS].to_numpy(dtype=float)
    Y_va = df_val[OUTPUT_COLS].to_numpy(dtype=float)

    X_te = df_test[INPUT_COLS].to_numpy(dtype=float)
    Y_te = df_test[OUTPUT_COLS].to_numpy(dtype=float)

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s = sx.transform(X_tr)
    Xva_s = sx.transform(X_va)
    Xte_s = sx.transform(X_te)

    Ytr_s = sy.transform(Y_tr)
    Yva_s = sy.transform(Y_va)
    Yte_s = sy.transform(Y_te)

    delta_tr = Ytr_s[:, ITER2_IDX] - Ytr_s[:, ITER1_IDX]
    bias_delta = delta_tr.mean(axis=0)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    x_tr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
    x_va = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    y_va = torch.tensor(Yva_s, dtype=torch.float32, device=device)
    x_te = torch.tensor(Xte_s, dtype=torch.float32, device=device)
    y_te = torch.tensor(Yte_s, dtype=torch.float32, device=device)

    return {
        "X_tr": X_tr, "Y_tr": Y_tr,
        "X_va": X_va, "Y_va": Y_va,
        "X_te": X_te, "Y_te": Y_te,
        "Xtr_s": Xtr_s, "Ytr_s": Ytr_s,
        "Xva_s": Xva_s, "Yva_s": Yva_s,
        "Xte_s": Xte_s, "Yte_s": Yte_s,
        "sx": sx, "sy": sy,
        "x_tr": x_tr, "y_tr": y_tr,
        "x_va": x_va, "y_va": y_va,
        "x_te": x_te, "y_te": y_te,
        "bias_delta_t": bias_delta_t,
    }


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(str(ROOT_OUT))
    ensure_dir(str(SPLIT_DIR))
    ensure_dir(str(ART_DIR))

    device = get_device()
    print(f"[INFO] device = {device}")

    # 1) load full dataset
    df = load_dataset()

    in_stats = summarize(df, INPUT_COLS)
    out_stats = summarize(df, OUTPUT_COLS)
    print_stats("INPUT STATS", in_stats)
    print_stats("OUTPUT STATS", out_stats)

    with open(META_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_stats": in_stats,
                "output_stats": out_stats,
                "input_cols": INPUT_COLS,
                "output_cols": OUTPUT_COLS,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 2) freeze split
    df_train, df_val, df_test, idx_train, idx_val, idx_test = get_or_make_split(df)

    # 3) build tensors/scalers
    pack = build_scaled_pack(df_train, df_val, df_test, device)

    # 4) hyperparameter search
    print(f"\n================= OPTUNA FIXED SURROGATE LEVEL {LEVEL} =================")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )

    study.optimize(
        objective_factory(
            level=LEVEL,
            x_tr=pack["x_tr"],
            y_tr=pack["y_tr"],
            x_va=pack["x_va"],
            y_va=pack["y_va"],
            Xtr_np=pack["Xtr_s"],
            Ytr_np=pack["Ytr_s"],
            bias_delta_t=pack["bias_delta_t"],
            device=device,
        ),
        n_trials=TRIALS
    )

    best_params = study.best_params
    best_value = float(study.best_value)

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "level": LEVEL,
                "best_value": best_value,
                "best_params": best_params,
                "run_tag": RUN_TAG,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[BEST] level={LEVEL}  value={best_value:.6g}")
    print(best_params)

    # 5) train final fixed surrogate
    model, mono_pairs = train_with_params(
        best_params=best_params,
        level=LEVEL,
        x_tr=pack["x_tr"],
        y_tr=pack["y_tr"],
        x_va=pack["x_va"],
        y_va=pack["y_va"],
        Xtr_np=pack["Xtr_s"],
        Ytr_np=pack["Ytr_s"],
        bias_delta_t=pack["bias_delta_t"],
        device=device,
    )

    # 6) save checkpoint + scalers
    torch.save(
        {
            "level": LEVEL,
            "run_tag": RUN_TAG,
            "best_params": best_params,
            "model_state_dict": model.state_dict(),
            "split_meta_path": str(SPLIT_META_JSON),
            "train_indices_path": str(TRAIN_IDX_CSV),
            "val_indices_path": str(VAL_IDX_CSV),
            "test_indices_path": str(TEST_IDX_CSV),
        },
        CKPT_PT,
    )

    with open(SCALERS_PKL, "wb") as f:
        pickle.dump({"sx": pack["sx"], "sy": pack["sy"]}, f)

    # 7) evaluate on frozen test set
    with torch.no_grad():
        mu_te_s, logvar_te = model(pack["x_te"])
        var_te_s = torch.exp(logvar_te)

    mu_te_s_np = _to_numpy(mu_te_s)
    y_te_s_np = _to_numpy(pack["y_te"])

    mu_te = pack["sy"].inverse_transform(mu_te_s_np)
    y_te_true = pack["sy"].inverse_transform(y_te_s_np)
    sigma_te = np.sqrt(_to_numpy(var_te_s)) * pack["sy"].scale_

    basic = compute_basic_metrics(y_te_true, mu_te)
    prob90 = compute_prob_metrics_gaussian(y_te_true, mu_te, sigma_te, alpha=0.10)
    test_nll = float(gaussian_nll(pack["y_te"], mu_te_s, logvar_te).item())

    idx_map = {c: i for i, c in enumerate(OUTPUT_COLS)}
    viol = eval_inequality_violation(mu_te, idx_map)

    primary_idx = [OUTPUT_COLS.index(c) for c in PRIMARY_OUTPUTS]

    metrics = {
        "level": LEVEL,
        "run_tag": RUN_TAG,
        "fixed_surrogate": True,
        "split_meta_path": str(SPLIT_META_JSON),
        "test_nll_standardized": test_nll,
        "basic_all_mean": {
            "MAE_mean": float(np.mean(basic["MAE"])),
            "RMSE_mean": float(np.mean(basic["RMSE"])),
            "R2_mean": float(np.mean(basic["R2"])),
        },
        "basic_primary_mean": {
            "MAE_mean": float(np.mean(basic["MAE"][primary_idx])),
            "RMSE_mean": float(np.mean(basic["RMSE"][primary_idx])),
            "R2_mean": float(np.mean(basic["R2"][primary_idx])),
        },
        "prob90_all_mean": {
            "PICP_mean": float(np.mean(prob90["PICP"])),
            "MPIW_mean": float(np.mean(prob90["MPIW"])),
            "CRPS_mean": float(np.mean(prob90["CRPS"])),
        },
        "prob90_primary_mean": {
            "PICP_mean": float(np.mean(prob90["PICP"][primary_idx])),
            "MPIW_mean": float(np.mean(prob90["MPIW"][primary_idx])),
            "CRPS_mean": float(np.mean(prob90["CRPS"][primary_idx])),
        },
        "ineq_violation_rates_on_mu": viol,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 8) save per-dim table
    per_dim_rows = []
    for j, name in enumerate(OUTPUT_COLS):
        per_dim_rows.append({
            "level": LEVEL,
            "run_tag": RUN_TAG,
            "output": name,
            "iter": "iter1" if j < 8 else "iter2",
            "MAE": float(basic["MAE"][j]),
            "RMSE": float(basic["RMSE"][j]),
            "R2": float(basic["R2"][j]),
            "PICP90": float(prob90["PICP"][j]),
            "MPIW90": float(prob90["MPIW"][j]),
            "CRPS": float(prob90["CRPS"][j]),
        })
    pd.DataFrame(per_dim_rows).to_csv(PER_DIM_CSV, index=False, encoding="utf-8-sig")

    # 9) save raw test predictions
    pred_dump = {
        "level": LEVEL,
        "run_tag": RUN_TAG,
        "fixed_surrogate": True,
        "y_true": y_te_true.tolist(),
        "mu": mu_te.tolist(),
        "sigma": sigma_te.tolist(),
        "output_names": OUTPUT_COLS,
        "split_meta_path": str(SPLIT_META_JSON),
    }
    with open(TEST_PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(pred_dump, f, indent=2, ensure_ascii=False)

    # 10) save sanity csv
    sanity_rows = []
    for j, name in enumerate(OUTPUT_COLS):
        yt = y_te_true[:, j]
        yp = mu_te[:, j]
        sanity_rows.append({
            "output": name,
            "y_true_mean": float(np.mean(yt)),
            "y_true_std": float(np.std(yt)),
            "y_pred_mean": float(np.mean(yp)),
            "y_pred_std": float(np.std(yp)),
            "y_true_min": float(np.min(yt)),
            "y_true_max": float(np.max(yt)),
            "y_pred_min": float(np.min(yp)),
            "y_pred_max": float(np.max(yp)),
        })
    pd.DataFrame(sanity_rows).to_csv(SANITY_CSV, index=False, encoding="utf-8-sig")

    print("\n[DONE] Fixed surrogate prepared successfully.")
    print(f"[ART] split saved to: {SPLIT_DIR}")
    print(f"[ART] surrogate artifacts saved to: {ART_DIR}")
    print(f"[ART] checkpoint: {CKPT_PT}")
    print(f"[ART] scalers:   {SCALERS_PKL}")
    print(f"[ART] metrics:   {METRICS_JSON}")


if __name__ == "__main__":
    main()