# run_train_fixed_surrogates.py
# ============================================================
# UNIFIED training script: replaces both
#   run_prepare_fixed_surrogate.py  (Level 2)
#   run_fixed_surrogate_train_base.py (Level 0)
#
# What it does:
#   1) Load dataset, create/load ONE fixed split (shared by both levels)
#   2) Run Optuna for Level 0  → fixed_surrogate_fixed_base/
#   3) Run Optuna for Level 2  → fixed_surrogate_fixed_level2/
#
# All downstream scripts (forward UQ, Sobol, inverse, OOD, speed)
# read from fixed_surrogate_fixed_base/ or fixed_surrogate_fixed_level2/
# as configured in paper_experiment_config.py.
#
# Run:
#   python run_train_fixed_surrogates.py
# ============================================================
import shutil
import json
import pickle
import random
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
    PRIMARY_OUTPUTS,
    DELTA_PAIRS,
)

from run_phys_levels_main import (
    load_dataset,
    seed_all,
    get_device,
    ensure_dir,
    summarize,
    print_stats,
    HeteroMLP,
    gaussian_nll,
    objective_factory,
    train_with_params,
    compute_basic_metrics,
    compute_prob_metrics_gaussian,
    eval_inequality_violation,
    _to_numpy,
)

# ============================================================
# Settings — do not change unless you know what you're doing
# ============================================================

TRAIN_LEVELS  = [0, 2]          # train both levels in one run
REMAKE_SPLIT  = True           # True = overwrite existing frozen split
TEST_FRAC     = 0.15
VAL_FRAC      = 0.1765          # of train+val → gives ~68/15/15 overall

LEVEL_TAGS = {0: "fixed_base", 2: "fixed_level2"}

ROOT_OUT   = Path(OUT_DIR)
SPLIT_DIR  = ROOT_OUT / "fixed_split"

SPLIT_META_JSON = SPLIT_DIR / "split_meta.json"
TRAIN_CSV       = SPLIT_DIR / "train.csv"
VAL_CSV         = SPLIT_DIR / "val.csv"
TEST_CSV        = SPLIT_DIR / "test.csv"
TRAIN_IDX_CSV   = SPLIT_DIR / "train_indices.csv"
VAL_IDX_CSV     = SPLIT_DIR / "val_indices.csv"
TEST_IDX_CSV    = SPLIT_DIR / "test_indices.csv"


# ============================================================
# Split helpers
# ============================================================

def make_split_once(df: pd.DataFrame):
    idx_all = np.arange(len(df))
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=TEST_FRAC, random_state=SEED, shuffle=True
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=VAL_FRAC, random_state=SEED, shuffle=True
    )
    idx_train = np.sort(idx_train)
    idx_val   = np.sort(idx_val)
    idx_test  = np.sort(idx_test)

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    ensure_dir(str(SPLIT_DIR))
    pd.DataFrame({"index": idx_train}).to_csv(TRAIN_IDX_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame({"index": idx_val}).to_csv(VAL_IDX_CSV,   index=False, encoding="utf-8-sig")
    pd.DataFrame({"index": idx_test}).to_csv(TEST_IDX_CSV,  index=False, encoding="utf-8-sig")
    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV,   index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV,  index=False, encoding="utf-8-sig")

    meta = {
        "csv_path": CSV_PATH,
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac_within_train": VAL_FRAC,
        "n_total": len(df),
        "n_train": len(df_train),
        "n_val":   len(df_val),
        "n_test":  len(df_test),
        "input_cols":  INPUT_COLS,
        "output_cols": OUTPUT_COLS,
    }
    with open(SPLIT_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[SPLIT] Created: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, idx_train, idx_val, idx_test


def load_frozen_split():
    for p in [SPLIT_META_JSON, TRAIN_CSV, VAL_CSV, TEST_CSV,
              TRAIN_IDX_CSV, VAL_IDX_CSV, TEST_IDX_CSV]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing frozen split file: {p}\n"
                                    f"Set REMAKE_SPLIT = True to create it.")

    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    idx_train = pd.read_csv(TRAIN_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_val   = pd.read_csv(VAL_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_test  = pd.read_csv(TEST_IDX_CSV)["index"].to_numpy(dtype=int)

    print(f"[SPLIT] Loaded frozen: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, idx_train, idx_val, idx_test


def get_or_make_split(df):
    if (not REMAKE_SPLIT) and SPLIT_META_JSON.exists():
        return load_frozen_split()
    return make_split_once(df)


# ============================================================
# Scaling
# ============================================================

def build_pack(df_train, df_val, df_test, device):
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

    # bias for fixed-point regularization (iter2 - iter1 delta mean)
    pair_iter1_idx = [OUTPUT_COLS.index(a) for a, b in DELTA_PAIRS]
    pair_iter2_idx = [OUTPUT_COLS.index(b) for a, b in DELTA_PAIRS]

    delta_tr = Ytr_s[:, pair_iter2_idx] - Ytr_s[:, pair_iter1_idx]
    bias_delta = delta_tr.mean(axis=0)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    return {
        "X_tr": X_tr, "Y_tr": Y_tr,
        "X_va": X_va, "Y_va": Y_va,
        "X_te": X_te, "Y_te": Y_te,
        "Xtr_s": Xtr_s, "Ytr_s": Ytr_s,
        "Xva_s": Xva_s, "Yva_s": Yva_s,
        "Xte_s": Xte_s, "Yte_s": Yte_s,
        "sx": sx, "sy": sy,
        "x_tr": torch.tensor(Xtr_s, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(Ytr_s, dtype=torch.float32, device=device),
        "x_va": torch.tensor(Xva_s, dtype=torch.float32, device=device),
        "y_va": torch.tensor(Yva_s, dtype=torch.float32, device=device),
        "x_te": torch.tensor(Xte_s, dtype=torch.float32, device=device),
        "y_te": torch.tensor(Yte_s, dtype=torch.float32, device=device),
        "bias_delta_t": bias_delta_t,
    }


# ============================================================
# Train one level and save all artifacts
# ============================================================

def train_and_save_level(level, tag, pack, df_train, df_val, df_test,
                          idx_train, idx_val, idx_test, device):

    art_dir = ROOT_OUT / f"fixed_surrogate_{tag}"
    ensure_dir(str(art_dir))

    best_json   = art_dir / f"best_level{level}.json"
    ckpt_pt     = art_dir / f"checkpoint_level{level}.pt"
    scalers_pkl = art_dir / f"scalers_level{level}.pkl"
    metrics_json = art_dir / f"metrics_level{level}.json"
    per_dim_csv  = art_dir / f"paper_metrics_per_dim_level{level}.csv"
    focus_csv    = art_dir / f"paper_focus_metrics_level{level}.csv"
    test_pred_json = art_dir / f"test_predictions_level{level}.json"
    meta_stats_json = art_dir / "meta_stats.json"

    # ---- meta stats ----
    X_all = np.vstack([pack["X_tr"], pack["X_va"], pack["X_te"]])
    Y_all = np.vstack([pack["Y_tr"], pack["Y_va"], pack["Y_te"]])
    in_stats  = {c: {"mean": float(X_all[:,i].mean()), "std": float(X_all[:,i].std()),
                      "min": float(X_all[:,i].min()), "max": float(X_all[:,i].max())}
                 for i, c in enumerate(INPUT_COLS)}
    out_stats = {c: {"mean": float(Y_all[:,j].mean()), "std": float(Y_all[:,j].std()),
                      "min": float(Y_all[:,j].min()), "max": float(Y_all[:,j].max())}
                 for j, c in enumerate(OUTPUT_COLS)}
    with open(meta_stats_json, "w", encoding="utf-8") as f:
        json.dump({"level": level, "tag": tag,
                   "input_cols": INPUT_COLS, "output_cols": OUTPUT_COLS,
                   "input_stats": in_stats, "output_stats": out_stats},
                  f, indent=2, ensure_ascii=False)

    # ---- Optuna hyperparameter search ----
    print(f"\n{'='*60}")
    print(f"  OPTUNA  Level {level}  ({TRIALS} trials)")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    study.optimize(
        objective_factory(
            level=level,
            x_tr=pack["x_tr"],
            y_tr=pack["y_tr"],
            x_va=pack["x_va"],
            y_va=pack["y_va"],
            Xtr_np=pack["Xtr_s"],
            Ytr_np=pack["Ytr_s"],
            bias_delta_t=pack["bias_delta_t"],
            device=device,
        ),
        n_trials=TRIALS,
    )

    best_params = study.best_params
    best_value  = float(study.best_value)
    print(f"[BEST] level={level}  val_nll={best_value:.6f}")
    print(f"       params={best_params}")

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump({"level": level, "tag": tag,
                   "best_value": best_value, "best_params": best_params},
                  f, indent=2, ensure_ascii=False)

    # ---- Train final model ----
    model, _ = train_with_params(
        best_params=best_params,
        level=level,
        x_tr=pack["x_tr"],
        y_tr=pack["y_tr"],
        x_va=pack["x_va"],
        y_va=pack["y_va"],
        Xtr_np=pack["Xtr_s"],
        Ytr_np=pack["Ytr_s"],
        bias_delta_t=pack["bias_delta_t"],
        device=device,
    )
    model.eval()

    # ---- Save checkpoint + scalers ----
    torch.save({
        "level": level,
        "tag": tag,
        "best_params": best_params,
        "model_state_dict": model.state_dict(),
        "split_dir": str(SPLIT_DIR),
        "train_indices_path": str(TRAIN_IDX_CSV),
        "val_indices_path":   str(VAL_IDX_CSV),
        "test_indices_path":  str(TEST_IDX_CSV),
    }, ckpt_pt)

    with open(scalers_pkl, "wb") as f:
        pickle.dump({"sx": pack["sx"], "sy": pack["sy"]}, f)

    print(f"[SAVED] checkpoint → {ckpt_pt}")
    print(f"[SAVED] scalers    → {scalers_pkl}")

    # ---- Evaluate on test set ----
    with torch.no_grad():
        mu_te_s, logvar_te = model(pack["x_te"])
        var_te_s = torch.exp(logvar_te)

    mu_te    = pack["sy"].inverse_transform(_to_numpy(mu_te_s))
    y_te     = pack["sy"].inverse_transform(_to_numpy(pack["y_te"]))
    sigma_te = np.sqrt(_to_numpy(var_te_s)) * pack["sy"].scale_

    basic  = compute_basic_metrics(y_te, mu_te)
    prob90 = compute_prob_metrics_gaussian(y_te, mu_te, sigma_te, alpha=0.10)

    nll_te_s = gaussian_nll(pack["y_te"], mu_te_s, logvar_te).item()

    # Primary outputs only
    pri_idx = [OUTPUT_COLS.index(c) for c in PRIMARY_OUTPUTS]
    pri_metrics = {
        "RMSE_mean": float(np.mean([basic["RMSE"][i] for i in pri_idx])),
        "R2_mean":   float(np.mean([basic["R2"][i]   for i in pri_idx])),
        "PICP90_mean": float(np.mean([prob90["PICP"][i] for i in pri_idx])),
        "MPIW90_mean": float(np.mean([prob90["MPIW"][i] for i in pri_idx])),
        "CRPS_mean":   float(np.mean([prob90["CRPS"][i] for i in pri_idx])),
    }

    metrics = {
        "level": level,
        "tag": tag,
        "test_nll_standardized": nll_te_s,
        "primary_outputs": PRIMARY_OUTPUTS,
        "primary_metrics": pri_metrics,
        "all_outputs_RMSE_mean": float(np.mean(basic["RMSE"])),
        "all_outputs_R2_mean":   float(np.mean(basic["R2"])),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Per-output CSV
    per_dim_rows = []
    for j, name in enumerate(OUTPUT_COLS):
        per_dim_rows.append({
            "level": level, "output": name,
            "MAE":    float(basic["MAE"][j]),
            "RMSE":   float(basic["RMSE"][j]),
            "R2":     float(basic["R2"][j]),
            "PICP90": float(prob90["PICP"][j]),
            "MPIW90": float(prob90["MPIW"][j]),
            "CRPS":   float(prob90["CRPS"][j]),
        })
    pd.DataFrame(per_dim_rows).to_csv(per_dim_csv, index=False, encoding="utf-8-sig")

    # Focus (primary) outputs CSV
    focus_rows = [r for r in per_dim_rows if r["output"] in PRIMARY_OUTPUTS]
    pd.DataFrame(focus_rows).to_csv(focus_csv, index=False, encoding="utf-8-sig")

    # Test predictions (for downstream scripts that need them)
    pred_dump = {
        "level": level,
        "test_indices": idx_test.tolist(),
        "mu_te": mu_te.tolist(),
        "sigma_te": sigma_te.tolist(),
        "y_te_true": y_te.tolist(),
    }
    with open(test_pred_json, "w", encoding="utf-8") as f:
        json.dump(pred_dump, f, ensure_ascii=False)

    # ---- Root-level compatibility sync ----
    root_metrics_json = ROOT_OUT / f"metrics_level{level}.json"
    root_test_pred_json = ROOT_OUT / f"test_predictions_level{level}.json"
    shutil.copy2(metrics_json, root_metrics_json)
    shutil.copy2(test_pred_json, root_test_pred_json)

    print(f"\n[METRICS] Level {level}:")
    print(f"  test NLL (std):  {nll_te_s:.4f}")
    print(f"  primary RMSE:    {pri_metrics['RMSE_mean']:.4f}")
    print(f"  primary R²:      {pri_metrics['R2_mean']:.4f}")
    print(f"  primary PICP90:  {pri_metrics['PICP90_mean']:.4f}")
    print(f"  primary MPIW90:  {pri_metrics['MPIW90_mean']:.4f}")

    return model


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(str(ROOT_OUT))
    device = get_device()
    print(f"[INFO] device = {device}")
    print(f"[INFO] levels = {TRAIN_LEVELS}")
    print(f"[INFO] REMAKE_SPLIT = {REMAKE_SPLIT}")

    df = load_dataset()
    print(f"[INFO] dataset loaded: n={len(df)}")

    df_train, df_val, df_test, idx_train, idx_val, idx_test = get_or_make_split(df)

    pack = build_pack(df_train, df_val, df_test, device)

    for level in TRAIN_LEVELS:
        tag = LEVEL_TAGS[level]
        train_and_save_level(
            level, tag, pack,
            df_train, df_val, df_test,
            idx_train, idx_val, idx_test,
            device,
        )

    # ---- Build root-level summary table for paper compatibility ----
    summary_rows = []
    for level in TRAIN_LEVELS:
        tag = LEVEL_TAGS[level]
        metrics_path = ROOT_OUT / f"fixed_surrogate_{tag}" / f"metrics_level{level}.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        row = {
            "level": level,
            "test_nll_std": m["test_nll_standardized"],
            "RMSE_mean_primary": m["primary_metrics"]["RMSE_mean"],
            "R2_mean_primary": m["primary_metrics"]["R2_mean"],
            "PICP90_mean_primary": m["primary_metrics"]["PICP90_mean"],
            "MPIW90_mean_primary": m["primary_metrics"]["MPIW90_mean"],
            "CRPS_mean_primary": m["primary_metrics"]["CRPS_mean"],
        }
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        ROOT_OUT / "paper_metrics_table.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{'='*60}")
    print("  ALL DONE")
    print(f"  Artifacts saved under: {ROOT_OUT}")
    for level in TRAIN_LEVELS:
        tag = LEVEL_TAGS[level]
        print(f"  Level {level}: {ROOT_OUT / f'fixed_surrogate_{tag}'}/")
    print(f"  Shared split: {SPLIT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
