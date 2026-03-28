# run_fixed_surrogate_train_base.py
# ============================================================
# Train fixed-split BASE surrogate (level 0 only)
# Reuse the same fixed split as fixed_level2
# and keep artifact layout consistent with fixed-surrogate pipeline
# ============================================================

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    SEED,
)

from run_phys_levels_main import (
    get_device,
    gaussian_nll,
    compute_basic_metrics,
    compute_prob_metrics_gaussian,
    eval_inequality_violation,
    train_with_params,
)

# ============================================================
# Settings
# ============================================================

LEVEL = 0
RUN_TAG = "fixed_base"
N_TRIALS = 40   # 可先 20 快跑，正式再 40

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
FOCUS_CSV = ART_DIR / f"paper_focus_metrics_level{LEVEL}.csv"
TEST_PRED_JSON = ART_DIR / f"test_predictions_level{LEVEL}.json"
SANITY_JSON = ART_DIR / f"sanity_level{LEVEL}.json"
SANITY_CSV = ART_DIR / f"sanity_level{LEVEL}.csv"

# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_output_sanity(y_true, y_pred, output_cols, near_constant_std_tol=1e-8):
    rows = []
    for j, name in enumerate(output_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append({
            "output": name,
            "y_true_mean": float(np.mean(yt)),
            "y_true_std": float(np.std(yt)),
            "y_true_min": float(np.min(yt)),
            "y_true_max": float(np.max(yt)),
            "y_pred_mean": float(np.mean(yp)),
            "y_pred_std": float(np.std(yp)),
            "y_pred_min": float(np.min(yp)),
            "y_pred_max": float(np.max(yp)),
            "is_near_constant_target": bool(np.std(yt) < near_constant_std_tol),
        })
    return rows


def load_fixed_split():
    for p in [
        SPLIT_META_JSON,
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        TRAIN_IDX_CSV,
        VAL_IDX_CSV,
        TEST_IDX_CSV,
    ]:
        require_file(p)

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    idx_train = pd.read_csv(TRAIN_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_val = pd.read_csv(VAL_IDX_CSV)["index"].to_numpy(dtype=int)
    idx_test = pd.read_csv(TEST_IDX_CSV)["index"].to_numpy(dtype=int)

    X_tr = df_train[INPUT_COLS].to_numpy(dtype=float)
    X_va = df_val[INPUT_COLS].to_numpy(dtype=float)
    X_te = df_test[INPUT_COLS].to_numpy(dtype=float)

    Y_tr = df_train[OUTPUT_COLS].to_numpy(dtype=float)
    Y_va = df_val[OUTPUT_COLS].to_numpy(dtype=float)
    Y_te = df_test[OUTPUT_COLS].to_numpy(dtype=float)

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "X_tr": X_tr,
        "X_va": X_va,
        "X_te": X_te,
        "Y_tr": Y_tr,
        "Y_va": Y_va,
        "Y_te": Y_te,
    }


def scale_split(split):
    sx = StandardScaler().fit(split["X_tr"])
    sy = StandardScaler().fit(split["Y_tr"])

    Xtr_s = sx.transform(split["X_tr"])
    Xva_s = sx.transform(split["X_va"])
    Xte_s = sx.transform(split["X_te"])

    Ytr_s = sy.transform(split["Y_tr"])
    Yva_s = sy.transform(split["Y_va"])
    Yte_s = sy.transform(split["Y_te"])

    return {
        **split,
        "sx": sx,
        "sy": sy,
        "Xtr_s": Xtr_s,
        "Xva_s": Xva_s,
        "Xte_s": Xte_s,
        "Ytr_s": Ytr_s,
        "Yva_s": Yva_s,
        "Yte_s": Yte_s,
    }


def make_tensors(split_s, device):
    x_tr = torch.tensor(split_s["Xtr_s"], dtype=torch.float32, device=device)
    y_tr = torch.tensor(split_s["Ytr_s"], dtype=torch.float32, device=device)
    x_va = torch.tensor(split_s["Xva_s"], dtype=torch.float32, device=device)
    y_va = torch.tensor(split_s["Yva_s"], dtype=torch.float32, device=device)
    x_te = torch.tensor(split_s["Xte_s"], dtype=torch.float32, device=device)
    y_te = torch.tensor(split_s["Yte_s"], dtype=torch.float32, device=device)

    # level 0 不使用 fixed-point / monotonicity / inequality
    # 但 train_with_params 仍要求传入
    bias_delta_t = torch.zeros(8, dtype=torch.float32, device=device)

    return {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "x_va": x_va,
        "y_va": y_va,
        "x_te": x_te,
        "y_te": y_te,
        "bias_delta_t": bias_delta_t,
    }


def save_meta_stats(split):
    input_stats = {}
    output_stats = {}

    X_all = np.vstack([split["X_tr"], split["X_va"], split["X_te"]])
    Y_all = np.vstack([split["Y_tr"], split["Y_va"], split["Y_te"]])

    for i, c in enumerate(INPUT_COLS):
        col = X_all[:, i]
        input_stats[c] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    for j, c in enumerate(OUTPUT_COLS):
        col = Y_all[:, j]
        output_stats[c] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    meta = {
        "run_tag": RUN_TAG,
        "level": LEVEL,
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
        "input_stats": input_stats,
        "output_stats": output_stats,
    }
    save_json(meta, META_STATS_JSON)


# ============================================================
# Optuna objective for BASE model
# ============================================================

def objective_factory_base(x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, bias_delta_t, device):
    def objective(trial):
        best_params = {
            "width": trial.suggest_int("width", 64, 256, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-8, 1e-3, log=True),
            "batch": trial.suggest_categorical("batch", [32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 120, 300),
            "clip": trial.suggest_float("clip", 0.5, 5.0, log=True),

            "w_data": trial.suggest_float("w_data", 0.5, 5.0, log=True),
            "w_fp": 0.0,
            "w_mono": 0.0,
            "w_ineq": 0.0,

            # level0 不实际使用，但保留键更稳
            "rho_abs_min": 0.25,
            "mono_topk": 40,
        }

        model, _ = train_with_params(
            best_params=best_params,
            level=LEVEL,
            x_tr=x_tr,
            y_tr=y_tr,
            x_va=x_va,
            y_va=y_va,
            Xtr_np=Xtr_np,
            Ytr_np=Ytr_np,
            bias_delta_t=bias_delta_t,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            mu_va, logvar_va = model(x_va)
            val = gaussian_nll(y_va, mu_va, logvar_va).item()

        return float(val)

    return objective


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(ART_DIR)
    device = get_device()

    split = load_fixed_split()
    split_s = scale_split(split)
    ten = make_tensors(split_s, device)

    save_meta_stats(split)

    print("\n================= OPTUNA BASE =================")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )

    study.optimize(
        objective_factory_base(
            x_tr=ten["x_tr"],
            y_tr=ten["y_tr"],
            x_va=ten["x_va"],
            y_va=ten["y_va"],
            Xtr_np=split_s["Xtr_s"],
            Ytr_np=split_s["Ytr_s"],
            bias_delta_t=ten["bias_delta_t"],
            device=device,
        ),
        n_trials=N_TRIALS,
    )

    best_params = study.best_params.copy()
    best_params.update({
        "w_fp": 0.0,
        "w_mono": 0.0,
        "w_ineq": 0.0,
        "rho_abs_min": 0.25,
        "mono_topk": 40,
    })

    print(f"[BEST BASE] value={study.best_value:.6g}")
    print(best_params)

    save_json({
        "level": LEVEL,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "run_tag": RUN_TAG,
        "fixed_surrogate": True,
        "split_meta_path": str(SPLIT_META_JSON),
    }, BEST_JSON)

    # -------------------------
    # Retrain with best params
    # -------------------------
    model, mono_pairs = train_with_params(
        best_params=best_params,
        level=LEVEL,
        x_tr=ten["x_tr"],
        y_tr=ten["y_tr"],
        x_va=ten["x_va"],
        y_va=ten["y_va"],
        Xtr_np=split_s["Xtr_s"],
        Ytr_np=split_s["Ytr_s"],
        bias_delta_t=ten["bias_delta_t"],
        device=device,
    )

    # -------------------------
    # Test evaluation
    # -------------------------
    model.eval()
    with torch.no_grad():
        mu_te_s, logvar_te = model(ten["x_te"])
        var_te_s = torch.exp(logvar_te)

    mu_te_s_np = mu_te_s.detach().cpu().numpy()
    y_te_s_np = ten["y_te"].detach().cpu().numpy()
    sigma_te = np.sqrt(var_te_s.detach().cpu().numpy()) * split_s["sy"].scale_

    mu_te = split_s["sy"].inverse_transform(mu_te_s_np)
    y_te_true = split_s["sy"].inverse_transform(y_te_s_np)

    basic = compute_basic_metrics(y_te_true, mu_te)
    prob90 = compute_prob_metrics_gaussian(y_te_true, mu_te, sigma_te, alpha=0.10)
    viol = eval_inequality_violation(mu_te, OUTPUT_COLS)
    test_nll = float(gaussian_nll(ten["y_te"], mu_te_s, logvar_te).item())

    sanity_rows = compute_output_sanity(y_te_true, mu_te, OUTPUT_COLS)
    save_json(sanity_rows, SANITY_JSON)
    pd.DataFrame(sanity_rows).to_csv(SANITY_CSV, index=False, encoding="utf-8-sig")

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
        "n_train": int(split["X_tr"].shape[0]),
        "n_val": int(split["X_va"].shape[0]),
        "n_test": int(split["X_te"].shape[0]),
    }
    save_json(metrics, METRICS_JSON)

    pred_dump = {
        "level": LEVEL,
        "run_tag": RUN_TAG,
        "fixed_surrogate": True,
        "y_true": y_te_true.tolist(),
        "mu": mu_te.tolist(),
        "sigma": sigma_te.tolist(),
        "output_names": OUTPUT_COLS,
    }
    save_json(pred_dump, TEST_PRED_JSON)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "best_params": best_params,
        "level": LEVEL,
        "run_tag": RUN_TAG,
        "fixed_surrogate": True,
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
        "mono_pairs": mono_pairs,
    }
    torch.save(ckpt, CKPT_PT)

    with open(SCALERS_PKL, "wb") as f:
        pickle.dump({"sx": split_s["sx"], "sy": split_s["sy"]}, f)

    per_dim_rows = []
    for j, name in enumerate(OUTPUT_COLS):
        per_dim_rows.append({
            "level": LEVEL,
            "run_tag": RUN_TAG,
            "output": name,
            "iter": "iter1" if j < 8 else "iter2",
            "group": "primary" if name in PRIMARY_OUTPUTS else "secondary",
            "MAE": float(basic["MAE"][j]),
            "RMSE": float(basic["RMSE"][j]),
            "R2": float(basic["R2"][j]),
            "PICP90": float(prob90["PICP"][j]),
            "MPIW90": float(prob90["MPIW"][j]),
            "CRPS": float(prob90["CRPS"][j]),
        })
    pd.DataFrame(per_dim_rows).to_csv(PER_DIM_CSV, index=False, encoding="utf-8-sig")

    focus_rows = []
    for name in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(name)
        focus_rows.append({
            "level": LEVEL,
            "run_tag": RUN_TAG,
            "output": name,
            "MAE": float(basic["MAE"][j]),
            "RMSE": float(basic["RMSE"][j]),
            "R2": float(basic["R2"][j]),
            "PICP90": float(prob90["PICP"][j]),
            "MPIW90": float(prob90["MPIW"][j]),
            "CRPS": float(prob90["CRPS"][j]),
        })
    pd.DataFrame(focus_rows).to_csv(FOCUS_CSV, index=False, encoding="utf-8-sig")

    print("\n[OK] Finished fixed base surrogate training.")
    print("Saved:")
    print(" -", BEST_JSON)
    print(" -", CKPT_PT)
    print(" -", SCALERS_PKL)
    print(" -", META_STATS_JSON)
    print(" -", METRICS_JSON)
    print(" -", PER_DIM_CSV)
    print(" -", FOCUS_CSV)
    print(" -", TEST_PRED_JSON)
    print(" -", SANITY_JSON)
    print(" -", SANITY_CSV)


if __name__ == "__main__":
    main()