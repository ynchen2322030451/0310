# run_compare_fixed_models.py
# ============================================================
# Compare fixed BASE surrogate vs fixed LEVEL2 surrogate
# on the SAME frozen test split.
#
# Main outputs:
#   - experiments_phys_levels/paper_fixed_model_compare_summary.csv
#   - experiments_phys_levels/paper_fixed_model_compare_primary.csv
#   - experiments_phys_levels/paper_fixed_model_compare_per_output.csv
#   - experiments_phys_levels/paper_fixed_model_compare_summary.json
# ============================================================

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
    SEED,
)
from run_phys_levels_main import (
    HeteroMLP,
    get_device,
    compute_basic_metrics,
    compute_prob_metrics_gaussian,
    gaussian_nll,
)

# ============================================================
# Settings
# ============================================================

ROOT_OUT = Path(OUT_DIR)
SPLIT_DIR = ROOT_OUT / "fixed_split"

TEST_CSV = SPLIT_DIR / "test.csv"

ARTIFACTS = {
    "base": ROOT_OUT / "fixed_surrogate_fixed_base",
    "level2": ROOT_OUT / "fixed_surrogate_fixed_level2",
}

SUMMARY_CSV = ROOT_OUT / "paper_fixed_model_compare_summary.csv"
PRIMARY_CSV = ROOT_OUT / "paper_fixed_model_compare_primary.csv"
PER_OUTPUT_CSV = ROOT_OUT / "paper_fixed_model_compare_per_output.csv"
SUMMARY_JSON = ROOT_OUT / "paper_fixed_model_compare_summary.json"

IGNORE_OUTPUTS = {
    "iteration1_keff",  # 你已经明确说了忽略它
}

# ============================================================
# Utilities
# ============================================================

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_test_data():
    require_file(TEST_CSV)
    df = pd.read_csv(TEST_CSV)

    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)
    return df, X, Y


def load_artifact(tag: str):
    art_dir = ARTIFACTS[tag]
    require_file(art_dir)

    # 自动匹配 checkpoint / scaler / best_json
    ckpt_candidates = list(art_dir.glob("checkpoint_level*.pt"))
    scaler_candidates = list(art_dir.glob("scalers_level*.pkl"))
    best_candidates = list(art_dir.glob("best_level*.json"))

    if len(ckpt_candidates) != 1:
        raise RuntimeError(f"{tag}: expected exactly 1 checkpoint_level*.pt, got {len(ckpt_candidates)}")
    if len(scaler_candidates) != 1:
        raise RuntimeError(f"{tag}: expected exactly 1 scalers_level*.pkl, got {len(scaler_candidates)}")
    if len(best_candidates) != 1:
        raise RuntimeError(f"{tag}: expected exactly 1 best_level*.json, got {len(best_candidates)}")

    ckpt_path = ckpt_candidates[0]
    scaler_path = scaler_candidates[0]
    best_json_path = best_candidates[0]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    with open(best_json_path, "r", encoding="utf-8") as f:
        best_obj = json.load(f)

    return {
        "artifact_dir": str(art_dir),
        "ckpt_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "best_json_path": str(best_json_path),
        "ckpt": ckpt,
        "sx": scalers["sx"],
        "sy": scalers["sy"],
        "best_obj": best_obj,
    }


def build_model_from_ckpt(ckpt: dict, device):
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
def predict_raw(model, sx, sy, X_raw: np.ndarray, device):
    Xs = sx.transform(X_raw)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=device)

    mu_s, logvar_s = model(Xt)

    mu_s_np = mu_s.detach().cpu().numpy()
    logvar_s_np = logvar_s.detach().cpu().numpy()

    mu_raw = sy.inverse_transform(mu_s_np)
    sigma_raw = np.sqrt(np.exp(logvar_s_np)) * sy.scale_

    return mu_raw, sigma_raw, mu_s, logvar_s


def make_group(name: str) -> str:
    if name in PRIMARY_OUTPUTS:
        return "primary"
    return "secondary"


def aggregate_metrics_for_subset(per_output_df: pd.DataFrame, output_names: list):
    sub = per_output_df[per_output_df["output"].isin(output_names)].copy()
    if len(sub) == 0:
        return {
            "n_outputs": 0,
            "MAE_mean": np.nan,
            "RMSE_mean": np.nan,
            "R2_mean": np.nan,
            "PICP90_mean": np.nan,
            "MPIW90_mean": np.nan,
            "CRPS_mean": np.nan,
        }

    return {
        "n_outputs": int(len(sub)),
        "MAE_mean": float(sub["MAE"].mean()),
        "RMSE_mean": float(sub["RMSE"].mean()),
        "R2_mean": float(sub["R2"].mean()),
        "PICP90_mean": float(sub["PICP90"].mean()),
        "MPIW90_mean": float(sub["MPIW90"].mean()),
        "CRPS_mean": float(sub["CRPS"].mean()),
    }


def main():
    seed_all(SEED)
    device = get_device()
    print(f"[INFO] device = {device}")

    _, X_test, Y_test = load_test_data()

    all_per_output_rows = []
    summary_rows = []
    primary_rows = []
    summary_json_obj = {
        "device": str(device),
        "test_csv": str(TEST_CSV),
        "artifacts": {},
        "model_rank_hint": {},
    }

    comparable_outputs = [o for o in OUTPUT_COLS if o not in IGNORE_OUTPUTS]
    comparable_primary = [o for o in PRIMARY_OUTPUTS if o not in IGNORE_OUTPUTS]

    for tag in ["base", "level2"]:
        print(f"\n[INFO] evaluating {tag}")

        art = load_artifact(tag)
        model = build_model_from_ckpt(art["ckpt"], device)

        mu_raw, sigma_raw, mu_s, logvar_s = predict_raw(
            model=model,
            sx=art["sx"],
            sy=art["sy"],
            X_raw=X_test,
            device=device,
        )

        # standardized targets for NLL
        Y_test_s = art["sy"].transform(Y_test)
        Y_test_s_t = torch.tensor(Y_test_s, dtype=torch.float32, device=device)
        test_nll = float(gaussian_nll(Y_test_s_t, mu_s, logvar_s).item())

        basic = compute_basic_metrics(Y_test, mu_raw)
        prob90 = compute_prob_metrics_gaussian(Y_test, mu_raw, sigma_raw, alpha=0.10)

        per_output_rows = []
        for j, out in enumerate(OUTPUT_COLS):
            row = {
                "model_tag": tag,
                "output": out,
                "iter": "iter1" if j < 8 else "iter2",
                "group": make_group(out),
                "ignored_in_main_compare": bool(out in IGNORE_OUTPUTS),
                "MAE": float(basic["MAE"][j]),
                "RMSE": float(basic["RMSE"][j]),
                "R2": float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS": float(prob90["CRPS"][j]),
                "y_true_std": float(np.std(Y_test[:, j])),
                "y_pred_std": float(np.std(mu_raw[:, j])),
            }
            per_output_rows.append(row)
            all_per_output_rows.append(row)

        df_tag = pd.DataFrame(per_output_rows)

        # overall comparable outputs
        agg_all = aggregate_metrics_for_subset(df_tag, comparable_outputs)
        agg_primary = aggregate_metrics_for_subset(df_tag, comparable_primary)

        summary_rows.append({
            "model_tag": tag,
            "scope": "all_comparable_outputs",
            "test_nll_standardized": test_nll,
            **agg_all,
        })

        summary_rows.append({
            "model_tag": tag,
            "scope": "primary_outputs_only",
            "test_nll_standardized": test_nll,
            **agg_primary,
        })

        for out in comparable_primary:
            r = df_tag[df_tag["output"] == out].iloc[0]
            primary_rows.append({
                "model_tag": tag,
                "output": out,
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "R2": float(r["R2"]),
                "PICP90": float(r["PICP90"]),
                "MPIW90": float(r["MPIW90"]),
                "CRPS": float(r["CRPS"]),
                "y_true_std": float(r["y_true_std"]),
                "y_pred_std": float(r["y_pred_std"]),
            })

        summary_json_obj["artifacts"][tag] = {
            "artifact_dir": art["artifact_dir"],
            "ckpt_path": art["ckpt_path"],
            "scaler_path": art["scaler_path"],
            "best_json_path": art["best_json_path"],
            "best_value": art["best_obj"].get("best_value", None),
            "best_params": art["best_obj"].get("best_params", {}),
        }

    # save detailed per-output table
    df_per = pd.DataFrame(all_per_output_rows).sort_values(["output", "model_tag"]).reset_index(drop=True)
    df_per.to_csv(PER_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # save summary
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # save primary-only table
    df_primary = pd.DataFrame(primary_rows).sort_values(["output", "model_tag"]).reset_index(drop=True)
    df_primary.to_csv(PRIMARY_CSV, index=False, encoding="utf-8-sig")

    # very simple winner hints
    winner_hint = {}
    for out in comparable_primary:
        tmp = df_primary[df_primary["output"] == out].copy()
        tmp = tmp.sort_values(["MAE", "RMSE", "CRPS"], ascending=[True, True, True])
        winner_hint[out] = {
            "recommended_by_error": tmp.iloc[0]["model_tag"],
            "rows": tmp.to_dict(orient="records"),
        }

    summary_json_obj["model_rank_hint"] = winner_hint

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json_obj, f, indent=2, ensure_ascii=False)

    print("\n[DONE] model comparison finished.")
    print(f"[DONE] {SUMMARY_CSV}")
    print(f"[DONE] {PRIMARY_CSV}")
    print(f"[DONE] {PER_OUTPUT_CSV}")
    print(f"[DONE] {SUMMARY_JSON}")


if __name__ == "__main__":
    main()