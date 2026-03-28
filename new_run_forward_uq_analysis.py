# run_forward_uq_analysis.py
# ============================================================
# Forward UQ analysis for paper
#
# Main purpose:
#   1) load frozen fixed-surrogate artifacts for baseline / regularized models
#   2) propagate input uncertainty using the SAME input samples across models
#   3) export mu-only summaries (mean mapping only)
#   4) export predictive summaries (including surrogate predictive variance)
#   5) export threshold-based stress failure probabilities
#   6) export stress-keff joint samples
#   7) export CVR comparison metrics
#
# Notes:
#   - level 0 -> fixed_surrogate_fixed_base
#   - level 2 -> fixed_surrogate_fixed_level2
#   - input bounds are shared and loaded from available fixed-surrogate meta_stats
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
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    THRESHOLD_SWEEP,
    PAPER_LEVELS,
    SEED,
)
from run_phys_levels_main import HeteroMLP, get_device


# ============================================================
# User settings
# ============================================================

N_SAMPLES = 20000
DRAW_PREDICTIVE_SAMPLES = True
LEVELS_TO_RUN = [lv for lv in PAPER_LEVELS if lv in [0, 2]]

# input sampling mode:
#   "uniform_meta" -> use min/max from meta_stats.json
INPUT_SAMPLING_MODE = "uniform_meta"

META_STATS_CANDIDATES = [
    os.path.join(OUT_DIR, "fixed_surrogate_fixed_level2", "meta_stats.json"),
    os.path.join(OUT_DIR, "fixed_surrogate_fixed_base", "meta_stats.json"),
]

PRIMARY_QS = [0.05, 0.25, 0.50, 0.75, 0.95]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    return bounds, meta_path


def sample_inputs_uniform_meta(n: int, bounds, rng: np.random.RandomState):
    x = np.zeros((n, len(bounds)), dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        x[:, j] = rng.uniform(lo, hi, size=n)
    return x


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
def predict_mu_sigma_raw(model, sx, sy, x_raw: np.ndarray, device):
    x_s = sx.transform(x_raw)
    xt = torch.tensor(x_s, dtype=torch.float32, device=device)

    mu_s, logvar_s = model(xt)

    mu_s_np = mu_s.detach().cpu().numpy()
    logvar_s_np = logvar_s.detach().cpu().numpy()

    mu_raw = sy.inverse_transform(mu_s_np)
    sigma_raw = np.sqrt(np.exp(logvar_s_np)) * sy.scale_

    return mu_raw, sigma_raw


def maybe_draw_predictive(mu_raw: np.ndarray, sigma_raw: np.ndarray, rng: np.random.RandomState):
    return rng.normal(loc=mu_raw, scale=np.maximum(sigma_raw, 1e-12))


def summarize_series(x: np.ndarray):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def coefficient_of_variation_ratio(x: np.ndarray):
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if abs(mu) < 1e-12:
        return np.nan
    return abs(sd / mu)


def export_all_outputs_table(y_vals: np.ndarray, level: int, tag: str):
    rows = []
    for j, out in enumerate(OUTPUT_COLS):
        st = summarize_series(y_vals[:, j])
        rows.append({"output": out, **st})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_all_outputs_{tag}_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_primary_outputs_table(y_vals: np.ndarray, level: int, tag: str):
    rows = []
    for out in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(out)
        st = summarize_series(y_vals[:, j])
        rows.append({"output": out, **st})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_primary_outputs_{tag}_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_joint_stress_keff(y_vals: np.ndarray, level: int, tag: str):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    j_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    df = pd.DataFrame({
        PRIMARY_STRESS_OUTPUT: y_vals[:, j_stress],
        PRIMARY_AUXILIARY_OUTPUT: y_vals[:, j_keff],
    })
    path = os.path.join(OUT_DIR, f"forward_uq_joint_stress_keff_{tag}_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_failure_prob(y_vals: np.ndarray, level: int, tag: str):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    stress = y_vals[:, j_stress]

    rows = []
    for thr in THRESHOLD_SWEEP:
        rows.append({
            "level": level,
            "tag": tag,
            "threshold_MPa": float(thr),
            "n_samples": int(len(stress)),
            "failure_probability": float(np.mean(stress > thr)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_failure_prob_{tag}_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_cvr(y_vals: np.ndarray, level: int, tag: str):
    rows = []
    for out in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(out)
        vals = y_vals[:, j]
        rows.append({
            "level": level,
            "tag": tag,
            "output": out,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "CVR": float(coefficient_of_variation_ratio(vals)),
        })

    df = pd.DataFrame(rows)
    path_csv = os.path.join(OUT_DIR, f"forward_uq_cvr_{tag}_level{level}.csv")
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    summary = {
        "level": int(level),
        "tag": tag,
        "overall_primary_CVR": float(np.nanmean(df["CVR"].values)),
        "outputs": df.to_dict(orient="records"),
    }
    path_json = os.path.join(OUT_DIR, f"forward_uq_cvr_summary_{tag}_level{level}.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return df, path_csv, summary, path_json


def build_summary_row(level: int, y_vals: np.ndarray, tag: str):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    j_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    stress = y_vals[:, j_stress]
    keff = y_vals[:, j_keff]

    overall_primary_cvr = []
    for out in PRIMARY_OUTPUTS:
        vals = y_vals[:, OUTPUT_COLS.index(out)]
        overall_primary_cvr.append(coefficient_of_variation_ratio(vals))
    overall_primary_cvr = float(np.nanmean(overall_primary_cvr))

    row = {
        "level": int(level),
        "tag": tag,
        "n_samples": int(len(y_vals)),
        "stress_mean": float(np.mean(stress)),
        "stress_std": float(np.std(stress)),
        "stress_q95": float(np.quantile(stress, 0.95)),
        "keff_mean": float(np.mean(keff)),
        "keff_std": float(np.std(keff)),
        "p_fail_110": float(np.mean(stress > 110.0)),
        "p_fail_120": float(np.mean(stress > 120.0)),
        "p_fail_131": float(np.mean(stress > 131.0)),
        "overall_primary_CVR": overall_primary_cvr,
    }
    return row


def merge_mu_pred_rows(mu_row: dict, pred_row: dict):
    return {
        "level": mu_row["level"],
        "n_samples": mu_row["n_samples"],
        "draw_predictive_samples": True,

        "stress_mean_mu": mu_row["stress_mean"],
        "stress_std_mu": mu_row["stress_std"],
        "stress_q95_mu": mu_row["stress_q95"],
        "keff_mean_mu": mu_row["keff_mean"],
        "keff_std_mu": mu_row["keff_std"],
        "p_fail_110_mu": mu_row["p_fail_110"],
        "p_fail_120_mu": mu_row["p_fail_120"],
        "p_fail_131_mu": mu_row["p_fail_131"],
        "overall_primary_CVR_mu": mu_row["overall_primary_CVR"],

        "stress_mean_pred": pred_row["stress_mean"],
        "stress_std_pred": pred_row["stress_std"],
        "stress_q95_pred": pred_row["stress_q95"],
        "keff_mean_pred": pred_row["keff_mean"],
        "keff_std_pred": pred_row["keff_std"],
        "p_fail_110_pred": pred_row["p_fail_110"],
        "p_fail_120_pred": pred_row["p_fail_120"],
        "p_fail_131_pred": pred_row["p_fail_131"],
        "overall_primary_CVR_pred": pred_row["overall_primary_CVR"],

        "delta_stress_mean_pred_minus_mu": pred_row["stress_mean"] - mu_row["stress_mean"],
        "delta_stress_std_pred_minus_mu": pred_row["stress_std"] - mu_row["stress_std"],
        "delta_stress_q95_pred_minus_mu": pred_row["stress_q95"] - mu_row["stress_q95"],
        "delta_keff_mean_pred_minus_mu": pred_row["keff_mean"] - mu_row["keff_mean"],
        "delta_keff_std_pred_minus_mu": pred_row["keff_std"] - mu_row["keff_std"],
        "delta_p_fail_110_pred_minus_mu": pred_row["p_fail_110"] - mu_row["p_fail_110"],
        "delta_p_fail_120_pred_minus_mu": pred_row["p_fail_120"] - mu_row["p_fail_120"],
        "delta_p_fail_131_pred_minus_mu": pred_row["p_fail_131"] - mu_row["p_fail_131"],
        "delta_overall_primary_CVR_pred_minus_mu": pred_row["overall_primary_CVR"] - mu_row["overall_primary_CVR"],
    }


def main():
    ensure_dir(OUT_DIR)
    seed_all(SEED)

    device = get_device()
    print(f"[INFO] device = {device}")

    bounds, bounds_meta_path = load_input_bounds()
    print(f"[INFO] input bounds loaded from: {bounds_meta_path}")

    if INPUT_SAMPLING_MODE != "uniform_meta":
        raise ValueError(f"Unsupported INPUT_SAMPLING_MODE: {INPUT_SAMPLING_MODE}")

    # IMPORTANT: same input samples for all levels
    rng_x = np.random.RandomState(SEED + 1)
    x_samples = sample_inputs_uniform_meta(N_SAMPLES, bounds, rng_x)

    input_path = os.path.join(OUT_DIR, "forward_uq_input_samples.csv")
    pd.DataFrame(x_samples, columns=INPUT_COLS).to_csv(
        input_path, index=False, encoding="utf-8-sig"
    )

    summary_rows = []

    for level in LEVELS_TO_RUN:
        print(f"\n[INFO] Running forward UQ for level {level}")

        ckpt, sx, sy, ckpt_path, scaler_path = load_checkpoint_and_scalers(level)
        print(f"[INFO] checkpoint = {ckpt_path}")
        print(f"[INFO] scaler     = {scaler_path}")

        model = build_model_from_ckpt(ckpt, device)

        mu_raw, sigma_raw = predict_mu_sigma_raw(model, sx, sy, x_samples, device)

        if DRAW_PREDICTIVE_SAMPLES:
            rng_y = np.random.RandomState(SEED + 100 * level + 2)
            y_pred = maybe_draw_predictive(mu_raw, sigma_raw, rng_y)
        else:
            y_pred = mu_raw.copy()

        # save compressed draws
        npz_path = os.path.join(OUT_DIR, f"forward_uq_samples_level{level}.npz")
        np.savez_compressed(
            npz_path,
            x_samples=x_samples,
            mu_raw=mu_raw,
            sigma_raw=sigma_raw,
            y_draw=y_pred,
            input_cols=np.array(INPUT_COLS, dtype=object),
            output_cols=np.array(OUTPUT_COLS, dtype=object),
        )

        # -------- mu-only exports --------
        export_all_outputs_table(mu_raw, level, tag="mu")
        export_primary_outputs_table(mu_raw, level, tag="mu")
        export_failure_prob(mu_raw, level, tag="mu")
        export_joint_stress_keff(mu_raw, level, tag="mu")
        export_cvr(mu_raw, level, tag="mu")
        mu_row = build_summary_row(level, mu_raw, tag="mu")

        # -------- predictive exports --------
        export_all_outputs_table(y_pred, level, tag="pred")
        export_primary_outputs_table(y_pred, level, tag="pred")
        export_failure_prob(y_pred, level, tag="pred")
        export_joint_stress_keff(y_pred, level, tag="pred")
        export_cvr(y_pred, level, tag="pred")
        pred_row = build_summary_row(level, y_pred, tag="pred")

        # merged summary row for paper
        row = merge_mu_pred_rows(mu_row, pred_row)
        summary_rows.append(row)

        print(f"[OK] level {level} completed.")
        print(json.dumps(row, indent=2, ensure_ascii=False))

    # combined summary
    df_summary = pd.DataFrame(summary_rows).sort_values("level").reset_index(drop=True)
    paper_summary_path = os.path.join(OUT_DIR, "paper_forward_uq_summary.csv")
    df_summary.to_csv(paper_summary_path, index=False, encoding="utf-8-sig")

    print("\n[DONE] Forward UQ analysis completed.")
    print(f"[DONE] Saved paper summary to: {paper_summary_path}")


if __name__ == "__main__":
    main()