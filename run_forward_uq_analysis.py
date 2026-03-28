# run_forward_uq_analysis.py
# ============================================================
# Forward UQ analysis for paper
#
# Main purpose:
#   1) use frozen baseline / regularized surrogate
#   2) propagate input uncertainty
#   3) export primary-output summaries
#   4) export threshold-based stress failure probabilities
#   5) export stress-keff joint samples
#   6) export CVR comparison metrics
#
# Notes:
#   - level 2 is forced to load the fixed surrogate artifacts
#   - level 0 still loads the legacy baseline artifacts
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
    FIXED_CKPT_PATH,
    FIXED_SCALER_PATH,
)
from run_phys_levels_main import HeteroMLP, get_device


# ============================================================
# User settings
# ============================================================

N_SAMPLES = 20000
DRAW_PREDICTIVE_SAMPLES = True
LEVELS_TO_RUN = [lv for lv in PAPER_LEVELS if lv in [0, 2]]

# input sampling mode:
#   "uniform_meta"  -> use min/max from meta_stats.json
INPUT_SAMPLING_MODE = "uniform_meta"

META_STATS_CANDIDATES = [
    os.path.join(OUT_DIR, "fixed_surrogate_fixed_level2", "meta_stats.json"),
    os.path.join(OUT_DIR, "meta_stats.json"),
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


# run_forward_uq_analysis.py 里替换
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


def export_all_outputs_table(y_draw: np.ndarray, level: int):
    rows = []
    for j, out in enumerate(OUTPUT_COLS):
        st = summarize_series(y_draw[:, j])
        rows.append({"output": out, **st})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_all_outputs_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_primary_outputs_table(y_draw: np.ndarray, level: int):
    rows = []
    for out in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(out)
        st = summarize_series(y_draw[:, j])
        rows.append({"output": out, **st})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_primary_outputs_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_joint_stress_keff(y_draw: np.ndarray, level: int):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    j_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    df = pd.DataFrame({
        PRIMARY_STRESS_OUTPUT: y_draw[:, j_stress],
        PRIMARY_AUXILIARY_OUTPUT: y_draw[:, j_keff],
    })
    path = os.path.join(OUT_DIR, f"forward_uq_joint_stress_keff_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_failure_prob(y_draw: np.ndarray, level: int):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    stress = y_draw[:, j_stress]

    rows = []
    for thr in THRESHOLD_SWEEP:
        rows.append({
            "level": level,
            "threshold_MPa": float(thr),
            "n_samples": int(len(stress)),
            "failure_probability": float(np.mean(stress > thr)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"forward_uq_failure_prob_level{level}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, path


def export_cvr(y_draw: np.ndarray, level: int):
    rows = []
    for out in PRIMARY_OUTPUTS:
        j = OUTPUT_COLS.index(out)
        vals = y_draw[:, j]
        rows.append({
            "level": level,
            "output": out,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "CVR": float(coefficient_of_variation_ratio(vals)),
        })

    df = pd.DataFrame(rows)
    path_csv = os.path.join(OUT_DIR, f"forward_uq_cvr_level{level}.csv")
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    summary = {
        "level": int(level),
        "overall_primary_CVR": float(np.nanmean(df["CVR"].values)),
        "outputs": df.to_dict(orient="records"),
    }
    path_json = os.path.join(OUT_DIR, f"forward_uq_cvr_summary_level{level}.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return df, path_csv, summary, path_json


def build_paper_summary(level: int, y_draw: np.ndarray):
    j_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    j_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    stress = y_draw[:, j_stress]
    keff = y_draw[:, j_keff]

    overall_primary_cvr = []
    for out in PRIMARY_OUTPUTS:
        vals = y_draw[:, OUTPUT_COLS.index(out)]
        overall_primary_cvr.append(coefficient_of_variation_ratio(vals))
    overall_primary_cvr = float(np.nanmean(overall_primary_cvr))

    row = {
        "level": int(level),
        "n_samples": int(len(y_draw)),
        "draw_predictive_samples": bool(DRAW_PREDICTIVE_SAMPLES),
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


def main():
    ensure_dir(OUT_DIR)
    seed_all(SEED)

    device = get_device()
    print(f"[INFO] device = {device}")

    bounds, bounds_meta_path = load_input_bounds()
    print(f"[INFO] input bounds loaded from: {bounds_meta_path}")

    summary_rows = []

    for level in LEVELS_TO_RUN:
        print(f"\n[INFO] Running forward UQ for level {level}")

        ckpt, sx, sy, ckpt_path, scaler_path = load_checkpoint_and_scalers(level)
        print(f"[INFO] checkpoint = {ckpt_path}")
        print(f"[INFO] scaler     = {scaler_path}")

        model = build_model_from_ckpt(ckpt, device)

        rng_x = np.random.RandomState(SEED + 100 * level + 1)
        rng_y = np.random.RandomState(SEED + 100 * level + 2)

        if INPUT_SAMPLING_MODE != "uniform_meta":
            raise ValueError(f"Unsupported INPUT_SAMPLING_MODE: {INPUT_SAMPLING_MODE}")

        x_samples = sample_inputs_uniform_meta(N_SAMPLES, bounds, rng_x)
        mu_raw, sigma_raw = predict_mu_sigma_raw(model, sx, sy, x_samples, device)

        if DRAW_PREDICTIVE_SAMPLES:
            y_draw = maybe_draw_predictive(mu_raw, sigma_raw, rng_y)
        else:
            y_draw = mu_raw.copy()

        # save sampled inputs once for reproducibility
        input_path = os.path.join(OUT_DIR, "forward_uq_input_samples.csv")
        if not os.path.exists(input_path):
            pd.DataFrame(x_samples, columns=INPUT_COLS).to_csv(
                input_path, index=False, encoding="utf-8-sig"
            )

        # save compressed draws
        npz_path = os.path.join(OUT_DIR, f"forward_uq_samples_level{level}.npz")
        np.savez_compressed(
            npz_path,
            x_samples=x_samples,
            mu_raw=mu_raw,
            sigma_raw=sigma_raw,
            y_draw=y_draw,
            input_cols=np.array(INPUT_COLS, dtype=object),
            output_cols=np.array(OUTPUT_COLS, dtype=object),
        )

        # detailed exports
        df_all, path_all = export_all_outputs_table(y_draw, level)
        df_primary, path_primary = export_primary_outputs_table(y_draw, level)
        df_fail, path_fail = export_failure_prob(y_draw, level)
        df_joint, path_joint = export_joint_stress_keff(y_draw, level)
        df_cvr, path_cvr_csv, cvr_summary, path_cvr_json = export_cvr(y_draw, level)

        # paper summary row
        row = build_paper_summary(level, y_draw)
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