# run_sobol_analysis.py
# ============================================================
# Sobol analysis using trained best model
# Focus on main paper outputs:
#   - iteration2_max_global_stress
#   - iteration2_keff
# ============================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

from paper_experiment_config import (
    OUT_DIR, INPUT_COLS, OUTPUT_COLS, PAPER_LEVELS, PRIMARY_SA_OUTPUTS, SEED
)
from run_phys_levels_main import HeteroMLP, get_device


def load_checkpoint_and_scalers(level):
    ckpt_path = os.path.join(OUT_DIR, f"checkpoint_level{level}.pt")
    scaler_path = os.path.join(OUT_DIR, f"scalers_level{level}.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return ckpt, scalers["sx"], scalers["sy"]


def sample_uniform_from_training_ranges(n, stats_json_path):
    with open(stats_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    input_stats = meta["input_stats"]

    X = np.zeros((n, len(INPUT_COLS)), dtype=float)
    for i, c in enumerate(INPUT_COLS):
        lo = input_stats[c]["min"]
        hi = input_stats[c]["max"]
        X[:, i] = np.random.uniform(lo, hi, size=n)
    return X


def build_model_from_ckpt(ckpt, device):
    best_params = ckpt["best_params"]
    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(best_params["width"]),
        depth=int(best_params["depth"]),
        dropout=float(best_params["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_mu(model, sx, sy, X_raw, device):
    Xs = sx.transform(X_raw)
    x = torch.tensor(Xs, dtype=torch.float32, device=device)
    mu_s, _ = model(x)
    mu = sy.inverse_transform(mu_s.detach().cpu().numpy())
    return mu


def sobol_jansen(model, sx, sy, output_name, N, device):
    meta_path = os.path.join(OUT_DIR, "meta_stats.json")
    A = sample_uniform_from_training_ranges(N, meta_path)
    B = sample_uniform_from_training_ranges(N, meta_path)

    idx_out = OUTPUT_COLS.index(output_name)

    YA = predict_mu(model, sx, sy, A, device)[:, idx_out]
    YB = predict_mu(model, sx, sy, B, device)[:, idx_out]

    VY = np.var(np.concatenate([YA, YB]), ddof=1) + 1e-12

    rows = []
    for i, name in enumerate(INPUT_COLS):
        ABi = A.copy()
        ABi[:, i] = B[:, i]

        YABi = predict_mu(model, sx, sy, ABi, device)[:, idx_out]

        # Jansen estimators
        S1 = 1.0 - np.mean((YB - YABi) ** 2) / (2.0 * VY)
        ST = np.mean((YA - YABi) ** 2) / (2.0 * VY)

        rows.append({
            "output": output_name,
            "input": name,
            "S1": float(S1),
            "ST": float(ST),
        })

    return rows


def main():
    np.random.seed(SEED)
    device = get_device()

    # only main comparison levels for paper
    levels_for_sa = [0, 2]

    all_rows = []
    for level in levels_for_sa:
        ckpt, sx, sy = load_checkpoint_and_scalers(level)
        model = build_model_from_ckpt(ckpt, device)

        for output_name in PRIMARY_SA_OUTPUTS:
            rows = sobol_jansen(
                model=model,
                sx=sx,
                sy=sy,
                output_name=output_name,
                N=512,   # can increase later
                device=device,
            )
            for r in rows:
                r["level"] = level
            all_rows.extend(rows)

        print(f"[OK] Sobol done for level {level}")

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "paper_sobol_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved Sobol results: {out_csv}")


if __name__ == "__main__":
    main()