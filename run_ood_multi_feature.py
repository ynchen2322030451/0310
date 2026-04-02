# run_ood_multi_feature.py (FIXED)
# from claude
# ============================================================
# Multi-feature OOD evaluation:
# For each target feature, train on middle 80%, test on tails.
# Tests 4 features: alpha_slope (original), E_intercept,
# alpha_base, nu (top Sobol contributors for stress).
# Levels 0 and 2. Results saved as:
#   paper_ood_multi_feature_summary.csv
#   paper_ood_multi_feature_per_dim.csv
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paper_experiment_config import (
    CSV_PATH, OUT_DIR, INPUT_COLS, OUTPUT_COLS,
    OOD_KEEP_MIDDLE_RATIO, SEED,
    PRIMARY_STRESS_OUTPUT, PRIMARY_AUXILIARY_OUTPUT
)
from run_phys_levels_main import (
    load_dataset, HeteroMLP, gaussian_nll, _to_numpy,
    compute_basic_metrics, compute_prob_metrics_gaussian,
    build_mono_pairs_spearman, build_mono_pairs_bootstrap,
    loss_level1_shifted, loss_level1_band_shift,
    loss_level2_monotone_from_mu, loss_level3_ineq,
    logvar_floor_regularizer, get_device
)

# Features to test: original + top Sobol contributors to stress
OOD_FEATURES = [
    "E_intercept",   # dominant Sobol contributor to stress
    "alpha_base",    # second Sobol contributor
    "nu",            # third contributor
    "alpha_slope",   # original (keep for comparison)
]


def make_ood_split_by_feature(X, Y, feature_name, keep_middle_ratio=0.8):
    feat_idx = INPUT_COLS.index(feature_name)
    x_feat = X[:, feat_idx]

    q_low = (1.0 - keep_middle_ratio) / 2.0
    q_high = 1.0 - q_low

    lo = np.quantile(x_feat, q_low)
    hi = np.quantile(x_feat, q_high)

    in_mask = (x_feat >= lo) & (x_feat <= hi)
    ood_mask = ~in_mask

    X_in, Y_in = X[in_mask], Y[in_mask]
    X_ood, Y_ood = X[ood_mask], Y[ood_mask]

    meta = {
        "feature": feature_name,
        "low_quantile_value": float(lo),
        "high_quantile_value": float(hi),
        "n_in": int(X_in.shape[0]),
        "n_ood": int(X_ood.shape[0]),
    }
    return X_in, X_ood, Y_in, Y_ood, meta


def train_with_fixed_params(level, best_params, Xtr_s, Ytr_s, Xva_s, Yva_s, device):
    x_tr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
    x_va = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    y_va = torch.tensor(Yva_s, dtype=torch.float32, device=device)

    # FIX: OUTPUT_COLS structure is OUT1 (7 items) + OUT2 (8 items) = 15 total
    # OUT1 (iter1) indices: 0-6 (no iter1_keff)
    # OUT2 (iter2) indices: 7-14 (includes iter2_keff)
    # Compute delta as iter2 - iter1, padding iter1 with a zero column for iter2_keff
    iter1_outputs = Ytr_s[:, 0:7]   # shape (batch, 7)
    iter2_outputs = Ytr_s[:, 7:15]  # shape (batch, 8)
    # Pad iter1 with zeros at the front to match iter2's 8 dimensions
    iter1_padded = np.pad(iter1_outputs, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    delta_tr = iter2_outputs - iter1_padded  # shape (batch, 8)
    bias_delta_t = torch.tensor(delta_tr.mean(axis=0), dtype=torch.float32, device=device)

    model = HeteroMLP(
        in_dim=len(INPUT_COLS),
        out_dim=len(OUTPUT_COLS),
        width=int(best_params["width"]),
        depth=int(best_params["depth"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["wd"]),
    )

    if level >= 4 and model._delta_head is None:
        model._delta_head = torch.nn.Sequential(
            torch.nn.Linear(len(INPUT_COLS), 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 8)
        ).to(device)
        opt.add_param_group({"params": model._delta_head.parameters()})

    rho_min = float(best_params.get("rho_abs_min", 0.25))
    topk = int(best_params.get("mono_topk", 40))

    if level >= 4 and bool(best_params.get("use_boot", False)):
        mono_pairs = build_mono_pairs_bootstrap(
            Xtr_s, Ytr_s,
            rho_abs_min=rho_min, topk=topk,
            B=int(best_params.get("boot_B", 16)),
            sample_frac=float(best_params.get("boot_frac", 0.7)),
            stable_min=float(best_params.get("boot_stable_min", 0.8)),
        )
    elif level >= 2:
        mono_pairs = build_mono_pairs_spearman(
            Xtr_s, Ytr_s, rho_abs_min=rho_min, topk=topk
        )
    else:
        mono_pairs = []

    w_data  = float(best_params.get("w_data", 1.0))
    w_fp    = float(best_params.get("w_fp", 0.0))
    w_mono  = float(best_params.get("w_mono", 0.0))
    w_ineq  = float(best_params.get("w_ineq", 0.0))
    w_shift = float(best_params.get("w_shift", 0.0))
    w_logvar = float(best_params.get("w_logvar", 0.0))
    eps_band = float(best_params.get("eps_band", 0.0))
    logvar_floor = float(best_params.get("logvar_floor", -10.0))
    batch  = int(best_params["batch"])
    epochs = int(best_params["epochs"])
    clip   = float(best_params.get("clip", 2.0))

    best_val, best_state, bad, patience = 1e18, None, 0, 25
    n = x_tr.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            xb, yb = x_tr[b], y_tr[b]
            xb_req = xb.detach().clone().requires_grad_(True) if (level >= 2 and mono_pairs) else xb
            mu, logvar = model(xb_req)
            loss = w_data * gaussian_nll(yb, mu, logvar)
            if level >= 1:
                loss = loss + w_fp * loss_level1_shifted(mu, bias_delta_t)
            if level >= 2:
                loss = loss + w_mono * loss_level2_monotone_from_mu(mu, xb_req, mono_pairs)
            if level >= 3:
                loss = loss + w_ineq * loss_level3_ineq(mu, OUTPUT_COLS)
            if level >= 4:
                loss = loss + w_shift * loss_level1_band_shift(mu, xb_req, model, eps_band)
                loss = loss + w_logvar * logvar_floor_regularizer(logvar, floor=logvar_floor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_va, logvar_va = model(x_va)
            val = gaussian_nll(y_va, mu_va, logvar_va).item()
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def eval_ood_for_feature(feature_name, df, device):
    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)

    X_in, X_ood, Y_in, Y_ood, meta = make_ood_split_by_feature(
        X, Y, feature_name, OOD_KEEP_MIDDLE_RATIO
    )
    X_tr, X_va, Y_tr, Y_va = train_test_split(X_in, Y_in, test_size=0.1765, random_state=SEED)

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s = sx.transform(X_tr)
    Xva_s = sx.transform(X_va)
    Xood_s = sx.transform(X_ood)
    Ytr_s = sy.transform(Y_tr)
    Yva_s = sy.transform(Y_va)
    Yood_s = sy.transform(Y_ood)

    idx_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
    idx_keff   = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

    rows, per_dim_rows = [], []

    for level in [0, 2]:
        best_json = os.path.join(OUT_DIR, f"best_level{level}.json")
        if not os.path.exists(best_json):
            raise FileNotFoundError(f"Missing best json: {best_json}")
        with open(best_json, "r", encoding="utf-8") as f:
            best_params = json.load(f)["best_params"]

        model = train_with_fixed_params(level, best_params, Xtr_s, Ytr_s, Xva_s, Yva_s, device)

        x_ood_t = torch.tensor(Xood_s, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_s, logvar = model(x_ood_t)
            var_s = torch.exp(logvar)

        mu_ood    = sy.inverse_transform(_to_numpy(mu_s))
        y_true    = sy.inverse_transform(Yood_s)
        sigma_ood = np.sqrt(_to_numpy(var_s)) * sy.scale_

        basic  = compute_basic_metrics(y_true, mu_ood)
        prob90 = compute_prob_metrics_gaussian(y_true, mu_ood, sigma_ood, alpha=0.10)

        rows.append({
            "level": level,
            "ood_feature": feature_name,
            "n_in": meta["n_in"],
            "n_ood": meta["n_ood"],
            "MAE_mean_all":   float(np.mean(basic["MAE"])),
            "RMSE_mean_all":  float(np.mean(basic["RMSE"])),
            "R2_mean_all":    float(np.mean(basic["R2"])),
            "PICP90_mean_all": float(np.mean(prob90["PICP"])),
            "MPIW90_mean_all": float(np.mean(prob90["MPIW"])),
            "CRPS_mean_all":  float(np.mean(prob90["CRPS"])),
            "stress_MAE":    float(basic["MAE"][idx_stress]),
            "stress_RMSE":   float(basic["RMSE"][idx_stress]),
            "stress_R2":     float(basic["R2"][idx_stress]),
            "stress_PICP90": float(prob90["PICP"][idx_stress]),
            "stress_MPIW90": float(prob90["MPIW"][idx_stress]),
            "stress_CRPS":   float(prob90["CRPS"][idx_stress]),
            "keff_MAE":    float(basic["MAE"][idx_keff]),
            "keff_RMSE":   float(basic["RMSE"][idx_keff]),
            "keff_R2":     float(basic["R2"][idx_keff]),
            "keff_PICP90": float(prob90["PICP"][idx_keff]),
            "keff_MPIW90": float(prob90["MPIW"][idx_keff]),
            "keff_CRPS":   float(prob90["CRPS"][idx_keff]),
        })

        for j, name in enumerate(OUTPUT_COLS):
            per_dim_rows.append({
                "level": level,
                "ood_feature": feature_name,
                "output": name,
                "MAE":    float(basic["MAE"][j]),
                "RMSE":   float(basic["RMSE"][j]),
                "R2":     float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS":   float(prob90["CRPS"][j]),
            })

        print(f"  [OK] feature={feature_name}, level={level}, "
              f"stress_R2={basic['R2'][idx_stress]:.3f}, "
              f"stress_PICP90={prob90['PICP'][idx_stress]:.3f}")

    return rows, per_dim_rows, meta


def main():
    df = load_dataset()
    device = get_device()

    all_rows, all_per_dim, all_meta = [], [], {}

    for feat in OOD_FEATURES:
        print(f"\n--- OOD feature: {feat} ---")
        rows, per_dim_rows, meta = eval_ood_for_feature(feat, df, device)
        all_rows.extend(rows)
        all_per_dim.extend(per_dim_rows)
        all_meta[feat] = meta

    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUT_DIR, "paper_ood_multi_feature_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_per_dim).to_csv(
        os.path.join(OUT_DIR, "paper_ood_multi_feature_per_dim.csv"),
        index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(OUT_DIR, "paper_ood_multi_feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    # Print summary table
    df_out = pd.DataFrame(all_rows)
    print("\n=== OOD Multi-Feature Summary (stress R2) ===")
    pivot = df_out.pivot_table(index="ood_feature", columns="level", values="stress_R2")
    print(pivot.to_string())

    print("\n[DONE] Multi-feature OOD evaluation completed.")
    print(f"Saved to: {OUT_DIR}/paper_ood_multi_feature_summary.csv")


if __name__ == "__main__":
    main()