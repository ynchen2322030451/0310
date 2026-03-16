# run_ood_evaluation.py
# ============================================================
# Tail OOD evaluation:
# train on middle 80% of one key feature
# test on two tails 20%
# compare Level0 vs Level4
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
    OOD_FEATURE, OOD_KEEP_MIDDLE_RATIO, SEED,
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


def make_ood_split_by_feature(X, Y, feature_name, keep_middle_ratio=0.8):
    feat_idx = INPUT_COLS.index(feature_name)
    x_feat = X[:, feat_idx]

    q_low = (1.0 - keep_middle_ratio) / 2.0
    q_high = 1.0 - q_low

    lo = np.quantile(x_feat, q_low)
    hi = np.quantile(x_feat, q_high)

    in_mask = (x_feat >= lo) & (x_feat <= hi)
    ood_mask = ~in_mask

    X_in = X[in_mask]
    Y_in = Y[in_mask]
    X_ood = X[ood_mask]
    Y_ood = Y[ood_mask]

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

    delta_tr = Ytr_s[:, 8:16] - Ytr_s[:, 0:8]
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
            rho_abs_min=rho_min,
            topk=topk,
            B=int(best_params.get("boot_B", 16)),
            sample_frac=float(best_params.get("boot_frac", 0.7)),
            stable_min=float(best_params.get("boot_stable_min", 0.8)),
        )
    elif level >= 2:
        mono_pairs = build_mono_pairs_spearman(
            Xtr_s, Ytr_s,
            rho_abs_min=rho_min,
            topk=topk,
        )
    else:
        mono_pairs = []

    w_data = float(best_params.get("w_data", 1.0))
    w_fp = float(best_params.get("w_fp", 0.0))
    w_mono = float(best_params.get("w_mono", 0.0))
    w_ineq = float(best_params.get("w_ineq", 0.0))
    w_shift = float(best_params.get("w_shift", 0.0))
    w_logvar = float(best_params.get("w_logvar", 0.0))
    eps_band = float(best_params.get("eps_band", 0.0))
    logvar_floor = float(best_params.get("logvar_floor", -10.0))
    batch = int(best_params["batch"])
    epochs = int(best_params["epochs"])
    clip = float(best_params.get("clip", 2.0))

    best_val = 1e18
    best_state = None
    bad = 0
    patience = 25

    n = x_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)

        for s in range(0, n, batch):
            b = perm[s:s+batch]
            xb = x_tr[b]
            yb = y_tr[b]

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


def main():
    df = load_dataset()
    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)

    X_in, X_ood, Y_in, Y_ood, meta = make_ood_split_by_feature(
        X, Y, OOD_FEATURE, OOD_KEEP_MIDDLE_RATIO
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

    device = get_device()
    rows = []

    for level in [0, 2]:
        best_json = os.path.join(OUT_DIR, f"best_level{level}.json")
        if not os.path.exists(best_json):
            raise FileNotFoundError(
                f"Missing best json: {best_json}\n"
                f"Please run `python run_phys_levels_main.py` first."
            )

        with open(best_json, "r", encoding="utf-8") as f:
            best_obj = json.load(f)
        best_params = best_obj["best_params"]

        model = train_with_fixed_params(level, best_params, Xtr_s, Ytr_s, Xva_s, Yva_s, device)

        x_ood = torch.tensor(Xood_s, dtype=torch.float32, device=device)
        y_ood = torch.tensor(Yood_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            mu_ood_s, logvar_ood = model(x_ood)
            var_ood_s = torch.exp(logvar_ood)

        mu_ood = sy.inverse_transform(_to_numpy(mu_ood_s))
        y_ood_true = sy.inverse_transform(_to_numpy(y_ood))
        sigma_ood = np.sqrt(_to_numpy(var_ood_s)) * sy.scale_

        basic = compute_basic_metrics(y_ood_true, mu_ood)
        prob90 = compute_prob_metrics_gaussian(y_ood_true, mu_ood, sigma_ood, alpha=0.10)

        idx_stress = OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)
        idx_keff = OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)

        row = {
            "level": level,
            "ood_feature": meta["feature"],
            "n_ood": meta["n_ood"],

            # appendix-style overall mean
            "MAE_mean_all": float(np.mean(basic["MAE"])),
            "RMSE_mean_all": float(np.mean(basic["RMSE"])),
            "R2_mean_all": float(np.mean(basic["R2"])),
            "PICP90_mean_all": float(np.mean(prob90["PICP"])),
            "MPIW90_mean_all": float(np.mean(prob90["MPIW"])),
            "CRPS_mean_all": float(np.mean(prob90["CRPS"])),

            # paper-focus outputs
            "stress_MAE": float(basic["MAE"][idx_stress]),
            "stress_RMSE": float(basic["RMSE"][idx_stress]),
            "stress_R2": float(basic["R2"][idx_stress]),
            "stress_PICP90": float(prob90["PICP"][idx_stress]),
            "stress_MPIW90": float(prob90["MPIW"][idx_stress]),
            "stress_CRPS": float(prob90["CRPS"][idx_stress]),

            "keff_MAE": float(basic["MAE"][idx_keff]),
            "keff_RMSE": float(basic["RMSE"][idx_keff]),
            "keff_R2": float(basic["R2"][idx_keff]),
            "keff_PICP90": float(prob90["PICP"][idx_keff]),
            "keff_MPIW90": float(prob90["MPIW"][idx_keff]),
            "keff_CRPS": float(prob90["CRPS"][idx_keff]),
        }

        rows.append(row)
        per_dim_rows = []
        for j, name in enumerate(OUTPUT_COLS):
            per_dim_rows.append({
                "level": level,
                "ood_feature": meta["feature"],
                "output": name,
                "MAE": float(basic["MAE"][j]),
                "RMSE": float(basic["RMSE"][j]),
                "R2": float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS": float(prob90["CRPS"][j]),
            })

        pd.DataFrame(per_dim_rows).to_csv(
            os.path.join(OUT_DIR, f"paper_ood_per_dim_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[OK] OOD done for level {level}")

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "paper_ood_results.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        with open(os.path.join(OUT_DIR, "paper_ood_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[DONE] OOD evaluation completed.")


if __name__ == "__main__":
    main()