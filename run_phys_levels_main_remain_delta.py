# run_phys_levels_main_remain_delta.py
# ============================================================
# Main training / evaluation script for paper experiments
# ============================================================

import os
import json
import math
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from paper_experiment_config import (
    DATA_ROOT, CSV_PATH, OUT_DIR, SEED, TRIALS, DEVICE,
    INPUT_COLS, OUTPUT_COLS, OUT1, OUT2,ITER2_IDX, ITER1_IDX,
    PRIMARY_OUTPUTS, PRIMARY_STRESS_OUTPUT, PRIMARY_AUXILIARY_OUTPUT,
    PAPER_LEVELS, PARAM_META
)


# ============================================================
# Utilities
# ============================================================
def build_iter_index_maps(output_cols):
    iter1_cols = [c for c in output_cols if c.startswith("iteration1_")]
    iter2_cols = [c for c in output_cols if c.startswith("iteration2_")]

    idx_map = {c: i for i, c in enumerate(output_cols)}
    iter1_idx = [idx_map[c] for c in iter1_cols]
    iter2_idx = [idx_map[c] for c in iter2_cols]

    return idx_map, iter1_cols, iter2_cols, iter1_idx, iter2_idx

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def summarize(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        out[c] = {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v) + 1e-12),
        }
    return out


def print_stats(title: str, st: Dict[str, Dict[str, float]]):
    print(f"\n==== {title} ====")
    for k, v in st.items():
        print(f"{k:30s} min={v['min']:.6g} max={v['max']:.6g} mean={v['mean']:.6g} std={v['std']:.6g}")


def gaussian_nll(y, mu, logvar, eps=1e-8):
    var = torch.exp(logvar).clamp(min=eps)
    return 0.5 * (math.log(2 * math.pi) + logvar + (y - mu) ** 2 / var).mean()


def huber_to_zero(x, delta=1.0):
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def erf_np(x):
    sign = np.sign(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * np.abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def gaussian_crps(mu, sigma, y, eps=1e-12):
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)
    Phi = 0.5 * (1.0 + erf_np(z / np.sqrt(2.0)))
    crps = sigma * (z * (2 * Phi - 1.0) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return crps


def compute_basic_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    r2 = np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])], dtype=float)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def compute_prob_metrics_gaussian(y_true, mu, sigma, alpha=0.10):
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054}
    z = z_map.get(alpha, 1.6448536269514722)
    lo = mu - z * sigma
    hi = mu + z * sigma
    cover = ((y_true >= lo) & (y_true <= hi)).mean(axis=0)
    mpiw = (hi - lo).mean(axis=0)
    crps = gaussian_crps(mu, sigma, y_true)
    crps_mean = np.mean(crps, axis=0)
    return {"PICP": cover, "MPIW": mpiw, "CRPS": crps_mean}

def compute_output_sanity(y_true, y_pred, output_cols, near_constant_std_tol=1e-8):
    """
    Per-output sanity check to detect:
      - near-constant outputs
      - tiny variance dimensions where R2 is not meaningful
      - prediction range mismatch
    """
    rows = []
    for j, name in enumerate(output_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        yt_std = float(np.std(yt))
        yp_std = float(np.std(yp))
        yt_min = float(np.min(yt))
        yt_max = float(np.max(yt))
        yp_min = float(np.min(yp))
        yp_max = float(np.max(yp))

        rows.append({
            "output": name,
            "y_true_mean": float(np.mean(yt)),
            "y_true_std": yt_std,
            "y_true_min": yt_min,
            "y_true_max": yt_max,
            "y_pred_mean": float(np.mean(yp)),
            "y_pred_std": yp_std,
            "y_pred_min": yp_min,
            "y_pred_max": yp_max,
            "is_near_constant_target": bool(yt_std < near_constant_std_tol),
        })
    return rows

def eval_inequality_violation(y_pred, output_cols):
    idx_map = {c: i for i, c in enumerate(output_cols)}
    def viol_rate(cond):
        return float(np.mean(cond))

    v = {}
    v["iter1_maxfuel_ge_avgfuel_viol_rate"] = viol_rate(
        y_pred[:, idx_map["iteration1_max_fuel_temp"]] < y_pred[:, idx_map["iteration1_avg_fuel_temp"]]
    )
    v["iter2_maxfuel_ge_avgfuel_viol_rate"] = viol_rate(
        y_pred[:, idx_map["iteration2_max_fuel_temp"]] < y_pred[:, idx_map["iteration2_avg_fuel_temp"]]
    )
    v["iter1_stress_nonneg_viol_rate"] = viol_rate(
        y_pred[:, idx_map["iteration1_max_global_stress"]] < 0.0
    )
    v["iter2_stress_nonneg_viol_rate"] = viol_rate(
        y_pred[:, idx_map["iteration2_max_global_stress"]] < 0.0
    )
    return v


@torch.no_grad()
def predict_mc_dropout(model, x, T=30):
    model.train()
    mus, vars_ale = [], []
    for _ in range(T):
        mu, logvar = model(x)
        mus.append(mu.unsqueeze(0))
        vars_ale.append(torch.exp(logvar).unsqueeze(0))
    mus = torch.cat(mus, dim=0)
    vars_ale = torch.cat(vars_ale, dim=0)
    mu_mean = mus.mean(dim=0)
    epi_var = mus.var(dim=0, unbiased=False)
    ale_var = vars_ale.mean(dim=0)
    total_var = epi_var + ale_var
    model.eval()
    return mu_mean, total_var, epi_var, ale_var


# ============================================================
# Model
# ============================================================

class HeteroMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = width
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(d, out_dim)
        self.logvar = nn.Linear(d, out_dim)

        self._delta_head = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_z=False):
        z = self.backbone(x)
        mu = self.mu(z)
        logvar = self.logvar(z).clamp(-20, 5)
        if return_z:
            return mu, logvar, z
        return mu, logvar


# ============================================================
# Losses
# ============================================================

def loss_level1_shifted(mu, bias_delta_t):
    return torch.tensor(0.0, device=mu.device)


def loss_level1_band_shift(mu, x, model, eps_band):
    return torch.tensor(0.0, device=mu.device)


def logvar_floor_regularizer(logvar, floor=-10.0):
    return F.relu(floor - logvar).mean()


def build_mono_pairs_spearman(Xtr, Ytr, rho_abs_min, topk):
    Xrk = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, Xtr)
    Yrk = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, Ytr)

    pairs = []
    for i in range(Xtr.shape[1]):
        xi = (Xrk[:, i] - Xrk[:, i].mean()) / (Xrk[:, i].std() + 1e-12)
        for j in range(Ytr.shape[1]):
            yj = (Yrk[:, j] - Yrk[:, j].mean()) / (Yrk[:, j].std() + 1e-12)
            rho = float(np.mean(xi * yj))
            if abs(rho) < rho_abs_min:
                continue
            sign = +1 if rho >= 0 else -1
            pairs.append((i, j, sign, abs(rho)))
    pairs.sort(key=lambda t: t[3], reverse=True)
    return pairs[:topk]


def build_mono_pairs_bootstrap(Xtr, Ytr, rho_abs_min, topk, B, sample_frac, stable_min):
    n = Xtr.shape[0]
    m = int(max(16, sample_frac * n))
    counts, mags, seen = {}, {}, {}
    rng = np.random.RandomState(SEED)

    for _ in range(B):
        idx = rng.choice(n, size=m, replace=True)
        pairs = build_mono_pairs_spearman(Xtr[idx], Ytr[idx], rho_abs_min=rho_abs_min, topk=topk * 5)
        for i, j, sign, w in pairs:
            key = (i, j)
            seen[key] = seen.get(key, 0) + 1
            counts[key] = counts.get(key, 0) + (1 if sign > 0 else -1)
            mags[key] = mags.get(key, 0.0) + float(w)

    out = []
    for (i, j), t in seen.items():
        c = counts[(i, j)]
        stable = abs(c) / float(t)
        if stable < stable_min:
            continue
        sign = +1 if c >= 0 else -1
        w = (mags[(i, j)] / float(t)) * stable
        out.append((i, j, sign, float(w)))

    out.sort(key=lambda t: t[3], reverse=True)
    return out[:topk]


def loss_level2_monotone_from_mu(mu, x, pairs):
    if not pairs:
        return torch.tensor(0.0, device=x.device)
    terms = []
    for i, j, sign, w in pairs:
        yj = mu[:, j].sum()
        gij = torch.autograd.grad(yj, x, create_graph=True, retain_graph=True)[0][:, i]
        viol = F.relu(-sign * gij)
        terms.append(float(w) * viol.mean())
    return torch.stack(terms).mean()


def loss_level3_ineq(mu, output_cols):
    idx_map = {c: i for i, c in enumerate(output_cols)}
    relu_mean = lambda x: F.relu(x).mean()
    l = 0.0
    l += relu_mean(mu[:, idx_map["iteration1_avg_fuel_temp"]] - mu[:, idx_map["iteration1_max_fuel_temp"]])
    l += relu_mean(mu[:, idx_map["iteration2_avg_fuel_temp"]] - mu[:, idx_map["iteration2_max_fuel_temp"]])
    l += relu_mean(-mu[:, idx_map["iteration1_max_global_stress"]])
    l += relu_mean(-mu[:, idx_map["iteration2_max_global_stress"]])
    return l


# ============================================================
# Data
# ============================================================

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=INPUT_COLS + OUTPUT_COLS).reset_index(drop=True)

    for c in INPUT_COLS + OUTPUT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in dataset: {c}")

    return df


def split_and_scale(df):
    idx_all = np.arange(len(df))
    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)

    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.15, random_state=SEED, shuffle=True
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.1765, random_state=SEED, shuffle=True
    )

    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    idx_test = np.sort(idx_test)

    X_tr, X_va, X_te = X[idx_train], X[idx_val], X[idx_test]
    Y_tr, Y_va, Y_te = Y[idx_train], Y[idx_val], Y[idx_test]

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s, Xva_s, Xte_s = sx.transform(X_tr), sx.transform(X_va), sx.transform(X_te)
    Ytr_s, Yva_s, Yte_s = sy.transform(Y_tr), sy.transform(Y_va), sy.transform(Y_te)

    return {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "X_tr": X_tr, "X_va": X_va, "X_te": X_te,
        "Y_tr": Y_tr, "Y_va": Y_va, "Y_te": Y_te,
        "Xtr_s": Xtr_s, "Xva_s": Xva_s, "Xte_s": Xte_s,
        "Ytr_s": Ytr_s, "Yva_s": Yva_s, "Yte_s": Yte_s,
        "sx": sx, "sy": sy,
    }

def save_current_split(df, split, out_dir):
    split_dir = os.path.join(out_dir, "saved_split_from_run_phys_levels_main")
    ensure_dir(split_dir)

    idx_train = split["idx_train"]
    idx_val = split["idx_val"]
    idx_test = split["idx_test"]

    pd.DataFrame({"index": idx_train}).to_csv(
        os.path.join(split_dir, "train_indices.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"index": idx_val}).to_csv(
        os.path.join(split_dir, "val_indices.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"index": idx_test}).to_csv(
        os.path.join(split_dir, "test_indices.csv"), index=False, encoding="utf-8-sig"
    )

    df.iloc[idx_train].reset_index(drop=True).to_csv(
        os.path.join(split_dir, "train.csv"), index=False, encoding="utf-8-sig"
    )
    df.iloc[idx_val].reset_index(drop=True).to_csv(
        os.path.join(split_dir, "val.csv"), index=False, encoding="utf-8-sig"
    )
    df.iloc[idx_test].reset_index(drop=True).to_csv(
        os.path.join(split_dir, "test.csv"), index=False, encoding="utf-8-sig"
    )

    meta = {
        "csv_path": CSV_PATH,
        "seed": SEED,
        "n_total": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
    }
    with open(os.path.join(split_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved split to: {split_dir}")
# ============================================================
# Optuna objective
# ============================================================

def objective_factory(level, x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, bias_delta_t, device):
    def objective(trial):
        width = trial.suggest_int("width", 64, 256, log=True)
        depth = trial.suggest_int("depth", 3, 8)
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e-3, log=True)
        batch = trial.suggest_categorical("batch", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 120, 300)
        clip = trial.suggest_float("clip", 0.5, 5.0, log=True)

        w_data = trial.suggest_float("w_data", 0.5, 5.0, log=True)
        w_fp = trial.suggest_float("w_fp", 1e-3, 5.0, log=True) if level >= 1 else 0.0
        w_mono = trial.suggest_float("w_mono", 1e-3, 10.0, log=True) if level >= 2 else 0.0
        w_ineq = trial.suggest_float("w_ineq", 1e-4, 5.0, log=True) if level >= 3 else 0.0

        rho_min = trial.suggest_float("rho_abs_min", 0.10, 0.55) if level >= 2 else 0.25
        topk = trial.suggest_int("mono_topk", 10, 120) if level >= 2 else 40

        use_boot = False
        boot_B = 0
        boot_frac = 0.7
        boot_stable_min = 0.8

        if level >= 4:
            use_boot = trial.suggest_categorical("use_boot", [True, False])
            if use_boot:
                boot_B = trial.suggest_int("boot_B", 8, 40)
                boot_frac = trial.suggest_float("boot_frac", 0.45, 0.90)
                boot_stable_min = trial.suggest_float("boot_stable_min", 0.70, 0.98)

        if level >= 4 and use_boot:
            mono_pairs = build_mono_pairs_bootstrap(
                Xtr_np, Ytr_np, rho_abs_min=float(rho_min), topk=int(topk),
                B=int(boot_B), sample_frac=float(boot_frac), stable_min=float(boot_stable_min)
            )
        elif level >= 2:
            mono_pairs = build_mono_pairs_spearman(
                Xtr_np, Ytr_np, rho_abs_min=float(rho_min), topk=int(topk)
            )
        else:
            mono_pairs = []

        w_shift = trial.suggest_float("w_shift", 1e-3, 10.0, log=True) if level >= 4 else 0.0
        eps_band = trial.suggest_float("eps_band", 0.00, 0.80) if level >= 4 else 0.0
        w_logvar = trial.suggest_float("w_logvar", 1e-5, 5e-2, log=True) if level >= 4 else 0.0
        logvar_floor = trial.suggest_float("logvar_floor", -14.0, -6.0) if level >= 4 else -10.0

        model = HeteroMLP(x_tr.shape[1], y_tr.shape[1], width, depth, dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        if level >= 4 and model._delta_head is None:
            model._delta_head = nn.Sequential(
                nn.Linear(x_tr.shape[1], 64),
                nn.SiLU(),
                nn.Linear(64, len(ITER2_IDX))
            ).to(device)
            for m in model._delta_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            opt.add_param_group({"params": model._delta_head.parameters()})

        n = x_tr.shape[0]
        best = 1e18
        bad = 0
        patience = 25

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)

            for s in range(0, n, batch):
                b = perm[s:s + batch]
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
                    loss = loss + w_shift * loss_level1_band_shift(mu, xb_req, model, eps_band=float(eps_band))
                    loss = loss + w_logvar * logvar_floor_regularizer(logvar, floor=float(logvar_floor))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()

            model.eval()
            with torch.no_grad():
                mu_va, logvar_va = model(x_va)
                val = gaussian_nll(y_va, mu_va, logvar_va).item()

            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val < best - 1e-6:
                best = val
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        return best

    return objective


# ============================================================
# Retrain with best params
# ============================================================

def train_with_params(best_params, level, x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, bias_delta_t, device):
    width = int(best_params["width"])
    depth = int(best_params["depth"])
    dropout = float(best_params["dropout"])
    lr = float(best_params["lr"])
    wd = float(best_params["wd"])
    batch = int(best_params["batch"])
    epochs = int(best_params["epochs"])
    clip = float(best_params.get("clip", 2.0))

    w_data = float(best_params.get("w_data", 1.0))
    w_fp = float(best_params.get("w_fp", 0.0))
    w_mono = float(best_params.get("w_mono", 0.0))
    w_ineq = float(best_params.get("w_ineq", 0.0))

    rho_min = float(best_params.get("rho_abs_min", 0.25))
    topk = int(best_params.get("mono_topk", 40))

    use_boot = bool(best_params.get("use_boot", False))
    if level >= 4 and use_boot:
        boot_B = int(best_params.get("boot_B", 16))
        boot_frac = float(best_params.get("boot_frac", 0.7))
        boot_stable_min = float(best_params.get("boot_stable_min", 0.8))
        mono_pairs = build_mono_pairs_bootstrap(
            Xtr_np, Ytr_np, rho_abs_min=float(rho_min), topk=int(topk),
            B=int(boot_B), sample_frac=float(boot_frac), stable_min=float(boot_stable_min)
        )
    elif level >= 2:
        mono_pairs = build_mono_pairs_spearman(
            Xtr_np, Ytr_np, rho_abs_min=float(rho_min), topk=int(topk)
        )
    else:
        mono_pairs = []

    w_shift = float(best_params.get("w_shift", 0.0))
    eps_band = float(best_params.get("eps_band", 0.0))
    w_logvar = float(best_params.get("w_logvar", 0.0))
    logvar_floor = float(best_params.get("logvar_floor", -10.0))

    model = HeteroMLP(x_tr.shape[1], y_tr.shape[1], width, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if level >= 4 and model._delta_head is None:
        model._delta_head = nn.Sequential(
            nn.Linear(x_tr.shape[1], 64),
            nn.SiLU(),
            nn.Linear(64, len(ITER2_IDX))
        ).to(device)
        for m in model._delta_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        opt.add_param_group({"params": model._delta_head.parameters()})

    n = x_tr.shape[0]
    best = 1e18
    best_state = None
    bad = 0
    patience = 25

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)

        for s in range(0, n, batch):
            b = perm[s:s + batch]
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
                loss = loss + w_shift * loss_level1_band_shift(mu, xb_req, model, eps_band=float(eps_band))
                loss = loss + w_logvar * logvar_floor_regularizer(logvar, floor=float(logvar_floor))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_va, logvar_va = model(x_va)
            val = gaussian_nll(y_va, mu_va, logvar_va).item()

        if val < best - 1e-6:
            best = val
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, mono_pairs


# ============================================================
# Main
# ============================================================

def main():
    seed_all(SEED)
    ensure_dir(OUT_DIR)
    device = get_device()

    df = load_dataset()
    idx_map, iter1_cols, iter2_cols, iter1_idx, iter2_idx = build_iter_index_maps(OUTPUT_COLS)

    print("\n==== OUTPUT GROUP CHECK ====")
    print("iter1_cols:", iter1_cols)
    print("iter2_cols:", iter2_cols)
    print("iter1_idx :", iter1_idx)
    print("iter2_idx :", iter2_idx)

    in_stats = summarize(df, INPUT_COLS)
    out_stats = summarize(df, OUTPUT_COLS)
    print_stats("INPUT STATS", in_stats)
    print_stats("OUTPUT STATS", out_stats)

    with open(os.path.join(OUT_DIR, "meta_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"PARAM_META": PARAM_META, "input_stats": in_stats, "output_stats": out_stats}, f, indent=2)

    split = split_and_scale(df)
    save_current_split(df, split, OUT_DIR)
    sx, sy = split["sx"], split["sy"]

    bias_delta = np.zeros(len(iter2_idx), dtype=float)

    x_tr = torch.tensor(split["Xtr_s"], dtype=torch.float32, device=device)
    y_tr = torch.tensor(split["Ytr_s"], dtype=torch.float32, device=device)
    x_va = torch.tensor(split["Xva_s"], dtype=torch.float32, device=device)
    y_va = torch.tensor(split["Yva_s"], dtype=torch.float32, device=device)
    x_te = torch.tensor(split["Xte_s"], dtype=torch.float32, device=device)
    y_te = torch.tensor(split["Yte_s"], dtype=torch.float32, device=device)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    results_summary_rows = []

    for level in PAPER_LEVELS:
        print(f"\n================= OPTUNA Level {level} =================")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        )

        study.optimize(
            objective_factory(
                level=level,
                x_tr=x_tr,
                y_tr=y_tr,
                x_va=x_va,
                y_va=y_va,
                Xtr_np=split["Xtr_s"],
                Ytr_np=split["Ytr_s"],
                bias_delta_t=bias_delta_t,
                device=device,
            ),
            n_trials=TRIALS
        )

        best_params = study.best_params
        print(f"[BEST L{level}] value={study.best_value:.6g}")
        print(best_params)

        with open(os.path.join(OUT_DIR, f"best_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_value": float(study.best_value),
                "best_params": best_params,
            }, f, indent=2)

        model, mono_pairs = train_with_params(
            best_params=best_params,
            level=level,
            x_tr=x_tr,
            y_tr=y_tr,
            x_va=x_va,
            y_va=y_va,
            Xtr_np=split["Xtr_s"],
            Ytr_np=split["Ytr_s"],
            bias_delta_t=bias_delta_t,
            device=device,
        )

        with torch.no_grad():
            mu_te_s, logvar_te = model(x_te)
            var_te_s = torch.exp(logvar_te)

        mu_te_s_np = _to_numpy(mu_te_s)
        y_te_s_np = _to_numpy(y_te)
        sigma_te = np.sqrt(_to_numpy(var_te_s)) * sy.scale_

        mu_te = sy.inverse_transform(mu_te_s_np)
        y_te_true = sy.inverse_transform(y_te_s_np)

        basic = compute_basic_metrics(y_te_true, mu_te)
        prob90 = compute_prob_metrics_gaussian(y_te_true, mu_te, sigma_te, alpha=0.10)
        viol = eval_inequality_violation(mu_te, OUTPUT_COLS)
        test_nll = float(gaussian_nll(y_te, mu_te_s, logvar_te).item())

        sanity_rows = compute_output_sanity(y_te_true, mu_te, OUTPUT_COLS)

        with open(os.path.join(OUT_DIR, f"sanity_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(sanity_rows, f, indent=2, ensure_ascii=False)

        pd.DataFrame(sanity_rows).to_csv(
            os.path.join(OUT_DIR, f"sanity_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        primary_idx = [OUTPUT_COLS.index(c) for c in PRIMARY_OUTPUTS]

        metrics = {
            "level": level,
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
            "mono_pairs_top": [
                {
                    "x": INPUT_COLS[i],
                    "y": OUTPUT_COLS[j],
                    "direction": "increasing" if sign > 0 else "decreasing",
                    "abs_rho": float(w),
                }
                for (i, j, sign, w) in mono_pairs[:50]
            ],
            "focus_outputs": {
                PRIMARY_STRESS_OUTPUT: {
                    "MAE": float(basic["MAE"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                    "RMSE": float(basic["RMSE"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                    "R2": float(basic["R2"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                    "PICP90": float(prob90["PICP"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                    "MPIW90": float(prob90["MPIW"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                    "CRPS": float(prob90["CRPS"][OUTPUT_COLS.index(PRIMARY_STRESS_OUTPUT)]),
                },
                PRIMARY_AUXILIARY_OUTPUT: {
                    "MAE": float(basic["MAE"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                    "RMSE": float(basic["RMSE"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                    "R2": float(basic["R2"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                    "PICP90": float(prob90["PICP"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                    "MPIW90": float(prob90["MPIW"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                    "CRPS": float(prob90["CRPS"][OUTPUT_COLS.index(PRIMARY_AUXILIARY_OUTPUT)]),
                },
            },
        }

        with open(os.path.join(OUT_DIR, f"metrics_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Save raw predictions
        pred_dump = {
            "y_true": y_te_true.tolist(),
            "mu": mu_te.tolist(),
            "sigma": sigma_te.tolist(),
            "output_names": OUTPUT_COLS,
            "level": level,
        }
        with open(os.path.join(OUT_DIR, f"test_predictions_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(pred_dump, f, indent=2, ensure_ascii=False)

        # Save checkpoint + scalers
        ckpt = {
            "model_state_dict": model.state_dict(),
            "best_params": best_params,
            "level": level,
            "input_cols": INPUT_COLS,
            "output_cols": OUTPUT_COLS,
        }
        torch.save(ckpt, os.path.join(OUT_DIR, f"checkpoint_level{level}.pt"))

        with open(os.path.join(OUT_DIR, f"scalers_level{level}.pkl"), "wb") as f:
            pickle.dump({"sx": sx, "sy": sy}, f)

        # Save per-dim CSV
        per_dim_rows = []
        for j, name in enumerate(OUTPUT_COLS):
            per_dim_rows.append({
                "level": level,
                "output": name,
                "group": "primary" if name in PRIMARY_OUTPUTS else "secondary",
                "iter": "iter1" if j < 8 else "iter2",
                "MAE": float(basic["MAE"][j]),
                "RMSE": float(basic["RMSE"][j]),
                "R2": float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS": float(prob90["CRPS"][j]),
            })
        pd.DataFrame(per_dim_rows).to_csv(
            os.path.join(OUT_DIR, f"paper_metrics_per_dim_level{level}.csv"),
            index=False, encoding="utf-8-sig"
        )
        focus_rows = []
        for name in [PRIMARY_STRESS_OUTPUT, PRIMARY_AUXILIARY_OUTPUT]:
            j = OUTPUT_COLS.index(name)
            focus_rows.append({
                "level": level,
                "output": name,
                "MAE": float(basic["MAE"][j]),
                "RMSE": float(basic["RMSE"][j]),
                "R2": float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS": float(prob90["CRPS"][j]),
            })

        pd.DataFrame(focus_rows).to_csv(
            os.path.join(OUT_DIR, f"paper_focus_metrics_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        results_summary_rows.append({
            "level": level,
            "test_nll_std": test_nll,
            "RMSE_mean_primary": metrics["basic_primary_mean"]["RMSE_mean"],
            "R2_mean_primary": metrics["basic_primary_mean"]["R2_mean"],
            "PICP90_mean_primary": metrics["prob90_primary_mean"]["PICP_mean"],
            "MPIW90_mean_primary": metrics["prob90_primary_mean"]["MPIW_mean"],
            "CRPS_mean_primary": metrics["prob90_primary_mean"]["CRPS_mean"],

            "stress_RMSE": metrics["focus_outputs"][PRIMARY_STRESS_OUTPUT]["RMSE"],
            "stress_R2": metrics["focus_outputs"][PRIMARY_STRESS_OUTPUT]["R2"],
            "stress_PICP90": metrics["focus_outputs"][PRIMARY_STRESS_OUTPUT]["PICP90"],

            "keff_RMSE": metrics["focus_outputs"][PRIMARY_AUXILIARY_OUTPUT]["RMSE"],
            "keff_R2": metrics["focus_outputs"][PRIMARY_AUXILIARY_OUTPUT]["R2"],
            "keff_PICP90": metrics["focus_outputs"][PRIMARY_AUXILIARY_OUTPUT]["PICP90"],
        })

        print(
            f"[METRICS L{level}] "
            f"NLL={test_nll:.6g}  "
            f"RMSE_primary={metrics['basic_primary_mean']['RMSE_mean']:.4g}  "
            f"PICP90_primary={metrics['prob90_primary_mean']['PICP_mean']:.3f}"
        )

    pd.DataFrame(results_summary_rows).to_csv(
        os.path.join(OUT_DIR, "paper_metrics_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[OK] Finished main training / evaluation.")


if __name__ == "__main__":
    main()