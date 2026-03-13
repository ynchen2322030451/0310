# run_phys_levels.py
# ------------------------------------------------------------
# One-stop runner:
# - hardcoded data root (debug-friendly)
# - auto stats + PARAM_META report
# - levels 0/1/2/3 toggle
# - Optuna auto-tunes: architecture + all loss weights + Level2 thresholds
# - prints eval + saves artifacts
# ------------------------------------------------------------

import os, json, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from paper_experiment_config import (
    PRIMARY_STRESS_THRESHOLD,
    THRESHOLD_SWEEP,
    PRIMARY_OUTPUTS,
    PRIMARY_STRESS_OUTPUT,
    PRIMARY_AUXILIARY_OUTPUT,
    PAPER_LEVELS,
    PRIMARY_SA_OUTPUTS,
    OOD_FEATURE,
    OOD_KEEP_MIDDLE_RATIO,
    ITER1_OUTPUTS,
    ITER2_OUTPUTS,
)

# ============== 0) HARD-CODE YOUR DATA ROOT HERE (only change this) ==============
DATA_ROOT = "/home/tjzs/Documents/fenics_data/fenics_data/txt_extract"
CSV_PATH = os.path.join(DATA_ROOT, "dataset_v3.csv")
# ================================================================================
AUX_PATHS = {
    # # stress summaries (often scalar per-sample)
    # "fuel_nearby_avestress": os.path.join(DATA_ROOT, "fuel_nearby_avestress.txt"),
    # "fuel_nearby_maxstress": os.path.join(DATA_ROOT, "fuel_nearby_maxstress.txt"),
    # "hp_nearby_avestress": os.path.join(DATA_ROOT, "hp_nearby_avestress.txt"),
    # "hp_nearby_maxstress": os.path.join(DATA_ROOT, "hp_nearby_maxstress.txt"),
    # # temperature list / profile
    # "fuel_T_list": os.path.join(DATA_ROOT, "fuel_T_list.txt"),
    # # optional npy
    # "thermal_output_data": os.path.join(DATA_ROOT, "thermal_output_data.npy"),
}

OUT_DIR = "./experiments_phys_levels"  # <-- 输出目录
SEED = 2026
TRIALS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- PARAM META -----------------------------
PARAM_META = {
    "E_slope": {"unit":"Pa/K","meaning":"E(T)=E_slope*T+E_intercept","sigma_frac":0.10},
    "E_intercept": {"unit":"Pa","meaning":"E(T)=E_slope*T+E_intercept","sigma_frac":0.10},
    "nu": {"unit":"-","meaning":"Poisson ratio","sigma_frac":0.10},
    "alpha_base": {"unit":"1/K","meaning":"alpha(T)=alpha_slope*T+alpha_base","sigma_frac":0.10},
    "alpha_slope": {"unit":"1/K^2","meaning":"alpha(T)=alpha_slope*T+alpha_base","sigma_frac":0.10},
    "SS316_T_ref": {"unit":"K","meaning":"T_ref in k(T)=k_slope*(T-T_ref)+k_ref","sigma_frac":0.10},
    "SS316_k_ref": {"unit":"W/(m·K)","meaning":"k_ref in k(T)","sigma_frac":0.10},
    "SS316_alpha": {"unit":"W/(m·K^2)","meaning":"k_slope in k(T) (user clarified)","sigma_frac":0.10},
}

INPUT_COLS = ["E_slope","E_intercept","nu","alpha_base","alpha_slope","SS316_T_ref","SS316_k_ref","SS316_alpha"]
OUT1 = [
    "iteration1_keff","iteration1_avg_fuel_temp","iteration1_max_fuel_temp",
    "iteration1_max_monolith_temp","iteration1_max_global_stress",
    "iteration1_monolith_new_temperature","iteration1_Hcore_after","iteration1_wall2"
]
OUT2 = [
    "iteration2_keff","iteration2_avg_fuel_temp","iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp","iteration2_max_global_stress",
    "iteration2_monolith_new_temperature","iteration2_Hcore_after","iteration2_wall2"
]
OUTPUT_COLS = OUT1 + OUT2
ITER1_IDX = list(range(0,8))
ITER2_IDX = list(range(8,16))

# ----------------------------- Utils -----------------------------
def make_ood_split_by_feature(X, Y, feature_name, keep_middle_ratio=0.8):
    """
    Train on middle region of one feature, test on tails (OOD).
    Example: keep_middle_ratio=0.8 -> middle 80% for train/val, tails 20% for test.
    """
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

    return X_in, X_ood, Y_in, Y_ood, {
        "feature": feature_name,
        "low_quantile_value": float(lo),
        "high_quantile_value": float(hi),
        "n_in": int(X_in.shape[0]),
        "n_ood": int(X_ood.shape[0]),
    }

def seed_all(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def summarize(df:pd.DataFrame, cols:List[str])->Dict[str,Dict[str,float]]:
    out={}
    for c in cols:
        v=df[c].to_numpy(dtype=float)
        out[c]={"min":float(v.min()),"max":float(v.max()),"mean":float(v.mean()),"std":float(v.std()+1e-12)}
    return out

def print_stats(title:str, st:Dict[str,Dict[str,float]]):
    print(f"\n==== {title} ====")
    for k,v in st.items():
        print(f"{k:30s} min={v['min']:.6g} max={v['max']:.6g} mean={v['mean']:.6g} std={v['std']:.6g}")

def gaussian_nll(y, mu, logvar, eps=1e-8):
    var=torch.exp(logvar).clamp(min=eps)
    return 0.5*(math.log(2*math.pi)+logvar+(y-mu)**2/var).mean()

def huber(x, delta=1.0):
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta)

# ============================= Eval metrics (paper-ready) =============================

def compute_failure_metrics_gaussian(mu, sigma, threshold):
    """
    mu, sigma: [N, D] numpy in ORIGINAL scale
    threshold: scalar stress threshold in MPa
    Returns failure probability estimate for iteration2_max_global_stress
    using Gaussian predictive distribution per sample, then averaged.
    """
    from math import erf as math_erf

    stress_idx = OUTPUT_COLS.index("iteration2_max_global_stress")
    mu_s = mu[:, stress_idx]
    sig_s = np.maximum(sigma[:, stress_idx], 1e-12)

    # Per-sample exceedance probability: P(Y > threshold)
    z = (threshold - mu_s) / sig_s
    cdf = 0.5 * (1.0 + np.vectorize(math_erf)(z / np.sqrt(2.0)))
    p_fail_each = 1.0 - cdf
    p_fail = float(np.mean(p_fail_each))

    # Deterministic exceedance based on predictive mean only
    p_fail_mu = float(np.mean(mu_s > threshold))

    return {
        "threshold": float(threshold),
        "p_fail_predictive": p_fail,
        "p_fail_mean_only": p_fail_mu,
        "stress_mu_mean": float(np.mean(mu_s)),
        "stress_mu_std": float(np.std(mu_s)),
        "stress_sigma_mean": float(np.mean(sig_s)),
    }


def compute_threshold_sweep(mu, sigma, thresholds):
    rows = []
    for thr in thresholds:
        res = compute_failure_metrics_gaussian(mu, sigma, thr)
        rows.append(res)
    return rows

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def gaussian_crps(mu, sigma, y, eps=1e-12):
    """
    Closed-form CRPS for Gaussian N(mu, sigma^2)
    CRPS = sigma * [ z*(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ], z=(y-mu)/sigma
    """
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    # phi, Phi
    phi = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5*z*z)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2*Phi - 1.0) + 2*phi - 1.0/np.sqrt(np.pi))
    return crps

def erf(x):
    # numpy-friendly erf
    # use torch if available? keep pure numpy
    # approximation via scipy would be nicer but avoid dependency
    # Abramowitz-Stegun approximation
    sign = np.sign(x)
    a1=0.254829592; a2=-0.284496736; a3=1.421413741; a4=-1.453152027; a5=1.061405429
    p=0.3275911
    t = 1.0/(1.0+p*np.abs(x))
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y

def compute_basic_metrics(y_true, y_pred):
    """
    y_true,y_pred: [N,D] numpy, in ORIGINAL scale.
    """
    print("\n===== DATA SANITY CHECK =====")
    print("y_true range:", np.min(y_true), np.max(y_true))
    print("y_pred range:", np.min(y_pred), np.max(y_pred))
    print("y_true mean/std:", np.mean(y_true), np.std(y_true))
    print("y_pred mean/std:", np.mean(y_pred), np.std(y_pred))
    eps=1e-12
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2, axis=0))

    r2 = []
    for j in range(y_true.shape[1]):

        yt = y_true[:, j]
        yp = y_pred[:, j]

        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - np.mean(yt))**2)

        r2_manual = 1 - ss_res / (ss_tot + 1e-12)
        r2_sklearn = r2_score(yt, yp)

        print("\n========== DIM", j, "==========")
        print("mean(y_true) =", np.mean(yt))
        print("std(y_true)  =", np.std(yt))
        print("mean(y_pred) =", np.mean(yp))
        print("std(y_pred)  =", np.std(yp))
        print("ss_res =", ss_res)
        print("ss_tot =", ss_tot)
        print("R2_manual =", r2_manual)
        print("R2_sklearn =", r2_sklearn)

        r2.append(r2_sklearn)
    r2=np.array(r2, dtype=float)
    print("\n===== DEBUG R2 CHECK =====")
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def compute_prob_metrics_gaussian(y_true, mu, sigma, alpha=0.10):
    """
    y_true, mu, sigma: [N,D] numpy in ORIGINAL scale.
    alpha=0.10 -> 90% PI
    Returns PICP (coverage), MPIW (mean interval width), CRPS (mean)
    """
    # z for central (1-alpha) interval
    # 90% -> z=1.64485; 95% -> 1.95996
    z_map = {0.10: 1.6448536269514722, 0.05: 1.959963984540054}
    z = z_map.get(alpha, 1.6448536269514722)
    lo = mu - z*sigma
    hi = mu + z*sigma
    cover = ((y_true >= lo) & (y_true <= hi)).mean(axis=0)  # per dim
    mpiw = (hi - lo).mean(axis=0)
    # CRPS per sample per dim then mean over samples
    crps = gaussian_crps(mu, sigma, y_true)
    crps_mean = np.mean(crps, axis=0)
    return {"PICP": cover, "MPIW": mpiw, "CRPS": crps_mean}

def eval_inequality_violation(y_pred, idx_map):
    """
    y_pred: [N,D] numpy in ORIGINAL scale.
    Returns violation rates for your Level3 inequalities.
    """
    def viol_rate(cond):
        return float(np.mean(cond))
    v={}
    # max_fuel >= avg_fuel
    v["iter1_maxfuel_ge_avgfuel_viol_rate"] = viol_rate(y_pred[:, idx_map["iteration1_max_fuel_temp"]] < y_pred[:, idx_map["iteration1_avg_fuel_temp"]])
    v["iter2_maxfuel_ge_avgfuel_viol_rate"] = viol_rate(y_pred[:, idx_map["iteration2_max_fuel_temp"]] < y_pred[:, idx_map["iteration2_avg_fuel_temp"]])
    # stress >= 0
    v["iter1_stress_nonneg_viol_rate"] = viol_rate(y_pred[:, idx_map["iteration1_max_global_stress"]] < 0.0)
    v["iter2_stress_nonneg_viol_rate"] = viol_rate(y_pred[:, idx_map["iteration2_max_global_stress"]] < 0.0)
    return v

@torch.no_grad()
def predict_mc_dropout(model, x, T=30):
    """
    MC Dropout for epistemic + aleatoric.
    Returns mu_mean, total_var where:
      total_var = epistemic_var + aleatoric_var_mean
    In STANDARDIZED space.
    """
    model.train()  # enable dropout
    mus=[]
    vars_ale=[]
    for _ in range(T):
        mu, logvar = model(x)
        mus.append(mu.unsqueeze(0))
        vars_ale.append(torch.exp(logvar).unsqueeze(0))
    mus = torch.cat(mus, dim=0)            # [T,N,D]
    vars_ale = torch.cat(vars_ale, dim=0)  # [T,N,D]
    mu_mean = mus.mean(dim=0)              # [N,D]
    epi_var = mus.var(dim=0, unbiased=False)
    ale_var = vars_ale.mean(dim=0)
    total_var = epi_var + ale_var
    model.eval()
    return mu_mean, total_var, epi_var, ale_var

# ----------------------------- Robust TXT loaders -----------------------------
def _parse_numbers_from_line(line:str)->List[float]:
    line=line.strip().replace(",", " ")
    if not line:
        return []
    parts=[p for p in line.split() if p]
    vals=[]
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    return vals

def load_txt_as_array(path:str)->np.ndarray:
    """
    Returns:
      - if file has N lines each with >=1 number: shape [N, K] (K can be 1)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing aux file: {path}")
    rows=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            vals=_parse_numbers_from_line(line)
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric data parsed from: {path}")
    # pad ragged rows to maxlen
    m=max(len(r) for r in rows)
    arr=np.full((len(rows), m), np.nan, dtype=float)
    for i,r in enumerate(rows):
        arr[i,:len(r)]=r
    return arr

def summarize_vector_rows(arr:np.ndarray)->Dict[str,np.ndarray]:
    """
    arr: [N,K] with NaNs for padding
    return per-sample summary vectors
    """
    # ignore NaNs
    mean=np.nanmean(arr, axis=1)
    mx=np.nanmax(arr, axis=1)
    p95=np.nanquantile(arr, 0.95, axis=1)
    return {"mean":mean, "max":mx, "p95":p95}

# ----------------------------- Model -----------------------------
class HeteroMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        layers=[]
        d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d,width), nn.SiLU()]
            if dropout>0: layers += [nn.Dropout(dropout)]
            d=width
        self.backbone=nn.Sequential(*layers)
        self.mu=nn.Linear(d,out_dim)
        self.logvar=nn.Linear(d,out_dim)

        # Level4: learned delta head and aux head (created lazily only if needed)
        self._delta_head = None
        self._aux_head = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, return_z: bool=False):
        z=self.backbone(x)
        mu=self.mu(z)
        logvar=self.logvar(z).clamp(-20,5)
        if return_z:
            return mu, logvar, z
        return mu, logvar
# ----------------------------- Level losses -----------------------------
def loss_level1_shifted(mu, bias_delta_t):
    r = (mu[:, ITER2_IDX] - mu[:, ITER1_IDX]) - bias_delta_t  # broadcast
    return huber(r, delta=1.0)

def loss_level1_band_shift(mu, x, model, eps_band: float):
    delta_hat = model._delta_head(x)  # assume created
    delta = (mu[:, ITER2_IDX] - mu[:, ITER1_IDX])
    r = torch.abs(delta - delta_hat) - eps_band
    return F.relu(r).mean()

def logvar_floor_regularizer(logvar, floor: float = -10.0):
    """
    Prevent 'cheating' by collapsing variances too hard.
    Penalize logvar going below a floor (in standardized output space).
    """
    return F.relu(floor - logvar).mean()

def build_mono_pairs_spearman(Xtr:np.ndarray, Ytr:np.ndarray, rho_abs_min:float, topk:int)->List[Tuple[int,int,int,float]]:
    # rank-based approx
    Xrk=np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(),0,Xtr)
    Yrk=np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(),0,Ytr)
    pairs=[]
    for i in range(Xtr.shape[1]):
        xi=(Xrk[:,i]-Xrk[:,i].mean())/(Xrk[:,i].std()+1e-12)
        for j in range(Ytr.shape[1]):
            yj=(Yrk[:,j]-Yrk[:,j].mean())/(Yrk[:,j].std()+1e-12)
            rho=float(np.mean(xi*yj))
            if abs(rho)<rho_abs_min: continue
            sign=+1 if rho>=0 else -1
            pairs.append((i,j,sign,abs(rho)))
    pairs.sort(key=lambda t:t[3], reverse=True)
    return pairs[:topk]

def build_mono_pairs_bootstrap(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    rho_abs_min: float,
    topk: int,
    B: int,
    sample_frac: float,
    stable_min: float
) -> List[Tuple[int,int,int,float]]:
    """
    Bootstrap-stable monotone pairs.
    Returns: (i, j, sign, weight) where weight ~ stability * mean|rho|
    """
    n = Xtr.shape[0]
    m = int(max(16, sample_frac * n))
    counts = {}   # (i,j) -> signed count accumulator
    mags = {}     # (i,j) -> sum |rho|
    seen = {}     # (i,j) -> times seen above threshold

    rng = np.random.RandomState(2026)

    for _ in range(B):
        idx = rng.choice(n, size=m, replace=True)
        pairs = build_mono_pairs_spearman(Xtr[idx], Ytr[idx], rho_abs_min=rho_abs_min, topk=topk*5)
        for i,j,sign,w in pairs:
            key=(i,j)
            seen[key] = seen.get(key, 0) + 1
            counts[key] = counts.get(key, 0) + (1 if sign>0 else -1)
            mags[key] = mags.get(key, 0.0) + float(w)

    out=[]
    for (i,j), t in seen.items():
        # stability: majority direction ratio
        c = counts[(i,j)]
        stable = abs(c) / float(t)  # in [0,1]
        if stable < stable_min:
            continue
        sign = +1 if c>=0 else -1
        w = (mags[(i,j)]/float(t)) * stable
        out.append((i,j,sign,float(w)))

    out.sort(key=lambda t:t[3], reverse=True)
    return out[:topk]

def loss_level2_monotone_from_mu(mu, x, pairs):
    if not pairs:
        return torch.tensor(0.0, device=x.device)
    # x must require grad
    terms=[]
    for i,j,sign,w in pairs:
        yj = mu[:,j].sum()
        gij = torch.autograd.grad(yj, x, create_graph=True, retain_graph=True)[0][:,i]
        viol = F.relu(-sign*gij)
        terms.append(float(w)*viol.mean())
    return torch.stack(terms).mean()

def loss_level2_monotone(model, x, pairs):
    if not pairs:
        return torch.tensor(0.0, device=x.device)
    x = x.detach().clone().requires_grad_(True)
    mu,_ = model(x)
    terms=[]
    for i,j,sign,w in pairs:
        yj = mu[:,j].sum()
        gij = torch.autograd.grad(yj, x, create_graph=True)[0][:,i]
        viol = F.relu(-sign*gij)
        terms.append(float(w)*viol.mean())
    return torch.stack(terms).mean()

def loss_level3_ineq(mu, idx_map):
    # purely definition-based inequalities, conservative
    # max_fuel_temp >= avg_fuel_temp (iter1, iter2)
    def relu(x): return F.relu(x).mean()
    l=0.0
    # iter1
    l += relu(mu[:, idx_map["iteration1_avg_fuel_temp"]] - mu[:, idx_map["iteration1_max_fuel_temp"]])
    # iter2
    l += relu(mu[:, idx_map["iteration2_avg_fuel_temp"]] - mu[:, idx_map["iteration2_max_fuel_temp"]])
    # stress non-neg
    l += relu(-mu[:, idx_map["iteration1_max_global_stress"]])
    l += relu(-mu[:, idx_map["iteration2_max_global_stress"]])
    return l

# ----------------------------- Objective -----------------------------
def objective_factory(level:int, pack):
    (x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, aux_z_tr, aux_z_va, bias_delta_t) = pack
    idx_map = {c:i for i,c in enumerate(OUTPUT_COLS)}

    def objective(trial):
        width = trial.suggest_int("width", 64, 256, log=True)
        depth = trial.suggest_int("depth", 3, 8)
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e-3, log=True)
        batch = trial.suggest_categorical("batch", [32,64,128])
        epochs = trial.suggest_int("epochs", 120, 300)

        # weights auto-tuned (no manual)
        w_data = trial.suggest_float("w_data", 0.5, 5.0, log=True)
        w_l1 = trial.suggest_float("w_fp", 1e-3, 5.0, log=True) if level>=1 else 0.0
        w_l2 = trial.suggest_float("w_mono", 1e-3, 10.0, log=True) if level>=2 else 0.0
        w_ineq = trial.suggest_float("w_ineq", 1e-4, 5.0, log=True) if level>=3 else 0.0
        w_aux = trial.suggest_float("w_aux", 1e-3, 10.0, log=True) if (level>=3 and aux_z_tr is not None) else 0.0

                # level2 selection thresholds auto-tuned
        rho_min = trial.suggest_float("rho_abs_min", 0.10, 0.55) if level>=2 else 0.25
        topk = trial.suggest_int("mono_topk", 10, 120) if level>=2 else 40

        # -------- Level4 monotone bootstrap stabilization (define BEFORE use) --------
        use_boot = False
        boot_B = 0
        boot_frac = 0.7
        boot_stable_min = 0.8

        if level >= 4:
            use_boot = trial.suggest_categorical("use_boot", [True, False])
            if use_boot:
                boot_B = trial.suggest_int("boot_B", 8, 40)                 # 建议 8~40 足够
                boot_frac = trial.suggest_float("boot_frac", 0.45, 0.90)
                boot_stable_min = trial.suggest_float("boot_stable_min", 0.70, 0.98)

        # -------- Build mono_pairs --------
        if level >= 4 and use_boot:
            mono_pairs = build_mono_pairs_bootstrap(
                Xtr_np, Ytr_np,
                rho_abs_min=float(rho_min),
                topk=int(topk),
                B=int(boot_B),
                sample_frac=float(boot_frac),
                stable_min=float(boot_stable_min)
            )
        elif level >= 2:
            mono_pairs = build_mono_pairs_spearman(
                Xtr_np, Ytr_np,
                rho_abs_min=float(rho_min),
                topk=int(topk)
            )
        else:
            mono_pairs = []

        model = HeteroMLP(x_tr.shape[1], y_tr.shape[1], width, depth, dropout).to(x_tr.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        clip = trial.suggest_float("clip", 0.5, 5.0, log=True)

        # Level4 extras (auto)
        w_shift = trial.suggest_float("w_shift", 1e-3, 10.0, log=True) if level>=4 else 0.0
        eps_band = trial.suggest_float("eps_band", 0.00, 0.80) if level>=4 else 0.0
        w_logvar = trial.suggest_float("w_logvar", 1e-5, 5e-2, log=True) if level>=4 else 0.0
        logvar_floor = trial.suggest_float("logvar_floor", -14.0, -6.0) if level>=4 else -10.0

        

        n = x_tr.shape[0]
        best = 1e18
        bad = 0
        patience = 25

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=x_tr.device)
            for s in range(0,n,batch):
                b = perm[s:s+batch]
                xb = x_tr[b]; yb = y_tr[b]

                # -------- ensure delta head exists for level>=4 --------
                if level >= 4 and model._delta_head is None:
                    model._delta_head = nn.Sequential(
                        nn.Linear(xb.shape[1], 64),
                        nn.SiLU(),
                        nn.Linear(64, len(ITER2_IDX))
                    ).to(x_tr.device)
                    for m in model._delta_head.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
                    opt.add_param_group({"params": model._delta_head.parameters()})

                # -------- single forward (no lost data loss) --------
                xb_req = xb.detach().clone().requires_grad_(True) if (level >= 2 and mono_pairs) else xb
                mu, logvar = model(xb_req)

                # data term (must be included)
                loss = w_data * gaussian_nll(yb, mu, logvar)

                if level>=1:
                    loss = loss + w_l1 * loss_level1_shifted(mu, bias_delta_t)

                if level>=2:
                    loss = loss + w_l2 * loss_level2_monotone_from_mu(mu, xb_req, mono_pairs)

                if level>=3:
                    loss = loss + w_ineq * loss_level3_ineq(mu, idx_map)

                # aux supervision from txt summaries (if available)
                if level>=3 and aux_z_tr is not None:
                    zb = aux_z_tr[b]
                    mu2, logvar2, z = model(xb_req, return_z=True) # reuse features

                    # create aux head on z
                    if model._aux_head is None:
                        model._aux_head = nn.Sequential(
                            nn.Linear(z.shape[1], 64),
                            nn.SiLU(),
                            nn.Linear(64, zb.shape[1])
                        ).to(x_tr.device)
                        for m in model._aux_head.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
                        opt.add_param_group({"params": model._aux_head.parameters()})

                    zhat = model._aux_head(z)
                    loss = loss + w_aux * F.mse_loss(zhat, zb)



                if level>=4:
                    # learned banded shift consistency
                    loss = loss + w_shift * loss_level1_band_shift(mu, xb_req, model, eps_band=float(eps_band))
                    # logvar anti-collapse regularizer
                    loss = loss + w_logvar * logvar_floor_regularizer(logvar, floor=float(logvar_floor))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()

            # val
            model.eval()
            with torch.no_grad():
                mu_va, logvar_va = model(x_va)
                val = gaussian_nll(y_va, mu_va, logvar_va).item()
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val < best - 1e-6:
                best = val; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        return best

    return objective

def train_with_params(best_params, level, pack, max_epochs_override=None):
    """
    Retrain a model using Optuna best_params, matching objective logic as closely as possible.
    - supports level0-4
    - supports bootstrap mono_pairs when use_boot=True
    - supports level4 delta_head + band shift + logvar floor regularizer
    - aux head uses backbone feature z (NOT mu.detach)
    """
    (x_tr, y_tr, x_va, y_va, Xtr_np, Ytr_np, aux_z_tr, aux_z_va, bias_delta_t) = pack
    device = x_tr.device

    # ---- arch + opt ----
    width = int(best_params["width"])
    depth = int(best_params["depth"])
    dropout = float(best_params["dropout"])
    lr = float(best_params["lr"])
    wd = float(best_params["wd"])
    batch = int(best_params["batch"])
    epochs = int(best_params["epochs"])
    if max_epochs_override is not None:
        epochs = int(max_epochs_override)

    model = HeteroMLP(x_tr.shape[1], y_tr.shape[1], width, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    clip = float(best_params.get("clip", 2.0))

    # ---- weights ----
    w_data = float(best_params.get("w_data", 1.0))
    w_fp   = float(best_params.get("w_fp", 0.0))     # level>=1
    w_mono = float(best_params.get("w_mono", 0.0))   # level>=2
    w_ineq = float(best_params.get("w_ineq", 0.0))   # level>=3
    w_aux  = float(best_params.get("w_aux", 0.0))    # level>=3 & aux available

    # level4 extras
    w_shift     = float(best_params.get("w_shift", 0.0))      # level>=4
    eps_band    = float(best_params.get("eps_band", 0.0))     # level>=4
    w_logvar    = float(best_params.get("w_logvar", 0.0))     # level>=4
    logvar_floor= float(best_params.get("logvar_floor", -10.0))

    # ---- build mono_pairs (match objective selection) ----
    rho_min = float(best_params.get("rho_abs_min", 0.25))
    topk    = int(best_params.get("mono_topk", 40))

    mono_pairs = []
    if level >= 2:
        if level >= 4 and bool(best_params.get("use_boot", False)):
            boot_B = int(best_params.get("boot_B", 16))
            boot_frac = float(best_params.get("boot_frac", 0.7))
            boot_stable_min = float(best_params.get("boot_stable_min", 0.8))
            mono_pairs = build_mono_pairs_bootstrap(
                Xtr_np, Ytr_np,
                rho_abs_min=float(rho_min),
                topk=int(topk),
                B=int(boot_B),
                sample_frac=float(boot_frac),
                stable_min=float(boot_stable_min),
            )
        else:
            mono_pairs = build_mono_pairs_spearman(
                Xtr_np, Ytr_np,
                rho_abs_min=float(rho_min),
                topk=int(topk),
            )

    idx_map = {c:i for i,c in enumerate(OUTPUT_COLS)}

    # ---- lazy heads (match objective behavior) ----
    def ensure_delta_head():
        if model._delta_head is None:
            model._delta_head = nn.Sequential(
                nn.Linear(x_tr.shape[1], 64),
                nn.SiLU(),
                nn.Linear(64, len(ITER2_IDX))
            ).to(device)
            for m in model._delta_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            opt.add_param_group({"params": model._delta_head.parameters()})

    def ensure_aux_head(z_dim, out_dim):
        if model._aux_head is None:
            model._aux_head = nn.Sequential(
                nn.Linear(z_dim, 64),
                nn.SiLU(),
                nn.Linear(64, out_dim)
            ).to(device)
            for m in model._aux_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            opt.add_param_group({"params": model._aux_head.parameters()})

    # ---- training ----
    n = x_tr.shape[0]
    best = 1e18
    best_state = None
    bad = 0
    patience = 25

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)

        for s in range(0, n, batch):
            b = perm[s:s+batch]
            xb = x_tr[b]
            yb = y_tr[b]

            # level4 head
            if level >= 4:
                ensure_delta_head()

            # IMPORTANT: mono loss needs grad wrt inputs
            xb_req = xb.detach().clone().requires_grad_(True) if (level >= 2 and mono_pairs) else xb

            # single forward
            if level >= 3 and aux_z_tr is not None and w_aux > 0:
                mu, logvar, z = model(xb_req, return_z=True)
            else:
                mu, logvar = model(xb_req)
                z = None

            loss = w_data * gaussian_nll(yb, mu, logvar)

            if level >= 1:
                loss = loss + w_fp * loss_level1_shifted(mu, bias_delta_t)

            if level >= 2:
                loss = loss + w_mono * loss_level2_monotone_from_mu(mu, xb_req, mono_pairs)

            if level >= 3:
                loss = loss + w_ineq * loss_level3_ineq(mu, idx_map)

            # aux supervision (use z, do NOT detach)
            if level >= 3 and aux_z_tr is not None and w_aux > 0:
                zb = aux_z_tr[b]
                ensure_aux_head(z_dim=z.shape[1], out_dim=zb.shape[1])
                zhat = model._aux_head(z)
                loss = loss + w_aux * F.mse_loss(zhat, zb)

            if level >= 4:
                # band shift uses same xb_req
                loss = loss + w_shift * loss_level1_band_shift(mu, xb_req, model, eps_band=float(eps_band))
                loss = loss + w_logvar * logvar_floor_regularizer(logvar, floor=float(logvar_floor))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # ---- validation ----
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

def run_ood_evaluation(best_params, level, X, Y, feature_name, out_dir):
    """
    Train on in-distribution middle region, test on tail OOD region.
    Saves OOD metrics json.
    """
    device = torch.device(DEVICE)

    X_in, X_ood, Y_in, Y_ood, meta = make_ood_split_by_feature(X, Y, feature_name)

    # split in-distribution into train/val
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_in, Y_in, test_size=0.1765, random_state=SEED
    )

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)

    Xtr_s = sx.transform(X_tr)
    Xva_s = sx.transform(X_va)
    Xood_s = sx.transform(X_ood)

    Ytr_s = sy.transform(Y_tr)
    Yva_s = sy.transform(Y_va)
    Yood_s = sy.transform(Y_ood)

    x_tr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
    x_va = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    y_va = torch.tensor(Yva_s, dtype=torch.float32, device=device)
    x_ood = torch.tensor(Xood_s, dtype=torch.float32, device=device)
    y_ood = torch.tensor(Yood_s, dtype=torch.float32, device=device)

    delta_tr = Ytr_s[:, ITER2_IDX] - Ytr_s[:, ITER1_IDX]
    bias_delta = delta_tr.mean(axis=0)
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    pack_ood = (
        x_tr, y_tr, x_va, y_va,
        Xtr_s, Ytr_s,
        None, None,
        bias_delta_t
    )

    model, mono_pairs = train_with_params(best_params, level, pack_ood)

    with torch.no_grad():
        mu_ood_s, logvar_ood = model(x_ood)
        var_ood_s = torch.exp(logvar_ood)

    mu_ood = sy.inverse_transform(_to_numpy(mu_ood_s))
    y_ood_true = sy.inverse_transform(_to_numpy(y_ood))
    sigma_ood = np.sqrt(_to_numpy(var_ood_s)) * sy.scale_

    basic = compute_basic_metrics(y_ood_true, mu_ood)
    prob90 = compute_prob_metrics_gaussian(y_ood_true, mu_ood, sigma_ood, alpha=0.10)

    out = {
        "meta": meta,
        "basic_all_mean": {
            "MAE_mean": float(np.mean(basic["MAE"])),
            "RMSE_mean": float(np.mean(basic["RMSE"])),
            "R2_mean": float(np.mean(basic["R2"])),
        },
        "prob90_all_mean": {
            "PICP_mean": float(np.mean(prob90["PICP"])),
            "MPIW_mean": float(np.mean(prob90["MPIW"])),
            "CRPS_mean": float(np.mean(prob90["CRPS"])),
        }
    }

    with open(os.path.join(out_dir, f"ood_level{level}_{feature_name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OOD] Saved: {os.path.join(out_dir, f'ood_level{level}_{feature_name}.json')}")
    return out

# ----------------------------- Main -----------------------------
def main():
    seed_all(SEED)
    ensure_dir(OUT_DIR)
    device = torch.device(DEVICE)

    # 1) load main table
    # If out_fenicsdata_coupled.txt is not a CSV, you must convert it.
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=INPUT_COLS + OUTPUT_COLS).reset_index(drop=True)

    for c in INPUT_COLS + OUTPUT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in main table: {c}")

    in_stats = summarize(df, INPUT_COLS)
    out_stats = summarize(df, OUTPUT_COLS)
    print_stats("INPUT STATS", in_stats)
    print_stats("OUTPUT STATS", out_stats)

    with open(os.path.join(OUT_DIR,"meta_stats.json"),"w",encoding="utf-8") as f:
        json.dump({"PARAM_META":PARAM_META,"input_stats":in_stats,"output_stats":out_stats}, f, indent=2)

    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)

    # split 70/15/15
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.15, random_state=SEED)
    X_tr, X_va, Y_tr, Y_va = train_test_split(X_tr, Y_tr, test_size=0.1765, random_state=SEED)

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(Y_tr)
    Xtr_s, Xva_s, Xte_s = sx.transform(X_tr), sx.transform(X_va), sx.transform(X_te)
    Ytr_s, Yva_s, Yte_s = sy.transform(Y_tr), sy.transform(Y_va), sy.transform(Y_te)

    # after Ytr_s is computed (numpy)
    delta_tr = Ytr_s[:, ITER2_IDX] - Ytr_s[:, ITER1_IDX]     # [Ntr, 8]
    bias_delta = delta_tr.mean(axis=0)                       # [8]
    bias_delta_t = torch.tensor(bias_delta, dtype=torch.float32, device=device)

    x_tr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Ytr_s, dtype=torch.float32, device=device)
    x_va = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    y_va = torch.tensor(Yva_s, dtype=torch.float32, device=device)
    x_te = torch.tensor(Xte_s, dtype=torch.float32, device=device)
    y_te = torch.tensor(Yte_s, dtype=torch.float32, device=device)

    # 2) build aux z from txt (optional)
    # We will attempt: fuel/hp nearby stress scalar OR vector summary, and fuel_T_list summaries.
    aux_cols = []
    aux_tr_list = []
    aux_va_list = []

    def attach_aux(name, arr_full):
        # arr_full corresponds to ALL samples in the same order as df
        # Here we assume the txt rows align with df rows. If not, you must provide an id key.
        # split using the same indices as train_test_split would require tracking indices;
        # for simplicity, we re-split with indices here.
        pass

    aux_z_tr = None
    aux_z_va = None
    # Minimal safe: only use scalar files that match N rows.
    try:
        N = len(df)
        feats = []
        feat_names = []
        for k, p in AUX_PATHS.items():
            if not os.path.exists(p):
                continue
            if p.endswith(".npy"):
                arr = np.load(p)
                # if it's [N, K], summarize to mean/max/p95 per sample
                if arr.ndim == 2 and arr.shape[0] == N:
                    summ = {"mean": arr.mean(1), "max": arr.max(1), "p95": np.quantile(arr, 0.95, axis=1)}
                    for sname, vec in summ.items():
                        feats.append(vec.reshape(-1,1))
                        feat_names.append(f"{k}_{sname}")
                continue

            arr = load_txt_as_array(p)  # [M,K]
            if arr.shape[0] != N:
                # can't align safely
                continue
            summ = summarize_vector_rows(arr)
            for sname, vec in summ.items():
                feats.append(vec.reshape(-1,1))
                feat_names.append(f"{k}_{sname}")

        if feats:
            Z = np.hstack(feats).astype(float)  # [N, Zdim]
            # split with indices explicitly
            idx = np.arange(N)
            idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=SEED)
            idx_tr, idx_va = train_test_split(idx_tr, test_size=0.1765, random_state=SEED)

            sz = StandardScaler().fit(Z[idx_tr])
            Ztr_s = sz.transform(Z[idx_tr])
            Zva_s = sz.transform(Z[idx_va])

            aux_z_tr = torch.tensor(Ztr_s, dtype=torch.float32, device=device)
            aux_z_va = torch.tensor(Zva_s, dtype=torch.float32, device=device)

            with open(os.path.join(OUT_DIR,"aux_features.json"),"w",encoding="utf-8") as f:
                json.dump({"aux_dim": int(Z.shape[1]), "aux_names": feat_names}, f, indent=2)
            print(f"\n[AUX] Loaded aligned aux features: dim={Z.shape[1]}")
        else:
            print("\n[AUX] No aligned aux features found (txt rows not matching N). Level3 will still run (ineq only).")
    except Exception as e:
        print(f"\n[AUX][WARN] Failed to load aux txt/npy: {e}\nLevel3 will run without aux supervision.")
        aux_z_tr = None
        aux_z_va = None

    # pack
    pack = (x_tr, y_tr, x_va, y_va, Xtr_s, Ytr_s, aux_z_tr, aux_z_va, bias_delta_t)

    # 3) Run studies for Level0/1/2/3 for comparison
    results = {}
    for level in PAPER_LEVELS:
    # for level in [4]:
        print(f"\n================= OPTUNA Level {level} =================")
        study = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=SEED),
                                   pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))
        study.optimize(objective_factory(level, pack), n_trials=TRIALS)

        results[f"level{level}"] = {
            "best_value": float(study.best_value),
            "best_params": study.best_params,
        }
        with open(os.path.join(OUT_DIR, f"best_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(results[f"level{level}"], f, indent=2)

        print(f"[BEST L{level}] value={study.best_value:.6g}")
        print(study.best_params)

        # --------- Train best model and evaluate on TEST (paper-ready metrics) ---------
        best_params = study.best_params
        model, mono_pairs = train_with_params(best_params, level, pack)

        # deterministic prediction (aleatoric only)
        with torch.no_grad():
            mu_te_s, logvar_te = model(x_te)
            var_te_s = torch.exp(logvar_te)

        # inverse transform to ORIGINAL scale
        mu_te_s_np = _to_numpy(mu_te_s)
        y_te_s_np = _to_numpy(y_te)

        mu_te = sy.inverse_transform(mu_te_s_np)
        y_te_true = sy.inverse_transform(y_te_s_np)

        # sigma in ORIGINAL scale: multiply by sy.scale_
        # because y = sy.scale_ * y_s + sy.mean_
        sigma_te = np.sqrt(_to_numpy(var_te_s)) * sy.scale_

        # metrics
        idx_map = {c:i for i,c in enumerate(OUTPUT_COLS)}
        basic = compute_basic_metrics(y_te_true, mu_te)
        # test NLL in standardized space (comparable with optimization)
        test_nll = float(gaussian_nll(y_te, mu_te_s, logvar_te).item())
        prob90 = compute_prob_metrics_gaussian(y_te_true, mu_te, sigma_te, alpha=0.10)  # 90% PI
        viol = eval_inequality_violation(mu_te, idx_map)

        # grouped metrics (iter1 vs iter2)
        iter1 = slice(0,8)
        iter2 = slice(8,16)
        basic_iter1 = {
            "MAE_mean": float(np.mean(basic["MAE"][iter1])),
            "RMSE_mean": float(np.mean(basic["RMSE"][iter1])),
            "R2_mean": float(np.mean(basic["R2"][iter1])),
        }
        basic_iter2 = {
            "MAE_mean": float(np.mean(basic["MAE"][iter2])),
            "RMSE_mean": float(np.mean(basic["RMSE"][iter2])),
            "R2_mean": float(np.mean(basic["R2"][iter2])),
        }
        prob90_iter1 = {
            "PICP_mean": float(np.mean(prob90["PICP"][iter1])),
            "MPIW_mean": float(np.mean(prob90["MPIW"][iter1])),
            "CRPS_mean": float(np.mean(prob90["CRPS"][iter1])),
        }
        prob90_iter2 = {
            "PICP_mean": float(np.mean(prob90["PICP"][iter2])),
            "MPIW_mean": float(np.mean(prob90["MPIW"][iter2])),
            "CRPS_mean": float(np.mean(prob90["CRPS"][iter2])),
        }

        # Optional epistemic via MC Dropout
        mc = {}
        if float(best_params.get("dropout", 0.0)) > 0.0:
            mu_mc_s, var_mc_s, epi_s, ale_s = predict_mc_dropout(model, x_te, T=30)
            mu_mc = sy.inverse_transform(_to_numpy(mu_mc_s))
            sigma_mc = np.sqrt(_to_numpy(var_mc_s)) * sy.scale_
            prob90_mc = compute_prob_metrics_gaussian(y_te_true, mu_mc, sigma_mc, alpha=0.10)
            mc = {
                "mc_dropout_T": 30,
                "prob90_PICP_mean_all": float(np.mean(prob90_mc["PICP"])),
                "prob90_MPIW_mean_all": float(np.mean(prob90_mc["MPIW"])),
                "prob90_CRPS_mean_all": float(np.mean(prob90_mc["CRPS"])),
                "epistemic_var_mean_all_stdspace": float(np.mean(_to_numpy(epi_s))),
                "aleatoric_var_mean_all_stdspace": float(np.mean(_to_numpy(ale_s))),
            }

        # save mono pairs (interpretability)
        readable_pairs = []
        if level >= 2:
            for (i,j,sign,w) in mono_pairs[:50]:
                readable_pairs.append({
                    "x": INPUT_COLS[i],
                    "y": OUTPUT_COLS[j],
                    "direction": "increasing" if sign>0 else "decreasing",
                    "abs_rho": float(w),
                })

        # --------- Safety threshold analysis ---------
        threshold_main = PRIMARY_STRESS_THRESHOLD
        threshold_sweep = THRESHOLD_SWEEP

        fail_main = compute_failure_metrics_gaussian(mu_te, sigma_te, threshold_main)
        fail_sweep = compute_threshold_sweep(mu_te, sigma_te, threshold_sweep)

        metrics = {
            "level": level,
            "test_nll_standardized": test_nll,
            "failure_main": fail_main,
            "failure_threshold_sweep": fail_sweep,

            "basic_all_mean": {
                "MAE_mean": float(np.mean(basic["MAE"])),
                "RMSE_mean": float(np.mean(basic["RMSE"])),
                "R2_mean": float(np.mean(basic["R2"])),
            },
            "basic_iter1": basic_iter1,
            "basic_iter2": basic_iter2,

            "prob90_all_mean": {
                "PICP_mean": float(np.mean(prob90["PICP"])),
                "MPIW_mean": float(np.mean(prob90["MPIW"])),
                "CRPS_mean": float(np.mean(prob90["CRPS"])),
            },
            "prob90_iter1": prob90_iter1,
            "prob90_iter2": prob90_iter2,

            "ineq_violation_rates_on_mu": viol,
            "mono_pairs_top": readable_pairs,
            "mc_dropout": mc,
        }

        with open(os.path.join(OUT_DIR, f"metrics_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n[METRICS L{level}] test_nll(std)={test_nll:.6g}  "
              f"RMSE_mean={metrics['basic_all_mean']['RMSE_mean']:.4g}  "
              f"PICP90_mean={metrics['prob90_all_mean']['PICP_mean']:.3f}")
        
        # --------- Save raw test predictions for downstream analysis ---------
        pred_dump = {
            "y_true": y_te_true.tolist(),
            "mu": mu_te.tolist(),
            "sigma": sigma_te.tolist(),
            "output_names": OUTPUT_COLS,
            "level": level,
        }

        with open(os.path.join(OUT_DIR, f"test_predictions_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(pred_dump, f, indent=2, ensure_ascii=False)

        print(f"[PRED] Saved raw test predictions: {os.path.join(OUT_DIR, f'test_predictions_level{level}.json')}")
        
        # --------- Save per-dimension paper metrics CSV ---------
        per_dim_rows = []
        for j, name in enumerate(OUTPUT_COLS):
            per_dim_rows.append({
                "level": level,
                "output": name,
                "iter": "iter1" if j < 8 else "iter2",
                "MAE": float(basic["MAE"][j]),
                "RMSE": float(basic["RMSE"][j]),
                "R2": float(basic["R2"][j]),
                "PICP90": float(prob90["PICP"][j]),
                "MPIW90": float(prob90["MPIW"][j]),
                "CRPS": float(prob90["CRPS"][j]),
            })

        df_per_dim = pd.DataFrame(per_dim_rows)
        csv_path = os.path.join(OUT_DIR, f"paper_metrics_per_dim_level{level}.csv")
        df_per_dim.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[PAPER] Saved per-dim CSV: {csv_path}")

    with open(os.path.join(OUT_DIR, "all_levels_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

        # --------- Build a summary CSV table for paper ---------
    rows=[]
    for level in PAPER_LEVELS:
        p = os.path.join(OUT_DIR, f"metrics_level{level}.json")
        if not os.path.exists(p): 
            continue
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "level": level,
            "test_nll_std": m["test_nll_standardized"],
            "RMSE_mean_all": m["basic_all_mean"]["RMSE_mean"],
            "R2_mean_all": m["basic_all_mean"]["R2_mean"],
            "PICP90_mean_all": m["prob90_all_mean"]["PICP_mean"],
            "MPIW90_mean_all": m["prob90_all_mean"]["MPIW_mean"],
            "CRPS_mean_all": m["prob90_all_mean"]["CRPS_mean"],
            "RMSE_mean_iter2": m["basic_iter2"]["RMSE_mean"],
            "PICP90_mean_iter2": m["prob90_iter2"]["PICP_mean"],
        })
                # --------- OOD evaluation (tail extrapolation) ---------
        try:
            ood_feature = OOD_FEATURE
            ood_res = run_ood_evaluation(best_params, level, X, Y, ood_feature, OUT_DIR)
        except Exception as e:
            print(f"[OOD][WARN] Failed OOD evaluation for level {level}: {e}")

    if rows:
        dfm = pd.DataFrame(rows)
        dfm.to_csv(os.path.join(OUT_DIR, "paper_metrics_table.csv"), index=False, encoding="utf-8-sig")
        print(f"\n[PAPER] Saved table: {os.path.join(OUT_DIR, 'paper_metrics_table.csv')}")
    print("\n[OK] Finished. Compare all_levels_summary.json for appendix tables.")

    

if __name__ == "__main__":
    main()