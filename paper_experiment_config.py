# paper_experiment_config.py
# ============================================================
# Frozen paper experiment specification
# ============================================================
import os
# -----------------------------
# Paths
# -----------------------------
DATA_ROOT = "/home/tjzs/Documents/fenics_data/fenics_data/txt_extract"
CSV_PATH = f"{DATA_ROOT}/dataset_v3.csv"
OUT_DIR = "./experiments_phys_levels"

# -----------------------------
# Fixed surrogate paths
# -----------------------------
FIXED_SURROGATE_TAG = "fixed_level2"
FIXED_SURROGATE_DIR = os.path.join(OUT_DIR, f"fixed_surrogate_{FIXED_SURROGATE_TAG}")
FIXED_SPLIT_DIR = os.path.join(OUT_DIR, "fixed_split")

FIXED_CKPT_PATH = os.path.join(FIXED_SURROGATE_DIR, "checkpoint_level2.pt")
FIXED_SCALER_PATH = os.path.join(FIXED_SURROGATE_DIR, "scalers_level2.pkl")
FIXED_META_PATH = os.path.join(FIXED_SURROGATE_DIR, "metrics_level2.json")
FIXED_TEST_PRED_PATH = os.path.join(FIXED_SURROGATE_DIR, "test_predictions_level2.json")
FIXED_BEST_PARAM_PATH = os.path.join(FIXED_SURROGATE_DIR, "best_level2.json")

# -----------------------------
# Randomness / device / search
# -----------------------------
SEED = 2026
TRIALS = 40
DEVICE = "cuda"

# -----------------------------
# Input / output columns
# -----------------------------
INPUT_COLS = [
    "E_slope", "E_intercept", "nu", "alpha_base",
    "alpha_slope", "SS316_T_ref", "SS316_k_ref", "SS316_alpha"
]

OUT1 = [
    # "iteration1_keff",
    "iteration1_avg_fuel_temp",
    "iteration1_max_fuel_temp",
    "iteration1_max_monolith_temp",
    "iteration1_max_global_stress",
    "iteration1_monolith_new_temperature",
    "iteration1_Hcore_after",
    "iteration1_wall2",
]

OUT2 = [
    "iteration2_keff",
    "iteration2_avg_fuel_temp",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_max_global_stress",
    "iteration2_monolith_new_temperature",
    "iteration2_Hcore_after",
    "iteration2_wall2",
]

OUTPUT_COLS = OUT1 + OUT2

ITER1_IDX = list(range(0, 7))
ITER2_IDX = list(range(7, 15))

# -----------------------------
# Main paper outputs
# -----------------------------
PRIMARY_OUTPUTS = [
    "iteration2_keff",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_max_global_stress",
    "iteration2_wall2",
]

PRIMARY_STRESS_OUTPUT = "iteration2_max_global_stress"
PRIMARY_AUXILIARY_OUTPUT = "iteration2_keff"

PRIMARY_SA_OUTPUTS = [
    "iteration2_max_global_stress",
    "iteration2_keff",
]

# -----------------------------
# Paper levels
# -----------------------------
PAPER_LEVELS = [0, 2]

# -----------------------------
# Safety thresholds
# -----------------------------
PRIMARY_STRESS_THRESHOLD = 131.0  # MPa
THRESHOLD_SWEEP = [110.0, 120.0, 131.0]

# -----------------------------
# OOD
# -----------------------------
OOD_FEATURE = "alpha_slope"   # placeholder; can be updated after Sobol
OOD_KEEP_MIDDLE_RATIO = 0.80

# -----------------------------
# Param meta (for paper stats)
# -----------------------------
PARAM_META = {
    "E_slope": {"unit": "Pa/K", "meaning": "E(T)=E_slope*T+E_intercept"},
    "E_intercept": {"unit": "Pa", "meaning": "E(T)=E_slope*T+E_intercept"},
    "nu": {"unit": "-", "meaning": "Poisson ratio"},
    "alpha_base": {"unit": "1/K", "meaning": "alpha(T)=alpha_slope*T+alpha_base"},
    "alpha_slope": {"unit": "1/K^2", "meaning": "alpha(T)=alpha_slope*T+alpha_base"},
    "SS316_T_ref": {"unit": "K", "meaning": "T_ref in k(T)=k_slope*(T-T_ref)+k_ref"},
    "SS316_k_ref": {"unit": "W/(m·K)", "meaning": "k_ref in k(T)"},
    "SS316_alpha": {"unit": "W/(m·K^2)", "meaning": "k_slope in k(T)"},
}