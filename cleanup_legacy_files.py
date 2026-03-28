# cleanup_legacy_files.py
# ============================================================
# Move historical / currently-unused files into a timestamped
# legacy folder, instead of deleting them.
#
# Default behavior:
#   - DRY_RUN = True  -> preview only
#   - DRY_RUN = False -> actually move files
# ============================================================

import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/tjzs/Documents/0310")
EXP = ROOT / "experiments_phys_levels"
BENCH = EXP / "benchmark_case"

DRY_RUN = False

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LEGACY_DIR = EXP / f"_legacy_unused_{timestamp}"
LEGACY_BENCH_DIR = LEGACY_DIR / "benchmark_case"

# ------------------------------------------------------------
# 1) Root-level legacy result files in experiments_phys_levels
#    Keep:
#      - fixed_split/
#      - fixed_surrogate_fixed_level2/
#      - fixed_surrogate_fixed_base/
#      - current paper outputs
#    Move:
#      - old mixed root-level training artifacts
#      - old calibration/inverse intermediate outputs
#      - old duplicate summary files
# ------------------------------------------------------------
ROOT_FILES_TO_MOVE = [
    # old root-level best/checkpoint/scaler/metrics artifacts
    "best_level0.json",
    "best_level1.json",
    "best_level2.json",
    "best_level3.json",
    "best_level4.json",
    "checkpoint_level0.pt",
    "checkpoint_level1.pt",
    "checkpoint_level2.pt",
    "checkpoint_level4.pt",
    "meta_stats.json",
    "metrics_level0.json",
    "metrics_level1.json",
    "metrics_level2.json",
    "metrics_level3.json",
    "metrics_level4.json",
    "sanity_level0.csv",
    "sanity_level0.json",
    "sanity_level1.csv",
    "sanity_level1.json",
    "sanity_level2.csv",
    "sanity_level2.json",
    "sanity_level4.csv",
    "sanity_level4.json",
    "scalers_level0.pkl",
    "scalers_level1.pkl",
    "scalers_level2.pkl",
    "scalers_level4.pkl",
    "test_predictions_level0.json",
    "test_predictions_level1.json",
    "test_predictions_level2.json",
    "test_predictions_level4.json",
    "paper_focus_metrics_level0.csv",
    "paper_focus_metrics_level1.csv",
    "paper_focus_metrics_level2.csv",
    "paper_focus_metrics_level4.csv",
    "paper_metrics_per_dim_level0.csv",
    "paper_metrics_per_dim_level1.csv",
    "paper_metrics_per_dim_level2.csv",
    "paper_metrics_per_dim_level3.csv",
    "paper_metrics_per_dim_level4.csv",

    # old calibration / inverse mixed outputs
    "calibration_benchmark_case_summary.csv",
    "calibration_benchmark_case_summary_full.csv",
    "calibration_benchmark_case_summary_reduced.csv",
    "calibration_benchmark_case_summary_reduced_maintext.csv",
    "calibration_benchmark_meta.json",
    "calibration_benchmark_meta_full.json",
    "calibration_benchmark_meta_reduced.json",
    "calibration_benchmark_meta_reduced_maintext.json",
    "calibration_benchmark_observation_fit.csv",
    "calibration_benchmark_observation_fit_full.csv",
    "calibration_benchmark_observation_fit_reduced.csv",
    "calibration_benchmark_observation_fit_reduced_maintext.csv",
    "calibration_benchmark_observation_fit_summary.csv",
    "calibration_benchmark_observation_fit_summary_full.csv",
    "calibration_benchmark_observation_fit_summary_reduced.csv",
    "calibration_benchmark_observation_fit_summary_reduced_maintext.csv",
    "calibration_benchmark_parameter_recovery.csv",
    "calibration_benchmark_parameter_recovery_full.csv",
    "calibration_benchmark_parameter_recovery_reduced.csv",
    "calibration_benchmark_parameter_recovery_reduced_maintext.csv",
    "calibration_benchmark_parameter_recovery_summary.csv",
    "calibration_benchmark_parameter_recovery_summary_full.csv",
    "calibration_benchmark_parameter_recovery_summary_reduced.csv",
    "calibration_benchmark_parameter_recovery_summary_reduced_maintext.csv",

    "calibration_feasible_region_overview.csv",
    "calibration_feasible_region_summary_thr120.csv",
    "calibration_feasible_region_summary_thr131.csv",
    "calibration_logposterior_trace.csv",
    "calibration_observation_fit.csv",
    "calibration_posterior_predictive_mu.csv",
    "calibration_posterior_predictive_samples.csv",
    "calibration_posterior_predictive_sigma.csv",
    "calibration_posterior_samples.csv",
    "calibration_posterior_summary.csv",
    "calibration_run_meta.json",

    "inverse_benchmark_case_summary_reduced_maintext.csv",
    "inverse_benchmark_meta_reduced_maintext.json",
    "inverse_benchmark_observation_fit_reduced_maintext.csv",
    "inverse_benchmark_observation_fit_summary_reduced_maintext.csv",
    "inverse_benchmark_parameter_recovery_reduced_maintext.csv",
    "inverse_benchmark_parameter_recovery_summary_reduced_maintext.csv",
    "inverse_calibration_pool.csv",
    "inverse_case_indices_reduced_maintext.csv",
    "inverse_diagnostics_summary.json",
    "inverse_diagnostics_summary_full.json",
    "inverse_diagnostics_summary_reduced.json",
    "inverse_diagnostics_summary_reduced_maintext.json",
    "inverse_emulator_pool.csv",
    "inverse_split_meta.json",

    # old compare / contraction intermediates
    "paper_prior_posterior_contraction_reduced.csv",
    "paper_prior_posterior_contraction_summary_reduced.csv",
    "paper_prior_posterior_contraction_summary_reduced.json",

    # old iter compare intermediates
    "paper_iter1_iter2_forward_compare.csv",
    "paper_iter1_iter2_sobol_compare.csv",
    "paper_iter1_iter2_sobol_shift_summary.csv",
    "paper_iter2_keff_sobol_summary.csv",

    # old safety threshold level-wise raw files
    "safety_threshold_analysis_level0.json",
    "safety_threshold_analysis_level1.json",
    "safety_threshold_analysis_level2.json",
    "safety_threshold_analysis_level4.json",
    "safety_threshold_sweep_level0.csv",
    "safety_threshold_sweep_level1.csv",
    "safety_threshold_sweep_level2.csv",
    "safety_threshold_sweep_level4.csv",

    # old ood raw files
    "ood_level0_alpha_slope.json",
    "ood_level1_alpha_slope.json",
    "ood_level2_alpha_slope.json",
    "ood_level4_alpha_slope.json",

    # old misc
    "all_levels_summary.json",
]

# ------------------------------------------------------------
# 2) Keep these important/current files/directories in place
# ------------------------------------------------------------
KEEP_TOP_LEVEL_NAMES = {
    "fixed_split",
    "fixed_surrogate_fixed_level2",
    "fixed_surrogate_fixed_base",

    "paper_forward_uq_summary.csv",
    "paper_safety_threshold_summary.csv",
    "paper_safety_threshold_sweep_all_levels.csv",
    "paper_sobol_results.csv",
    "paper_sobol_results_with_ci.csv",
    "paper_sobol_results_with_ci_all_iters.csv",
    "paper_sobol_methods_ready_summary.csv",
    "paper_sobol_methods_ready_summary.json",
    "paper_sobol_results_ready_top_factors.csv",
    "paper_metrics_table.csv",
    "paper_inverse_full_vs_reduced_summary.csv",
    "paper_inverse_full_vs_reduced_summary.json",
    "paper_inverse_full_vs_reduced_parameter_table.csv",
    "paper_inverse_full_vs_reduced_observable_table.csv",
    "paper_speed_benchmark_detailed.json",
    "paper_speedup_benchmark.json",
    "paper_ood_results.csv",
    "paper_ood_meta.json",
    "paper_ood_per_dim_level0.csv",
    "paper_ood_per_dim_level2.csv",

    "forward_uq_all_outputs_level0.csv",
    "forward_uq_all_outputs_level2.csv",
    "forward_uq_primary_outputs_level0.csv",
    "forward_uq_primary_outputs_level2.csv",
    "forward_uq_failure_prob_level0.csv",
    "forward_uq_failure_prob_level2.csv",
    "forward_uq_cvr_level0.csv",
    "forward_uq_cvr_level2.csv",
    "forward_uq_cvr_summary_level0.json",
    "forward_uq_cvr_summary_level2.json",
    "forward_uq_joint_stress_keff_level0.csv",
    "forward_uq_joint_stress_keff_level2.csv",
    "forward_uq_input_samples.csv",
    "forward_uq_samples_level0.npz",
    "forward_uq_samples_level2.npz",

    "paper_forward_figures_final",
    "paper_inverse_figures_final",
    "paper_inverse_2d_figures_final",
    "benchmark_case",
}

# ------------------------------------------------------------
# 3) benchmark_case cleanup rule:
#    Move old variants, keep only *_reduced_maintext.csv
# ------------------------------------------------------------
def is_benchmark_legacy_file(name: str) -> bool:
    if not name.endswith(".csv"):
        return False

    # keep current maintext reduced outputs
    if name.endswith("_reduced_maintext.csv"):
        return False

    # move old variants
    if (
        "_full_chain.csv" in name
        or "_posterior_predictive.csv" in name
        or "_posterior_samples.csv" in name
        or "_prior_samples.csv" in name
        or "_full_chain_full.csv" in name
        or "_posterior_predictive_full.csv" in name
        or "_posterior_samples_full.csv" in name
        or "_prior_samples_full.csv" in name
        or "_full_chain_reduced.csv" in name
        or "_posterior_predictive_reduced.csv" in name
        or "_posterior_samples_reduced.csv" in name
        or "_prior_samples_reduced.csv" in name
    ):
        return True

    return False


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path):
    ensure_parent(dst)
    if DRY_RUN:
        print(f"[DRY] MOVE: {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))
        print(f"[OK] MOVE: {src} -> {dst}")


def main():
    if not EXP.exists():
        raise FileNotFoundError(f"Not found: {EXP}")

    print("=" * 80)
    print(f"ROOT: {ROOT}")
    print(f"EXP : {EXP}")
    print(f"LEGACY_DIR: {LEGACY_DIR}")
    print(f"DRY_RUN = {DRY_RUN}")
    print("=" * 80)

    planned = []

    # --------------------------------------------------------
    # A. explicitly listed root files
    # --------------------------------------------------------
    for rel in ROOT_FILES_TO_MOVE:
        src = EXP / rel
        if src.exists():
            dst = LEGACY_DIR / rel
            planned.append((src, dst))

    # --------------------------------------------------------
    # B. benchmark_case old variants
    # --------------------------------------------------------
    if BENCH.exists():
        for src in sorted(BENCH.iterdir()):
            if src.is_file() and is_benchmark_legacy_file(src.name):
                dst = LEGACY_BENCH_DIR / src.name
                planned.append((src, dst))

    # --------------------------------------------------------
    # C. optional safety net:
    #    find unknown loose files in EXP root that are not in keep-set
    #    and not directories we want to preserve
    # --------------------------------------------------------
    for src in sorted(EXP.iterdir()):
        if src.name.startswith("_legacy_unused_"):
            continue
        if src.name in KEEP_TOP_LEVEL_NAMES:
            continue
        if src.is_file():
            dst = LEGACY_DIR / src.name
            pair = (src, dst)
            if pair not in planned:
                planned.append(pair)

    # deduplicate
    unique = []
    seen = set()
    for src, dst in planned:
        key = (str(src), str(dst))
        if key not in seen:
            seen.add(key)
            unique.append((src, dst))
    planned = unique

    print(f"[INFO] Planned moves: {len(planned)}")
    for src, dst in planned:
        print(f"  - {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")

    if not planned:
        print("[INFO] Nothing to move.")
        return

    if not DRY_RUN:
        LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    for src, dst in planned:
        move_path(src, dst)

    print("=" * 80)
    if DRY_RUN:
        print("[DONE] Preview finished. Set DRY_RUN = False to actually move files.")
    else:
        print(f"[DONE] Legacy files moved to: {LEGACY_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()