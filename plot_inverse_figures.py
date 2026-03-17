# plot_inverse_figures.py
# ============================================================
# Purpose:
#   Plot inverse-UQ figures directly from saved CSV files.
#
# This script DOES NOT retrain any model.
# It only reads saved benchmark / posterior files and exports figures.
#
# Required input files:
#   1) calibration_benchmark_case_summary.csv
#      - used for: feasible fraction vs observed stress
#
#   2) calibration_benchmark_parameter_recovery_summary.csv
#      - used for: parameter recoverability summary
#
#   3) calibration_benchmark_observation_fit_summary.csv
#      - used for: posterior predictive fit summary
#
# Optional input files:
#   4) benchmark_caseXXX_posterior_samples.csv
#      - used for: representative posterior marginals
#
#   5) benchmark_caseXXX_full_chain.csv
#      - used for: representative trace plot
#
#   6) benchmark_caseXXX_posterior_predictive.csv
#      - used for: representative posterior predictive fit
#
#   7) calibration_benchmark_observation_fit.csv
#      - used together with (6) to recover representative observation values
#
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================

OUT_DIR = "./experiments_phys_levels"
FIG_DIR = os.path.join(OUT_DIR, "paper_inverse_figures_final")

PRIMARY_STRESS_THRESHOLD = 131.0

# Representative case selection:
# "manual", "closest_to_threshold", "lowest_stress", "highest_stress"
REP_CASE_MODE = "closest_to_threshold"
MANUAL_CASE_ID = 3

# Parameters shown in posterior / trace figures
KEY_PARAMS = ["E_intercept", "alpha_base", "alpha_slope", "SS316_k_ref"]

# Observables shown in representative predictive-fit figure
KEY_OBSERVABLES = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]

# Plot style requirements:
# 1) clean white background
# 2) one message per figure
# 3) no double-axis unless absolutely necessary
# 4) keep labels readable
# 5) save high-resolution png for paper drafting
DPI = 300


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_csv_optional(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def choose_representative_case(df_case: pd.DataFrame) -> int:
    if REP_CASE_MODE == "manual":
        return int(MANUAL_CASE_ID)
    elif REP_CASE_MODE == "lowest_stress":
        return int(df_case.loc[df_case["obs_stress"].idxmin(), "benchmark_case_id"])
    elif REP_CASE_MODE == "highest_stress":
        return int(df_case.loc[df_case["obs_stress"].idxmax(), "benchmark_case_id"])
    elif REP_CASE_MODE == "closest_to_threshold":
        idx = (df_case["obs_stress"] - PRIMARY_STRESS_THRESHOLD).abs().idxmin()
        return int(df_case.loc[idx, "benchmark_case_id"])
    else:
        raise ValueError(f"Unsupported REP_CASE_MODE: {REP_CASE_MODE}")


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 1
# Posterior feasible fraction versus observed stress
# File used:
#   calibration_benchmark_case_summary.csv
# Meaning:
#   As observed stress increases, how fast does the posterior
#   safety-feasible fraction collapse under 131 MPa?
# ============================================================

def plot_feasible_fraction_vs_stress(df_case: pd.DataFrame, save_path: str):
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(
        df_case["obs_stress"],
        df_case["feasible_fraction_131"],
        s=70
    )
    plt.axvline(
        PRIMARY_STRESS_THRESHOLD,
        linestyle="--",
        linewidth=1.5,
        label=f"{PRIMARY_STRESS_THRESHOLD:.0f} MPa threshold"
    )
    plt.xlabel("Observed stress (MPa)")
    plt.ylabel("Feasible posterior fraction at 131 MPa")
    plt.title("Posterior feasible fraction versus observed stress")
    plt.legend(frameon=True)
    savefig(save_path)


# ============================================================
# Figure 2a
# Parameter recoverability: coverage90
# File used:
#   calibration_benchmark_parameter_recovery_summary.csv
# Meaning:
#   Which parameters are more recoverable in repeated synthetic
#   calibration benchmarks?
# ============================================================

def plot_parameter_coverage(df_param: pd.DataFrame, save_path: str):
    dfp = df_param.sort_values("coverage90", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(dfp["parameter"], dfp["coverage90"])
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage90")
    plt.title("Parameter recoverability: credible-interval coverage")
    plt.xticks(rotation=30, ha="right")
    savefig(save_path)


# ============================================================
# Figure 2b
# Parameter recoverability: interval width
# File used:
#   calibration_benchmark_parameter_recovery_summary.csv
# Meaning:
#   How concentrated is the posterior interval for each parameter?
# ============================================================

def plot_parameter_width(df_param: pd.DataFrame, save_path: str):
    dfp = df_param.sort_values("mean_width90", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(dfp["parameter"], dfp["mean_width90"])
    plt.ylabel("Mean width90")
    plt.title("Parameter recoverability: posterior interval width")
    plt.xticks(rotation=30, ha="right")
    savefig(save_path)


# ============================================================
# Figure 3a
# Observation-fit summary: coverage90
# File used:
#   calibration_benchmark_observation_fit_summary.csv
# Meaning:
#   Does posterior predictive uncertainty cover the selected
#   observables well?
# ============================================================

def plot_observable_coverage(df_obs: pd.DataFrame, save_path: str):
    dfo = df_obs.copy()
    dfo = dfo.set_index("observable").loc[KEY_OBSERVABLES].reset_index()

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(dfo["observable"], dfo["coverage90"])
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage90")
    plt.title("Posterior predictive coverage by observable")
    plt.xticks(rotation=30, ha="right")
    savefig(save_path)


# ============================================================
# Figure 3b
# Observation-fit summary: mean absolute error
# File used:
#   calibration_benchmark_observation_fit_summary.csv
# Meaning:
#   How close is the posterior predictive mean to observations?
# ============================================================

def plot_observable_error(df_obs: pd.DataFrame, save_path: str):
    dfo = df_obs.copy()
    dfo = dfo.set_index("observable").loc[KEY_OBSERVABLES].reset_index()

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(dfo["observable"], dfo["mean_abs_error"])
    plt.ylabel("Mean absolute error")
    plt.title("Posterior predictive fit error by observable")
    plt.xticks(rotation=30, ha="right")
    savefig(save_path)


# ============================================================
# Figure 4 (optional)
# Representative posterior marginals
# File used:
#   benchmark_caseXXX_posterior_samples.csv
# Meaning:
#   What does the posterior distribution look like for a
#   representative calibration case?
# ============================================================

def plot_representative_posterior(case_id: int, save_path: str):
    path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_posterior_samples.csv")
    df = load_csv_optional(path)
    if df is None:
        print(f"[SKIP] Posterior samples not found: {path}")
        return

    cols = [c for c in KEY_PARAMS if c in df.columns]
    if not cols:
        print(f"[SKIP] No KEY_PARAMS found in posterior file: {path}")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(6.2, 2.4 * len(cols)))
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.hist(df[c], bins=40, density=True)
        ax.set_xlabel(c)
        ax.set_ylabel("Density")

    fig.suptitle(f"Representative posterior marginals (case {case_id})", y=0.995)
    savefig(save_path)


# ============================================================
# Figure 5 (optional)
# Representative trace plot
# File used:
#   benchmark_caseXXX_full_chain.csv
# Meaning:
#   Is the MCMC chain stable after burn-in?
# ============================================================

def plot_representative_trace(case_id: int, save_path: str):
    path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_full_chain.csv")
    df = load_csv_optional(path)
    if df is None:
        print(f"[SKIP] Full chain not found: {path}")
        return

    cols = [c for c in KEY_PARAMS if c in df.columns]
    if not cols:
        print(f"[SKIP] No KEY_PARAMS found in full chain file: {path}")
        return

    fig, axes = plt.subplots(len(cols), 1, figsize=(7.0, 2.0 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    x = np.arange(len(df))
    for ax, c in zip(axes, cols):
        ax.plot(x, df[c], linewidth=0.7)
        ax.set_ylabel(c)

    axes[-1].set_xlabel("MCMC iteration")
    fig.suptitle(f"Representative trace plot (case {case_id})", y=0.995)
    savefig(save_path)


# ============================================================
# Figure 6 (optional)
# Representative posterior predictive fit
# Files used:
#   benchmark_caseXXX_posterior_predictive.csv
#   calibration_benchmark_observation_fit.csv
# Meaning:
#   Can the posterior predictive distribution reproduce the
#   observation for a representative case?
#
# NOTE:
#   To avoid unreadable mixed scales, this figure is split into
#   three panels:
#     - stress
#     - temperatures
#     - wall2 + keff
# ============================================================

def plot_representative_predictive_fit(case_id: int, save_path: str):
    pred_path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_posterior_predictive.csv")
    fit_path = os.path.join(OUT_DIR, "calibration_benchmark_observation_fit.csv")

    df_pred = load_csv_optional(pred_path)
    df_fit = load_csv_optional(fit_path)

    if df_pred is None or df_fit is None:
        print(f"[SKIP] Predictive fit files missing for case {case_id}")
        return

    df_fit = df_fit[df_fit["benchmark_case_id"] == case_id].copy()
    if df_fit.empty:
        print(f"[SKIP] No detailed observation fit rows for case {case_id}")
        return

    rows = []
    for obs in KEY_OBSERVABLES:
        if obs not in df_pred.columns:
            continue
        sub = df_fit[df_fit["observable"] == obs]
        if sub.empty:
            continue
        rows.append({
            "observable": obs,
            "y_obs": float(sub["y_obs"].iloc[0]),
            "pred_mean": float(df_pred[obs].mean()),
            "q05": float(df_pred[obs].quantile(0.05)),
            "q95": float(df_pred[obs].quantile(0.95)),
        })

    dfr = pd.DataFrame(rows)
    if dfr.empty:
        print(f"[SKIP] No representative predictive-fit data for case {case_id}")
        return

    panel_groups = [
        ["iteration2_max_global_stress"],
        ["iteration2_max_fuel_temp", "iteration2_max_monolith_temp"],
        ["iteration2_wall2", "iteration2_keff"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for ax, group in zip(axes, panel_groups):
        sub = dfr[dfr["observable"].isin(group)].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = np.arange(len(sub))
        ax.errorbar(
            x,
            sub["pred_mean"],
            yerr=[sub["pred_mean"] - sub["q05"], sub["q95"] - sub["pred_mean"]],
            fmt="o",
            capsize=4,
            label="Posterior predictive mean ± 90% CI"
        )
        ax.scatter(x, sub["y_obs"], marker="x", s=80, label="Observation")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["observable"], rotation=30, ha="right")
        ax.set_ylabel("Value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle(f"Representative posterior predictive fit (case {case_id})", y=1.02)
    savefig(save_path)


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(FIG_DIR)

    # required summary tables
    case_summary_path = os.path.join(OUT_DIR, "calibration_benchmark_case_summary.csv")
    param_summary_path = os.path.join(OUT_DIR, "calibration_benchmark_parameter_recovery_summary.csv")
    obs_summary_path = os.path.join(OUT_DIR, "calibration_benchmark_observation_fit_summary.csv")

    df_case = load_csv_required(case_summary_path)
    df_param = load_csv_required(param_summary_path)
    df_obs = load_csv_required(obs_summary_path)

    rep_case_id = choose_representative_case(df_case)

    # Main figures
    plot_feasible_fraction_vs_stress(
        df_case,
        os.path.join(FIG_DIR, "inverse_feasible_fraction_vs_stress.png")
    )
    plot_parameter_coverage(
        df_param,
        os.path.join(FIG_DIR, "inverse_parameter_coverage90.png")
    )
    plot_parameter_width(
        df_param,
        os.path.join(FIG_DIR, "inverse_parameter_width90.png")
    )
    plot_observable_coverage(
        df_obs,
        os.path.join(FIG_DIR, "inverse_observable_coverage90.png")
    )
    plot_observable_error(
        df_obs,
        os.path.join(FIG_DIR, "inverse_observable_error.png")
    )

    # Optional representative-case figures
    plot_representative_posterior(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_posterior.png")
    )
    plot_representative_trace(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_trace.png")
    )
    plot_representative_predictive_fit(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_predictive_fit.png")
    )

    # Save a small note for plotting branch
    summary = {
        "figure_dir": FIG_DIR,
        "representative_case_id": int(rep_case_id),
        "main_figure_files": [
            "inverse_feasible_fraction_vs_stress.png",
            "inverse_parameter_coverage90.png",
            "inverse_parameter_width90.png",
            "inverse_observable_coverage90.png",
            "inverse_observable_error.png",
        ],
        "optional_figure_files": [
            f"representative_case_{rep_case_id:03d}_posterior.png",
            f"representative_case_{rep_case_id:03d}_trace.png",
            f"representative_case_{rep_case_id:03d}_predictive_fit.png",
        ],
        "notes": [
            "Main-text priority: inverse_feasible_fraction_vs_stress.png",
            "Representative predictive-fit figure is split into 3 panels to avoid unreadable mixed scales.",
            "Trace plot is for appendix / diagnostics, not for main text.",
            "Posterior marginals are more interpretable if plotting branch later adds prior-vs-posterior overlay."
        ]
    }

    with open(os.path.join(FIG_DIR, "plotting_readme.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Inverse plotting script finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()