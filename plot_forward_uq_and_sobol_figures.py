# plot_forward_uq_and_sobol_figures.py
# ============================================================
# Purpose:
#   Plot forward-UQ and Sobol figures directly from saved result files.
#
# This script DOES NOT retrain any model.
# It only reads saved summary CSVs and exports figures.
#
# Required input files:
#   1) paper_forward_uq_summary.csv
#      - used for: forward-UQ global summary and threshold risk
#
#   2) forward_uq_primary_outputs_level0.csv
#      - used for: primary output distribution summary (baseline)
#
#   3) forward_uq_primary_outputs_level2.csv
#      - used for: primary output distribution summary (regularized)
#
#   4) forward_uq_failure_prob_level0.csv
#      - used for: threshold-risk curve (baseline)
#
#   5) forward_uq_failure_prob_level2.csv
#      - used for: threshold-risk curve (regularized)
#
#   6) forward_uq_cvr_level0.csv
#      - used for: CVR bar chart (baseline)
#
#   7) forward_uq_cvr_level2.csv
#      - used for: CVR bar chart (regularized)
#
#   8) paper_sobol_results.csv
#      - used for: Sobol bar charts
#
# Optional input files:
#   9) forward_uq_joint_stress_keff_level0.csv
#   10) forward_uq_joint_stress_keff_level2.csv
#      - used for: stress-keff joint scatter
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
FIG_DIR = os.path.join(OUT_DIR, "paper_forward_figures_final")

PRIMARY_OUTPUTS = [
    "iteration2_keff",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_max_global_stress",
    "iteration2_wall2",
]

PRIMARY_STRESS_OUTPUT = "iteration2_max_global_stress"
PRIMARY_KEFF_OUTPUT = "iteration2_keff"

THRESHOLDS = [110.0, 120.0, 131.0]

DPI = 300

# Which outputs to show for Sobol
SOBOL_OUTPUTS = [
    "iteration2_max_global_stress",
    "iteration2_keff",
]

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


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 1
# Forward-UQ distribution summary for primary outputs
#
# Files used:
#   - forward_uq_primary_outputs_level0.csv
#   - forward_uq_primary_outputs_level2.csv
#
# Meaning:
#   Compare mean / q05 / q50 / q95 for the five primary outputs
#   between baseline and regularized model.
# ============================================================

def plot_primary_output_distributions(df0: pd.DataFrame, df2: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: stress
    for ax, outputs, title in zip(
        axes,
        [
            ["iteration2_max_global_stress"],
            ["iteration2_max_fuel_temp", "iteration2_max_monolith_temp"],
            ["iteration2_wall2", "iteration2_keff"],
        ],
        [
            "Stress",
            "Temperatures",
            "Geometry + reactivity",
        ]
    ):
        sub0 = df0[df0["output"].isin(outputs)].copy()
        sub2 = df2[df2["output"].isin(outputs)].copy()

        # keep same order as outputs list
        sub0["output"] = pd.Categorical(sub0["output"], categories=outputs, ordered=True)
        sub2["output"] = pd.Categorical(sub2["output"], categories=outputs, ordered=True)
        sub0 = sub0.sort_values("output")
        sub2 = sub2.sort_values("output")

        x = np.arange(len(outputs))
        offset = 0.12

        # baseline
        ax.errorbar(
            x - offset,
            sub0["q50"],
            yerr=[sub0["q50"] - sub0["q05"], sub0["q95"] - sub0["q50"]],
            fmt="o",
            capsize=4,
            label="Baseline (Level0)"
        )

        # regularized
        ax.errorbar(
            x + offset,
            sub2["q50"],
            yerr=[sub2["q50"] - sub2["q05"], sub2["q95"] - sub2["q50"]],
            fmt="s",
            capsize=4,
            label="Regularized (Level2)"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(outputs, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Forward uncertainty propagation: primary output distributions", y=1.03)
    savefig(save_path)


# ============================================================
# Figure 2
# Threshold-risk curve
#
# Files used:
#   - forward_uq_failure_prob_level0.csv
#   - forward_uq_failure_prob_level2.csv
#
# Meaning:
#   Show failure probability vs stress threshold
# ============================================================

def plot_threshold_risk_curve(df0: pd.DataFrame, df2: pd.DataFrame, save_path: str):
    plt.figure(figsize=(6.0, 4.5))
    plt.plot(df0["threshold_MPa"], df0["p_fail"], marker="o", label="Baseline (Level0)")
    plt.plot(df2["threshold_MPa"], df2["p_fail"], marker="s", label="Regularized (Level2)")
    plt.xlabel("Stress threshold (MPa)")
    plt.ylabel("Failure probability")
    plt.title("Stress exceedance probability versus threshold")
    plt.legend(frameon=True)
    savefig(save_path)


# ============================================================
# Figure 3
# CVR / uncertainty amplification
#
# Files used:
#   - forward_uq_cvr_level0.csv
#   - forward_uq_cvr_level2.csv
#
# Meaning:
#   Show which output amplifies uncertainty most strongly
# ============================================================

def plot_cvr_comparison(df0: pd.DataFrame, df2: pd.DataFrame, save_path: str):
    # keep only primary outputs and fixed order
    df0 = df0.set_index("output").loc[PRIMARY_OUTPUTS].reset_index()
    df2 = df2.set_index("output").loc[PRIMARY_OUTPUTS].reset_index()

    x = np.arange(len(PRIMARY_OUTPUTS))
    width = 0.35

    plt.figure(figsize=(7.6, 4.8))
    plt.bar(x - width/2, df0["CVR"], width=width, label="Baseline (Level0)")
    plt.bar(x + width/2, df2["CVR"], width=width, label="Regularized (Level2)")
    plt.xticks(x, PRIMARY_OUTPUTS, rotation=30, ha="right")
    plt.ylabel("CVR")
    plt.title("Uncertainty amplification across primary outputs")
    plt.legend(frameon=True)
    savefig(save_path)


# ============================================================
# Figure 4
# Sobol bars for stress and keff
#
# File used:
#   - paper_sobol_results.csv
#
# Meaning:
#   Show dominant uncertainty sources for the key outputs
# ============================================================

def plot_sobol_bars(df_sobol: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, len(SOBOL_OUTPUTS), figsize=(12.0, 4.8), sharey=False)
    if len(SOBOL_OUTPUTS) == 1:
        axes = [axes]

    for ax, out_name in zip(axes, SOBOL_OUTPUTS):
        sub = df_sobol[df_sobol["output"] == out_name].copy()

        # 0 vs 2 only
        sub = sub[sub["level"].isin([0, 2])]

        # pivot into wide format
        p0 = sub[sub["level"] == 0].set_index("input")
        p2 = sub[sub["level"] == 2].set_index("input")

        inputs = list(p0.index)
        x = np.arange(len(inputs))
        width = 0.35

        ax.bar(x - width/2, p0.loc[inputs, "S1"], width=width, label="Level0 S1")
        ax.bar(x + width/2, p2.loc[inputs, "S1"], width=width, label="Level2 S1")

        ax.set_xticks(x)
        ax.set_xticklabels(inputs, rotation=30, ha="right")
        ax.set_ylabel("First-order Sobol index")
        ax.set_title(out_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Variance-based global sensitivity analysis", y=1.03)
    savefig(save_path)


# ============================================================
# Figure 5 (optional)
# Stress-keff joint scatter
#
# Files used:
#   - forward_uq_joint_stress_keff_level0.csv
#   - forward_uq_joint_stress_keff_level2.csv
#
# Meaning:
#   Show the coupled forward-UQ joint response of stress and keff
# ============================================================

def plot_joint_stress_keff(df0: pd.DataFrame, df2: pd.DataFrame, save_path: str):
    if df0 is None or df2 is None:
        print("[SKIP] joint stress-keff files not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)

    axes[0].scatter(
        df0["iteration2_max_global_stress"],
        df0["iteration2_keff"],
        s=8
    )
    axes[0].set_title("Baseline (Level0)")
    axes[0].set_xlabel("iteration2_max_global_stress")
    axes[0].set_ylabel("iteration2_keff")

    axes[1].scatter(
        df2["iteration2_max_global_stress"],
        df2["iteration2_keff"],
        s=8
    )
    axes[1].set_title("Regularized (Level2)")
    axes[1].set_xlabel("iteration2_max_global_stress")

    fig.suptitle("Joint forward-UQ response of stress and keff", y=1.02)
    savefig(save_path)


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(FIG_DIR)

    # Required files
    df_forward_summary = load_csv_required(os.path.join(OUT_DIR, "paper_forward_uq_summary.csv"))
    df_primary0 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_primary_outputs_level0.csv"))
    df_primary2 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_primary_outputs_level2.csv"))
    df_fail0 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_failure_prob_level0.csv"))
    df_fail2 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_failure_prob_level2.csv"))
    df_cvr0 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_cvr_level0.csv"))
    df_cvr2 = load_csv_required(os.path.join(OUT_DIR, "forward_uq_cvr_level2.csv"))
    df_sobol = load_csv_required(os.path.join(OUT_DIR, "paper_sobol_results.csv"))

    # Optional files
    df_joint0 = load_csv_optional(os.path.join(OUT_DIR, "forward_uq_joint_stress_keff_level0.csv"))
    df_joint2 = load_csv_optional(os.path.join(OUT_DIR, "forward_uq_joint_stress_keff_level2.csv"))

    # Export figures
    plot_primary_output_distributions(
        df_primary0, df_primary2,
        os.path.join(FIG_DIR, "forward_primary_output_distributions.png")
    )

    plot_threshold_risk_curve(
        df_fail0, df_fail2,
        os.path.join(FIG_DIR, "forward_threshold_risk_curve.png")
    )

    plot_cvr_comparison(
        df_cvr0, df_cvr2,
        os.path.join(FIG_DIR, "forward_cvr_comparison.png")
    )

    plot_sobol_bars(
        df_sobol,
        os.path.join(FIG_DIR, "forward_sobol_bars.png")
    )

    plot_joint_stress_keff(
        df_joint0, df_joint2,
        os.path.join(FIG_DIR, "forward_joint_stress_keff.png")
    )

    summary = {
        "figure_dir": FIG_DIR,
        "main_figure_files": [
            "forward_primary_output_distributions.png",
            "forward_threshold_risk_curve.png",
            "forward_cvr_comparison.png",
            "forward_sobol_bars.png",
        ],
        "optional_figure_files": [
            "forward_joint_stress_keff.png",
        ],
        "notes": [
            "Main-text priority: threshold risk curve, CVR comparison, Sobol bars.",
            "Primary-output distribution figure is split into 3 panels to avoid unreadable mixed scales.",
            "Joint stress-keff scatter is optional and more suitable for appendix or discussion.",
            "This script only reads saved result files and does not rerun forward-UQ or Sobol analysis."
        ]
    }

    with open(os.path.join(FIG_DIR, "plotting_readme.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Forward-UQ / Sobol plotting script finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()