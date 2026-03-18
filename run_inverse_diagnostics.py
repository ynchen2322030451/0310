# run_inverse_diagnostics.py
# ============================================================
# Inverse UQ diagnostics and figure export for paper / meeting
#
# Reads benchmark summary CSVs and generates:
#   1) observed stress vs feasible fraction
#   2) parameter recovery summary
#   3) observable fit summary
#   4) representative posterior / predictive plots (if per-case files exist)
#
# Optional:
#   - trace plot, only if full chain is saved
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_experiment_config import OUT_DIR, PRIMARY_STRESS_THRESHOLD

# -----------------------------
# User settings
# -----------------------------
REPRESENTATIVE_CASE_MODE = "closest_to_threshold"  # "closest_to_threshold", "lowest_stress", "highest_stress", "manual"
MANUAL_CASE_ID = 3
RUN_TAG = "reduced"   # or "full"
RUN_SUFFIX = f"_{RUN_TAG}"
# only used if files exist
KEY_PARAMS = ["E_intercept", "alpha_base", "alpha_slope", "SS316_k_ref"]
KEY_OBSERVABLES = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]

FIG_DIR = os.path.join(OUT_DIR, f"paper_inverse_figures{RUN_SUFFIX}")
SUMMARY_JSON = os.path.join(OUT_DIR, f"inverse_diagnostics_summary{RUN_SUFFIX}.json")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv_must(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def choose_representative_case(df_case: pd.DataFrame) -> int:
    if REPRESENTATIVE_CASE_MODE == "manual":
        return int(MANUAL_CASE_ID)

    if REPRESENTATIVE_CASE_MODE == "lowest_stress":
        row = df_case.iloc[df_case["obs_stress"].argmin()]
        return int(row["benchmark_case_id"])

    if REPRESENTATIVE_CASE_MODE == "highest_stress":
        row = df_case.iloc[df_case["obs_stress"].argmax()]
        return int(row["benchmark_case_id"])

    if REPRESENTATIVE_CASE_MODE == "closest_to_threshold":
        row = df_case.iloc[(df_case["obs_stress"] - PRIMARY_STRESS_THRESHOLD).abs().argmin()]
        return int(row["benchmark_case_id"])

    raise ValueError(f"Unsupported REPRESENTATIVE_CASE_MODE: {REPRESENTATIVE_CASE_MODE}")


def plot_feasible_fraction_vs_stress(df_case: pd.DataFrame, save_path: str):
    plt.figure(figsize=(6.0, 4.5))
    plt.scatter(df_case["obs_stress"], df_case["feasible_fraction_131"], s=45)
    plt.axvline(PRIMARY_STRESS_THRESHOLD, linestyle="--", linewidth=1.2)
    plt.xlabel("Observed stress (MPa)")
    plt.ylabel("Feasible posterior fraction at 131 MPa")
    plt.title("Posterior feasible fraction versus observed stress")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_parameter_recovery_summary(df_param: pd.DataFrame, save_path: str):
    # sort by coverage90 descending then width
    dfp = df_param.copy()
    dfp = dfp.sort_values(["coverage90", "mean_width90"], ascending=[False, True]).reset_index(drop=True)

    x = np.arange(len(dfp))
    width = 0.42

    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.bar(x - width/2, dfp["coverage90"], width=width, label="Coverage90")
    ax1.set_ylabel("Coverage90")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, dfp["mean_width90"], width=width, alpha=0.7, label="Mean width90")
    ax2.set_ylabel("Mean width90")

    ax1.set_xticks(x)
    ax1.set_xticklabels(dfp["parameter"], rotation=30, ha="right")
    ax1.set_title("Parameter recoverability summary")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_observation_fit_summary(df_obs: pd.DataFrame, save_path: str):
    dfo = df_obs.copy()
    dfo = dfo.set_index("observable").loc[KEY_OBSERVABLES].reset_index()

    x = np.arange(len(dfo))
    width = 0.42

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.bar(x - width/2, dfo["coverage90"], width=width, label="Coverage90")
    ax1.set_ylabel("Coverage90")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, dfo["mean_abs_error"], width=width, alpha=0.7, label="Mean abs. error")
    ax2.set_ylabel("Mean abs. error")

    ax1.set_xticks(x)
    ax1.set_xticklabels(dfo["observable"], rotation=30, ha="right")
    ax1.set_title("Posterior predictive fit summary")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_representative_posterior_if_available(case_id: int, save_path: str):
    """
    Requires:
      benchmark_caseXXX_posterior_samples.csv
    """
    path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_posterior_samples{RUN_SUFFIX}.csv")
    if not os.path.exists(path):
        return False

    df = pd.read_csv(path)
    cols = [c for c in KEY_PARAMS if c in df.columns]
    if not cols:
        return False

    fig, axes = plt.subplots(len(cols), 1, figsize=(6.0, 2.2 * len(cols)))
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.hist(df[c], bins=40, density=True)
        ax.set_xlabel(c)
        ax.set_ylabel("Density")

    fig.suptitle(f"Representative posterior marginals (case {case_id})", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return True


def plot_representative_trace_if_available(case_id: int, save_path: str):
    """
    Requires:
      benchmark_caseXXX_full_chain.csv
    """
    path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_full_chain{RUN_SUFFIX}.csv")
    if not os.path.exists(path):
        return False

    df = pd.read_csv(path)
    cols = [c for c in KEY_PARAMS if c in df.columns]
    if not cols:
        return False

    fig, axes = plt.subplots(len(cols), 1, figsize=(7.0, 2.0 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    x = np.arange(len(df))
    for ax, c in zip(axes, cols):
        ax.plot(x, df[c], linewidth=0.6)
        ax.set_ylabel(c)

    axes[-1].set_xlabel("MCMC iteration")
    fig.suptitle(f"Representative trace plot (case {case_id})", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return True


def plot_representative_predictive_fit_if_available(case_id: int, save_path: str):
    """
    Requires:
      benchmark_caseXXX_posterior_predictive.csv
      plus row subset from observation_fit detailed csv if available
    """
    pred_path = os.path.join(OUT_DIR, f"benchmark_case{case_id:03d}_posterior_predictive{RUN_SUFFIX}.csv")
    fit_path = os.path.join(OUT_DIR, f"calibration_benchmark_observation_fit{RUN_SUFFIX}.csv")

    if not os.path.exists(pred_path):
        return False
    if not os.path.exists(fit_path):
        return False

    df_pred = pd.read_csv(pred_path)
    df_fit = pd.read_csv(fit_path)
    df_fit = df_fit[df_fit["benchmark_case_id"] == case_id].copy()

    if df_fit.empty:
        return False

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

    if not rows:
        return False

    dfr = pd.DataFrame(rows)
    x = np.arange(len(dfr))

    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(
        x,
        dfr["pred_mean"],
        yerr=[dfr["pred_mean"] - dfr["q05"], dfr["q95"] - dfr["pred_mean"]],
        fmt="o",
        capsize=4,
        label="Posterior predictive mean ± 90% CI"
    )
    plt.scatter(x, dfr["y_obs"], marker="x", s=70, label="Observation")
    plt.xticks(x, dfr["observable"], rotation=30, ha="right")
    plt.ylabel("Value")
    plt.title(f"Representative posterior predictive fit (case {case_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def main():
    ensure_dir(FIG_DIR)

    case_summary_path = os.path.join(
        OUT_DIR, f"calibration_benchmark_case_summary{RUN_SUFFIX}.csv"
    )
    param_summary_path = os.path.join(
        OUT_DIR, f"calibration_benchmark_parameter_recovery_summary{RUN_SUFFIX}.csv"
    )
    obs_summary_path = os.path.join(
        OUT_DIR, f"calibration_benchmark_observation_fit_summary{RUN_SUFFIX}.csv"
    )

    df_case = load_csv_must(case_summary_path)
    df_param = load_csv_must(param_summary_path)
    df_obs = load_csv_must(obs_summary_path)

    rep_case_id = choose_representative_case(df_case)

    # Main paper-oriented plots
    plot_feasible_fraction_vs_stress(
        df_case,
        os.path.join(FIG_DIR, f"inverse_feasible_fraction_vs_stress{RUN_SUFFIX}.png")
    )
    plot_parameter_recovery_summary(
        df_param,
        os.path.join(FIG_DIR, f"inverse_parameter_recoverability{RUN_SUFFIX}.png")
    )
    plot_observation_fit_summary(
        df_obs,
        os.path.join(FIG_DIR, f"inverse_observation_fit_summary{RUN_SUFFIX}.png")
    )

    # Optional representative-case plots
    has_post = plot_representative_posterior_if_available(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_posterior{RUN_SUFFIX}.png")
    )
    has_trace = plot_representative_trace_if_available(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_trace{RUN_SUFFIX}.png")
    )
    has_predfit = plot_representative_predictive_fit_if_available(
        rep_case_id,
        os.path.join(FIG_DIR, f"representative_case_{rep_case_id:03d}_predictive_fit{RUN_SUFFIX}.png")
    )

    summary = {
        "representative_case_id": int(rep_case_id),
        "n_benchmark_cases": int(len(df_case)),
        "mean_accept_rate": float(df_case["accept_rate"].mean()),
        "mean_obs_fit_error": float(df_case["mean_abs_obs_fit_error"].mean()),
        "mean_obs_coverage90": float(df_case["obs_coverage90_mean"].mean()),
        "mean_feasible_fraction_110": float(df_case["feasible_fraction_110"].mean()),
        "mean_feasible_fraction_120": float(df_case["feasible_fraction_120"].mean()),
        "mean_feasible_fraction_131": float(df_case["feasible_fraction_131"].mean()),
        "trace_plot_available": bool(has_trace),
        "representative_posterior_available": bool(has_post),
        "representative_predictive_fit_available": bool(has_predfit),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Inverse diagnostics exported to:", FIG_DIR)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()