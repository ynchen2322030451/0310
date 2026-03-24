# run_inverse_full_vs_reduced_compare.py
# ============================================================
# Compare full inverse vs reduced_maintext inverse
#
# Inputs:
#   - calibration_benchmark_case_summary_full.csv
#   - calibration_benchmark_case_summary_reduced_maintext.csv
#   - calibration_benchmark_parameter_recovery_summary_full.csv
#   - calibration_benchmark_parameter_recovery_summary_reduced_maintext.csv
#   - calibration_benchmark_observation_fit_summary_full.csv
#   - calibration_benchmark_observation_fit_summary_reduced_maintext.csv
#
# Outputs:
#   - paper_inverse_full_vs_reduced_summary.csv
#   - paper_inverse_full_vs_reduced_parameter_table.csv
#   - paper_inverse_full_vs_reduced_observable_table.csv
#   - paper_inverse_full_vs_reduced_summary.json
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/tjzs/Documents/0310")
OUT_DIR = ROOT / "experiments_phys_levels"

FULL_CASE = OUT_DIR / "calibration_benchmark_case_summary_full.csv"
RED_CASE = OUT_DIR / "calibration_benchmark_case_summary_reduced_maintext.csv"

FULL_PARAM = OUT_DIR / "calibration_benchmark_parameter_recovery_summary_full.csv"
RED_PARAM = OUT_DIR / "calibration_benchmark_parameter_recovery_summary_reduced_maintext.csv"

FULL_OBS = OUT_DIR / "calibration_benchmark_observation_fit_summary_full.csv"
RED_OBS = OUT_DIR / "calibration_benchmark_observation_fit_summary_reduced_maintext.csv"

OUT_SUMMARY = OUT_DIR / "paper_inverse_full_vs_reduced_summary.csv"
OUT_PARAM_TABLE = OUT_DIR / "paper_inverse_full_vs_reduced_parameter_table.csv"
OUT_OBS_TABLE = OUT_DIR / "paper_inverse_full_vs_reduced_observable_table.csv"
OUT_JSON = OUT_DIR / "paper_inverse_full_vs_reduced_summary.json"


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_all():
    for p in [FULL_CASE, RED_CASE, FULL_PARAM, RED_PARAM, FULL_OBS, RED_OBS]:
        require(p)

    df_case_full = pd.read_csv(FULL_CASE)
    df_case_red = pd.read_csv(RED_CASE)

    df_param_full = pd.read_csv(FULL_PARAM)
    df_param_red = pd.read_csv(RED_PARAM)

    df_obs_full = pd.read_csv(FULL_OBS)
    df_obs_red = pd.read_csv(RED_OBS)

    return df_case_full, df_case_red, df_param_full, df_param_red, df_obs_full, df_obs_red


def build_summary(df_case_full, df_case_red):
    rows = []

    def row_from_case(df, tag):
        return {
            "run_tag": tag,
            "n_cases": int(len(df)),
            "mean_accept_rate": float(df["accept_rate"].mean()),
            "mean_obs_fit_error": float(df["mean_abs_obs_fit_error"].mean()),
            "mean_obs_coverage90": float(df["obs_coverage90_mean"].mean()),
            "mean_feasible_fraction_110": float(df["feasible_fraction_110"].mean()),
            "mean_feasible_fraction_120": float(df["feasible_fraction_120"].mean()),
            "mean_feasible_fraction_131": float(df["feasible_fraction_131"].mean()),
        }

    rows.append(row_from_case(df_case_full, "full"))
    rows.append(row_from_case(df_case_red, "reduced_maintext"))

    df_summary = pd.DataFrame(rows)

    # add delta row: reduced - full
    full_row = df_summary[df_summary["run_tag"] == "full"].iloc[0]
    red_row = df_summary[df_summary["run_tag"] == "reduced_maintext"].iloc[0]

    delta = {
        "run_tag": "delta_reduced_minus_full",
        "n_cases": int(red_row["n_cases"] - full_row["n_cases"]),
        "mean_accept_rate": float(red_row["mean_accept_rate"] - full_row["mean_accept_rate"]),
        "mean_obs_fit_error": float(red_row["mean_obs_fit_error"] - full_row["mean_obs_fit_error"]),
        "mean_obs_coverage90": float(red_row["mean_obs_coverage90"] - full_row["mean_obs_coverage90"]),
        "mean_feasible_fraction_110": float(red_row["mean_feasible_fraction_110"] - full_row["mean_feasible_fraction_110"]),
        "mean_feasible_fraction_120": float(red_row["mean_feasible_fraction_120"] - full_row["mean_feasible_fraction_120"]),
        "mean_feasible_fraction_131": float(red_row["mean_feasible_fraction_131"] - full_row["mean_feasible_fraction_131"]),
    }

    df_summary = pd.concat([df_summary, pd.DataFrame([delta])], ignore_index=True)
    return df_summary


def build_parameter_table(df_param_full, df_param_red):
    df_full = df_param_full.copy()
    df_red = df_param_red.copy()

    df_full = df_full.rename(columns={
        "mean_abs_error": "full_mean_abs_error",
        "mean_width90": "full_mean_width90",
        "coverage90": "full_coverage90",
        "coverage50": "full_coverage50",
    })

    df_red = df_red.rename(columns={
        "mean_abs_error": "reduced_mean_abs_error",
        "mean_width90": "reduced_mean_width90",
        "coverage90": "reduced_coverage90",
        "coverage50": "reduced_coverage50",
    })

    df = pd.merge(df_full, df_red, on="parameter", how="outer")

    df["delta_mean_abs_error"] = df["reduced_mean_abs_error"] - df["full_mean_abs_error"]
    df["delta_mean_width90"] = df["reduced_mean_width90"] - df["full_mean_width90"]
    df["delta_coverage90"] = df["reduced_coverage90"] - df["full_coverage90"]
    df["delta_coverage50"] = df["reduced_coverage50"] - df["full_coverage50"]

    return df.sort_values("parameter").reset_index(drop=True)


def build_observable_table(df_obs_full, df_obs_red):
    df_full = df_obs_full.copy()
    df_red = df_obs_red.copy()

    df_full = df_full.rename(columns={
        "mean_abs_error": "full_mean_abs_error",
        "mean_width90": "full_mean_width90",
        "coverage90": "full_coverage90",
        "coverage95": "full_coverage95",
    })

    df_red = df_red.rename(columns={
        "mean_abs_error": "reduced_mean_abs_error",
        "mean_width90": "reduced_mean_width90",
        "coverage90": "reduced_coverage90",
        "coverage95": "reduced_coverage95",
    })

    df = pd.merge(df_full, df_red, on="observable", how="outer")

    df["delta_mean_abs_error"] = df["reduced_mean_abs_error"] - df["full_mean_abs_error"]
    df["delta_mean_width90"] = df["reduced_mean_width90"] - df["full_mean_width90"]
    df["delta_coverage90"] = df["reduced_coverage90"] - df["full_coverage90"]
    df["delta_coverage95"] = df["reduced_coverage95"] - df["full_coverage95"]

    return df.sort_values("observable").reset_index(drop=True)


def build_json(df_summary, df_param, df_obs):
    js = {
        "comparison": "full vs reduced_maintext",
        "headline": {
            "full_mean_obs_fit_error": float(
                df_summary[df_summary["run_tag"] == "full"]["mean_obs_fit_error"].iloc[0]
            ),
            "reduced_mean_obs_fit_error": float(
                df_summary[df_summary["run_tag"] == "reduced_maintext"]["mean_obs_fit_error"].iloc[0]
            ),
            "full_mean_feasible_fraction_131": float(
                df_summary[df_summary["run_tag"] == "full"]["mean_feasible_fraction_131"].iloc[0]
            ),
            "reduced_mean_feasible_fraction_131": float(
                df_summary[df_summary["run_tag"] == "reduced_maintext"]["mean_feasible_fraction_131"].iloc[0]
            ),
        },
        "largest_parameter_degradation_by_coverage90": [],
        "largest_observable_error_increase": [],
    }

    param_rank = df_param.sort_values("delta_coverage90").head(5)
    for _, r in param_rank.iterrows():
        js["largest_parameter_degradation_by_coverage90"].append({
            "parameter": r["parameter"],
            "full_coverage90": float(r["full_coverage90"]) if pd.notna(r["full_coverage90"]) else None,
            "reduced_coverage90": float(r["reduced_coverage90"]) if pd.notna(r["reduced_coverage90"]) else None,
            "delta_coverage90": float(r["delta_coverage90"]) if pd.notna(r["delta_coverage90"]) else None,
        })

    obs_rank = df_obs.sort_values("delta_mean_abs_error", ascending=False).head(5)
    for _, r in obs_rank.iterrows():
        js["largest_observable_error_increase"].append({
            "observable": r["observable"],
            "full_mean_abs_error": float(r["full_mean_abs_error"]) if pd.notna(r["full_mean_abs_error"]) else None,
            "reduced_mean_abs_error": float(r["reduced_mean_abs_error"]) if pd.notna(r["reduced_mean_abs_error"]) else None,
            "delta_mean_abs_error": float(r["delta_mean_abs_error"]) if pd.notna(r["delta_mean_abs_error"]) else None,
        })

    return js


def main():
    df_case_full, df_case_red, df_param_full, df_param_red, df_obs_full, df_obs_red = load_all()

    df_summary = build_summary(df_case_full, df_case_red)
    df_param = build_parameter_table(df_param_full, df_param_red)
    df_obs = build_observable_table(df_obs_full, df_obs_red)
    js = build_json(df_summary, df_param, df_obs)

    df_summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    df_param.to_csv(OUT_PARAM_TABLE, index=False, encoding="utf-8-sig")
    df_obs.to_csv(OUT_OBS_TABLE, index=False, encoding="utf-8-sig")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved:")
    print(" -", OUT_SUMMARY)
    print(" -", OUT_PARAM_TABLE)
    print(" -", OUT_OBS_TABLE)
    print(" -", OUT_JSON)
    print(df_summary)


if __name__ == "__main__":
    main()