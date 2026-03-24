import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "./experiments_phys_levels"

FULL_CASE = os.path.join(OUT_DIR, "calibration_benchmark_case_summary.csv")
RED_CASE = os.path.join(OUT_DIR, "calibration_benchmark_case_summary_reduced.csv")

FULL_PARAM = os.path.join(OUT_DIR, "calibration_benchmark_parameter_recovery_summary.csv")
RED_PARAM = os.path.join(OUT_DIR, "calibration_benchmark_parameter_recovery_summary_reduced.csv")

FULL_OBS = os.path.join(OUT_DIR, "calibration_benchmark_observation_fit_summary.csv")
RED_OBS = os.path.join(OUT_DIR, "calibration_benchmark_observation_fit_summary_reduced.csv")

OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "paper_inverse_full_vs_reduced_summary.csv")
OUT_PARAM_CSV = os.path.join(OUT_DIR, "paper_inverse_full_vs_reduced_parameter_table.csv")
OUT_OBS_CSV = os.path.join(OUT_DIR, "paper_inverse_full_vs_reduced_observable_table.csv")
OUT_JSON = os.path.join(OUT_DIR, "paper_inverse_full_vs_reduced_summary.json")

KEY_PARAMS = ["E_intercept", "alpha_base", "alpha_slope", "SS316_k_ref"]
KEY_OBS = [
    "iteration2_max_global_stress",
    "iteration2_max_fuel_temp",
    "iteration2_max_monolith_temp",
    "iteration2_wall2",
    "iteration2_keff",
]


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def read_case_summary(path: str, tag: str):
    df = pd.read_csv(path)
    row = {
        "run_tag": tag,
        "n_cases": int(len(df)),
        "mean_accept_rate": float(df["accept_rate"].mean()),
        "std_accept_rate": float(df["accept_rate"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_obs_fit_error": float(df["mean_abs_obs_fit_error"].mean()),
        "std_obs_fit_error": float(df["mean_abs_obs_fit_error"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_obs_coverage90": float(df["obs_coverage90_mean"].mean()),
        "mean_feasible_fraction_110": float(df["feasible_fraction_110"].mean()),
        "mean_feasible_fraction_120": float(df["feasible_fraction_120"].mean()),
        "mean_feasible_fraction_131": float(df["feasible_fraction_131"].mean()),
        "mean_obs_stress": float(df["obs_stress"].mean()),
        "stress_min": float(df["obs_stress"].min()),
        "stress_max": float(df["obs_stress"].max()),
    }
    return df, row


def read_param_summary(path: str, tag: str):
    df = pd.read_csv(path)
    if "parameter" not in df.columns:
        raise ValueError(f"`parameter` column not found in {path}")
    df = df[df["parameter"].isin(KEY_PARAMS)].copy()
    df["run_tag"] = tag
    return df


def read_obs_summary(path: str, tag: str):
    df = pd.read_csv(path)
    if "observable" not in df.columns:
        raise ValueError(f"`observable` column not found in {path}")
    df = df[df["observable"].isin(KEY_OBS)].copy()
    df["run_tag"] = tag
    return df


def build_param_compare(df_full: pd.DataFrame, df_red: pd.DataFrame):
    rows = []
    for p in KEY_PARAMS:
        a = df_full[df_full["parameter"] == p]
        b = df_red[df_red["parameter"] == p]

        if a.empty and b.empty:
            continue

        row = {"parameter": p}

        if not a.empty:
            row.update({
                "full_mean_abs_error": float(a["mean_abs_error"].iloc[0]),
                "full_mean_width90": float(a["mean_width90"].iloc[0]),
                "full_coverage90": float(a["coverage90"].iloc[0]),
                "full_coverage50": float(a["coverage50"].iloc[0]),
            })
        else:
            row.update({
                "full_mean_abs_error": np.nan,
                "full_mean_width90": np.nan,
                "full_coverage90": np.nan,
                "full_coverage50": np.nan,
            })

        if not b.empty:
            row.update({
                "reduced_mean_abs_error": float(b["mean_abs_error"].iloc[0]),
                "reduced_mean_width90": float(b["mean_width90"].iloc[0]),
                "reduced_coverage90": float(b["coverage90"].iloc[0]),
                "reduced_coverage50": float(b["coverage50"].iloc[0]),
            })
        else:
            row.update({
                "reduced_mean_abs_error": np.nan,
                "reduced_mean_width90": np.nan,
                "reduced_coverage90": np.nan,
                "reduced_coverage50": np.nan,
            })

        if np.isfinite(row["full_mean_width90"]) and np.isfinite(row["reduced_mean_width90"]):
            row["width90_ratio_reduced_over_full"] = row["reduced_mean_width90"] / row["full_mean_width90"]
        else:
            row["width90_ratio_reduced_over_full"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_obs_compare(df_full: pd.DataFrame, df_red: pd.DataFrame):
    rows = []
    for obs in KEY_OBS:
        a = df_full[df_full["observable"] == obs]
        b = df_red[df_red["observable"] == obs]

        if a.empty and b.empty:
            continue

        row = {"observable": obs}

        if not a.empty:
            row.update({
                "full_mean_abs_error": float(a["mean_abs_error"].iloc[0]),
                "full_mean_width90": float(a["mean_width90"].iloc[0]),
                "full_coverage90": float(a["coverage90"].iloc[0]),
                "full_coverage95": float(a["coverage95"].iloc[0]),
            })
        else:
            row.update({
                "full_mean_abs_error": np.nan,
                "full_mean_width90": np.nan,
                "full_coverage90": np.nan,
                "full_coverage95": np.nan,
            })

        if not b.empty:
            row.update({
                "reduced_mean_abs_error": float(b["mean_abs_error"].iloc[0]),
                "reduced_mean_width90": float(b["mean_width90"].iloc[0]),
                "reduced_coverage90": float(b["coverage90"].iloc[0]),
                "reduced_coverage95": float(b["coverage95"].iloc[0]),
            })
        else:
            row.update({
                "reduced_mean_abs_error": np.nan,
                "reduced_mean_width90": np.nan,
                "reduced_coverage90": np.nan,
                "reduced_coverage95": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    for p in [FULL_CASE, RED_CASE, FULL_PARAM, RED_PARAM, FULL_OBS, RED_OBS]:
        require_file(p)

    _, full_case_row = read_case_summary(FULL_CASE, "full")
    _, red_case_row = read_case_summary(RED_CASE, "reduced")

    df_summary = pd.DataFrame([full_case_row, red_case_row])
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    df_full_param = read_param_summary(FULL_PARAM, "full")
    df_red_param = read_param_summary(RED_PARAM, "reduced")
    df_param_compare = build_param_compare(df_full_param, df_red_param)
    df_param_compare.to_csv(OUT_PARAM_CSV, index=False, encoding="utf-8-sig")

    df_full_obs = read_obs_summary(FULL_OBS, "full")
    df_red_obs = read_obs_summary(RED_OBS, "reduced")
    df_obs_compare = build_obs_compare(df_full_obs, df_red_obs)
    df_obs_compare.to_csv(OUT_OBS_CSV, index=False, encoding="utf-8-sig")

    summary_json = {
        "full": full_case_row,
        "reduced": red_case_row,
        "main_takeaway": (
            "Use reduced inverse as an interpretability-enhanced formulation "
            "only if feasible-fraction conclusions remain close to the full inverse "
            "while dominant-parameter posterior widths become narrower."
        ),
        "recommended_key_parameters": KEY_PARAMS,
        "recommended_key_observables": KEY_OBS,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved:")
    print(" -", OUT_SUMMARY_CSV)
    print(" -", OUT_PARAM_CSV)
    print(" -", OUT_OBS_CSV)
    print(" -", OUT_JSON)


if __name__ == "__main__":
    main()