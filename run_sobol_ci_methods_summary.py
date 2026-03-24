# run_sobol_ci_methods_summary.py
import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "./experiments_phys_levels"

INPUT_FILE = os.path.join(OUT_DIR, "paper_sobol_results_with_ci.csv")

OUT_METHODS_CSV = os.path.join(OUT_DIR, "paper_sobol_methods_ready_summary.csv")
OUT_RESULTS_CSV = os.path.join(OUT_DIR, "paper_sobol_results_ready_top_factors.csv")
OUT_JSON = os.path.join(OUT_DIR, "paper_sobol_methods_ready_summary.json")

# -----------------------------
# User settings
# -----------------------------
PRIMARY_OUTPUTS = [
    "iteration2_max_global_stress",
    "iteration2_keff",
]

MAIN_LEVELS = [0, 2]

# A factor is considered "stable positive" if:
#   - CI lower bound > 0
# A factor is considered "dominant" if:
#   - ST_mean is among top-k
TOPK_PER_OUTPUT = 3

# Ignore tiny ST when ranking? optional
MIN_ST_FOR_REPORT = 0.0


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def classify_row(row):
    s1_low = float(row["S1_ci_low"])
    s1_high = float(row["S1_ci_high"])
    s1_mean = float(row["S1_raw_mean"])

    st_low = float(row["ST_ci_low"])
    st_high = float(row["ST_ci_high"])
    st_mean = float(row["ST_mean"])

    if s1_low > 0:
        s1_flag = "stable_positive"
    elif s1_high < 0:
        s1_flag = "stable_negative"
    else:
        s1_flag = "crosses_zero"

    if st_low > 0:
        st_flag = "stable_positive"
    elif st_high < 0:
        st_flag = "stable_negative"
    else:
        st_flag = "crosses_zero"

    interpretable = bool((st_low > 0) and (st_mean > MIN_ST_FOR_REPORT))

    return pd.Series({
        "S1_flag": s1_flag,
        "ST_flag": st_flag,
        "interpretable_ST": interpretable,
        "S1_crosses_zero": bool(s1_low <= 0 <= s1_high),
        "ST_crosses_zero": bool(st_low <= 0 <= st_high),
    })


def build_methods_ready_table(df: pd.DataFrame):
    df2 = df.copy()
    class_df = df2.apply(classify_row, axis=1)
    df2 = pd.concat([df2, class_df], axis=1)

    rows = []
    for level in sorted(df2["level"].unique()):
        for output in df2.loc[df2["level"] == level, "output"].unique():
            sub = df2[(df2["level"] == level) & (df2["output"] == output)].copy()
            sub = sub.sort_values("ST_mean", ascending=False).reset_index(drop=True)

            for rank, (_, r) in enumerate(sub.iterrows(), start=1):
                rows.append({
                    "level": int(level),
                    "output": r["output"],
                    "input": r["input"],
                    "rank_by_ST": int(rank),
                    "S1_raw_mean": float(r["S1_raw_mean"]),
                    "S1_raw_std": float(r["S1_raw_std"]),
                    "S1_ci_low": float(r["S1_ci_low"]),
                    "S1_ci_high": float(r["S1_ci_high"]),
                    "S1_flag": r["S1_flag"],
                    "ST_mean": float(r["ST_mean"]),
                    "ST_std": float(r["ST_std"]),
                    "ST_ci_low": float(r["ST_ci_low"]),
                    "ST_ci_high": float(r["ST_ci_high"]),
                    "ST_flag": r["ST_flag"],
                    "interpretable_ST": bool(r["interpretable_ST"]),
                    "recommend_interpretation": (
                        "yes" if bool(r["interpretable_ST"]) and rank <= TOPK_PER_OUTPUT else "no"
                    ),
                    "note": (
                        "top stable contributor"
                        if bool(r["interpretable_ST"]) and rank <= TOPK_PER_OUTPUT
                        else ("CI crosses zero; do not over-interpret" if bool(r["S1_crosses_zero"]) else "")
                    )
                })
    return pd.DataFrame(rows)


def build_results_ready_table(df_methods: pd.DataFrame):
    rows = []
    for level in MAIN_LEVELS:
        for output in PRIMARY_OUTPUTS:
            sub = df_methods[
                (df_methods["level"] == level)
                & (df_methods["output"] == output)
                & (df_methods["recommend_interpretation"] == "yes")
            ].copy()

            sub = sub.sort_values("rank_by_ST", ascending=True).head(TOPK_PER_OUTPUT)

            for _, r in sub.iterrows():
                rows.append({
                    "level": int(r["level"]),
                    "output": r["output"],
                    "input": r["input"],
                    "rank_by_ST": int(r["rank_by_ST"]),
                    "ST_mean": float(r["ST_mean"]),
                    "ST_ci_low": float(r["ST_ci_low"]),
                    "ST_ci_high": float(r["ST_ci_high"]),
                    "S1_raw_mean": float(r["S1_raw_mean"]),
                    "S1_ci_low": float(r["S1_ci_low"]),
                    "S1_ci_high": float(r["S1_ci_high"]),
                    "interpretation_label": "stable dominant contributor",
                })
    return pd.DataFrame(rows)


def build_json_summary(df_methods: pd.DataFrame, df_results: pd.DataFrame):
    out = {
        "source_file": INPUT_FILE,
        "primary_outputs": PRIMARY_OUTPUTS,
        "main_levels": MAIN_LEVELS,
        "topk_per_output": TOPK_PER_OUTPUT,
        "interpretation_rule": {
            "dominant": "top contributors ranked by ST_mean",
            "stable": "ST_ci_low > 0",
            "caution": "Rows whose CI crosses zero should not be over-interpreted"
        },
        "summary_by_output_level": {},
    }

    for level in MAIN_LEVELS:
        for output in PRIMARY_OUTPUTS:
            key = f"level{level}__{output}"
            sub = df_results[(df_results["level"] == level) & (df_results["output"] == output)].copy()

            out["summary_by_output_level"][key] = []
            for _, r in sub.iterrows():
                out["summary_by_output_level"][key].append({
                    "input": r["input"],
                    "rank_by_ST": int(r["rank_by_ST"]),
                    "ST_mean": float(r["ST_mean"]),
                    "ST_ci": [float(r["ST_ci_low"]), float(r["ST_ci_high"])],
                    "S1_raw_mean": float(r["S1_raw_mean"]),
                    "S1_ci": [float(r["S1_ci_low"]), float(r["S1_ci_high"])],
                    "label": r["interpretation_label"],
                })

    caution_rows = df_methods[
        (df_methods["level"].isin(MAIN_LEVELS))
        & (df_methods["output"].isin(PRIMARY_OUTPUTS))
        & ((df_methods["S1_flag"] == "crosses_zero") | (df_methods["ST_flag"] == "crosses_zero"))
    ][["level", "output", "input", "S1_flag", "ST_flag"]].copy()

    out["caution_rows"] = caution_rows.to_dict(orient="records")
    return out


def main():
    require_file(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_cols = [
        "output", "input", "level",
        "S1_raw_mean", "S1_raw_std", "S1_ci_low", "S1_ci_high",
        "ST_mean", "ST_std", "ST_ci_low", "ST_ci_high"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

    df_methods = build_methods_ready_table(df)
    df_results = build_results_ready_table(df_methods)
    js = build_json_summary(df_methods, df_results)

    df_methods.to_csv(OUT_METHODS_CSV, index=False, encoding="utf-8-sig")
    df_results.to_csv(OUT_RESULTS_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved:")
    print(" -", OUT_METHODS_CSV)
    print(" -", OUT_RESULTS_CSV)
    print(" -", OUT_JSON)


if __name__ == "__main__":
    main()