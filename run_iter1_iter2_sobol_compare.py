import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/home/tjzs/Documents/0310")
DATA_DIR = ROOT / "experiments_phys_levels"

INPUT_FILE = DATA_DIR / "paper_sobol_results_with_ci_all_iters.csv"

OUT_COMPARE = DATA_DIR / "paper_iter1_iter2_sobol_compare.csv"
OUT_SHIFT = DATA_DIR / "paper_iter1_iter2_sobol_shift_summary.csv"
OUT_KEFF = DATA_DIR / "paper_iter2_keff_sobol_summary.csv"

LEVELS = [0, 2]

# 这里只比较真正适合做 iter1 vs iter2 的输出
COMPARE_BASE_OUTPUTS = [
    "avg_fuel_temp",
    "max_fuel_temp",
    "max_monolith_temp",
    "max_global_stress",
    "wall2",
]

# keff 单独处理，只保留 iter2
KEFF_OUTPUT = "iteration2_keff"

TOPK = 3


def classify_stability(row):
    return float(row["ST_ci_low"]) > 0


def load_data():
    return pd.read_csv(INPUT_FILE)


def build_compare_table(df):
    rows = []

    for level in LEVELS:
        for base_out in COMPARE_BASE_OUTPUTS:
            out1 = f"iteration1_{base_out}"
            out2 = f"iteration2_{base_out}"

            sub1 = df[(df["level"] == level) & (df["output"] == out1)].copy()
            sub2 = df[(df["level"] == level) & (df["output"] == out2)].copy()

            if sub1.empty or sub2.empty:
                print(f"[WARN] Missing Sobol rows for level={level}, output={base_out}")
                continue

            sub1["stable"] = sub1.apply(classify_stability, axis=1)
            sub2["stable"] = sub2.apply(classify_stability, axis=1)

            sub1 = sub1.sort_values("ST_mean", ascending=False).reset_index(drop=True)
            sub2 = sub2.sort_values("ST_mean", ascending=False).reset_index(drop=True)

            rank1 = {r["input"]: i + 1 for i, (_, r) in enumerate(sub1.iterrows())}
            rank2 = {r["input"]: i + 1 for i, (_, r) in enumerate(sub2.iterrows())}

            all_inputs = sorted(set(sub1["input"]).union(set(sub2["input"])))

            for inp in all_inputs:
                r1 = sub1[sub1["input"] == inp]
                r2 = sub2[sub2["input"] == inp]

                iter1_st = float(r1["ST_mean"].iloc[0]) if not r1.empty else np.nan
                iter2_st = float(r2["ST_mean"].iloc[0]) if not r2.empty else np.nan

                rows.append({
                    "level": level,
                    "base_output": base_out,
                    "input": inp,

                    "iter1_rank": rank1.get(inp, np.nan),
                    "iter2_rank": rank2.get(inp, np.nan),

                    "iter1_ST_mean": iter1_st,
                    "iter2_ST_mean": iter2_st,

                    "iter1_ST_ci_low": float(r1["ST_ci_low"].iloc[0]) if not r1.empty else np.nan,
                    "iter1_ST_ci_high": float(r1["ST_ci_high"].iloc[0]) if not r1.empty else np.nan,

                    "iter2_ST_ci_low": float(r2["ST_ci_low"].iloc[0]) if not r2.empty else np.nan,
                    "iter2_ST_ci_high": float(r2["ST_ci_high"].iloc[0]) if not r2.empty else np.nan,

                    "iter1_stable": bool(r1["stable"].iloc[0]) if not r1.empty else False,
                    "iter2_stable": bool(r2["stable"].iloc[0]) if not r2.empty else False,

                    "ST_shift": iter2_st - iter1_st if pd.notna(iter1_st) and pd.notna(iter2_st) else np.nan,
                    "rank_shift": (
                        rank2.get(inp, np.nan) - rank1.get(inp, np.nan)
                        if pd.notna(rank1.get(inp, np.nan)) and pd.notna(rank2.get(inp, np.nan))
                        else np.nan
                    ),
                })

    return pd.DataFrame(rows)


def build_shift_summary(df_compare):
    rows = []

    if df_compare.empty:
        return pd.DataFrame(columns=[
            "level", "base_output",
            "iter1_top3", "iter2_top3",
            "top3_overlap_count",
            "stable_iter1_top", "stable_iter2_top",
            "same_top1", "same_top3_set"
        ])

    for level in LEVELS:
        for base_out in COMPARE_BASE_OUTPUTS:
            sub = df_compare[
                (df_compare["level"] == level) &
                (df_compare["base_output"] == base_out)
            ].copy()

            if sub.empty:
                continue

            sub1_top = sub.sort_values("iter1_rank").head(TOPK)
            sub2_top = sub.sort_values("iter2_rank").head(TOPK)

            top1 = list(sub1_top["input"])
            top2 = list(sub2_top["input"])
            overlap = len(set(top1).intersection(set(top2)))

            stable_iter1 = list(
                sub[sub["iter1_stable"]].sort_values("iter1_rank")["input"].head(TOPK)
            )
            stable_iter2 = list(
                sub[sub["iter2_stable"]].sort_values("iter2_rank")["input"].head(TOPK)
            )

            rows.append({
                "level": level,
                "base_output": base_out,
                "iter1_top3": ", ".join(top1),
                "iter2_top3": ", ".join(top2),
                "top3_overlap_count": overlap,
                "stable_iter1_top": ", ".join(stable_iter1),
                "stable_iter2_top": ", ".join(stable_iter2),
                "same_top1": (top1[0] == top2[0]) if len(top1) > 0 and len(top2) > 0 else False,
                "same_top3_set": (set(top1) == set(top2)) if len(top1) == 3 and len(top2) == 3 else False,
            })

    return pd.DataFrame(rows)


def build_iter2_keff_summary(df):
    rows = []

    for level in LEVELS:
        sub = df[(df["level"] == level) & (df["output"] == KEFF_OUTPUT)].copy()
        if sub.empty:
            print(f"[WARN] Missing {KEFF_OUTPUT} for level={level}")
            continue

        sub["stable"] = sub.apply(classify_stability, axis=1)
        sub = sub.sort_values("ST_mean", ascending=False).reset_index(drop=True)

        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            rows.append({
                "level": level,
                "output": KEFF_OUTPUT,
                "input": r["input"],
                "rank_by_ST": rank,
                "ST_mean": float(r["ST_mean"]),
                "ST_ci_low": float(r["ST_ci_low"]),
                "ST_ci_high": float(r["ST_ci_high"]),
                "stable": bool(r["stable"]),
                "recommend_interpretation": "yes" if (bool(r["stable"]) and rank <= TOPK) else "no",
            })

    return pd.DataFrame(rows)


def main():
    df = load_data()

    df_compare = build_compare_table(df)
    df_shift = build_shift_summary(df_compare)
    df_keff = build_iter2_keff_summary(df)

    df_compare.to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")
    df_shift.to_csv(OUT_SHIFT, index=False, encoding="utf-8-sig")
    df_keff.to_csv(OUT_KEFF, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved: {OUT_COMPARE}")
    print(f"[DONE] Saved: {OUT_SHIFT}")
    print(f"[DONE] Saved: {OUT_KEFF}")
    print(df_shift)
    print(df_keff)


if __name__ == "__main__":
    main()