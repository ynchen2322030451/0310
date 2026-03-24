import pandas as pd
from pathlib import Path

ROOT = Path("/home/tjzs/Documents/0310")
DATA_DIR = ROOT / "experiments_phys_levels"
OUT_FILE = DATA_DIR / "paper_iter1_iter2_forward_compare.csv"

LEVELS = [0, 2]

OUTPUTS = [
    "keff",
    "avg_fuel_temp",
    "max_fuel_temp",
    "max_monolith_temp",
    "max_global_stress",
    "wall2",
]

def process_level(level):
    file_path = DATA_DIR / f"forward_uq_all_outputs_level{level}.csv"
    df = pd.read_csv(file_path)

    rows = []

    for out in OUTPUTS:
        name_iter1 = f"iteration1_{out}"
        name_iter2 = f"iteration2_{out}"

        row1 = df[df["output"] == name_iter1]
        row2 = df[df["output"] == name_iter2]

        if row1.empty or row2.empty:
            print(f"[WARN] Missing rows for {out} in level {level}")
            continue

        row1 = row1.iloc[0]
        row2 = row2.iloc[0]

        iter1_mean = float(row1["mean"])
        iter1_std = float(row1["std"])
        iter1_q95 = float(row1["q95"])

        iter2_mean = float(row2["mean"])
        iter2_std = float(row2["std"])
        iter2_q95 = float(row2["q95"])

        rows.append({
            "level": level,
            "output": out,

            "iter1_mean": iter1_mean,
            "iter1_std": iter1_std,
            "iter1_q95": iter1_q95,

            "iter2_mean": iter2_mean,
            "iter2_std": iter2_std,
            "iter2_q95": iter2_q95,

            "mean_shift": iter2_mean - iter1_mean,
            "std_ratio": iter2_std / (iter1_std + 1e-12),
            "q95_shift": iter2_q95 - iter1_q95,

            "mean_shift_direction": (
                "increase" if iter2_mean > iter1_mean else "decrease"
            ),
            "uncertainty_change": (
                "amplified" if iter2_std > iter1_std else "reduced"
            ),
            "tail_change": (
                "higher" if iter2_q95 > iter1_q95 else "lower"
            ),
        })

    return rows

def main():
    all_rows = []

    for lvl in LEVELS:
        rows = process_level(lvl)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved: {OUT_FILE}")
    print(df_out)

if __name__ == "__main__":
    main()