import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from paper_experiment_config import (
    OUT_DIR,
    INPUT_COLS,
    OUTPUT_COLS,
    PRIMARY_OUTPUTS,
)

# =========================
# 配置
# =========================

DATA_PATH_CANDIDATES = [
    os.path.join(OUT_DIR, "fixed_split", "train.csv"),
    os.path.join(OUT_DIR, "fixed_split", "val.csv"),
    os.path.join(OUT_DIR, "fixed_split", "test.csv"),
]

SAVE_PATH = os.path.join(OUT_DIR, "dataset_sensitivity_spearman.csv")


# =========================
# 加载数据
# =========================

def load_dataset():
    dfs = []
    for p in DATA_PATH_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"[INFO] loaded: {p}, shape={df.shape}")
    if len(dfs) == 0:
        raise FileNotFoundError("No dataset found.")
    return pd.concat(dfs, ignore_index=True)


# =========================
# 计算 Spearman
# =========================

def compute_spearman(df):
    rows = []

    for out in PRIMARY_OUTPUTS:
        y = df[out].values

        for inp in INPUT_COLS:
            x = df[inp].values

            rho, pval = spearmanr(x, y)

            rows.append({
                "output": out,
                "input": inp,
                "spearman_rho": rho,
                "abs_rho": abs(rho),
                "p_value": pval,
            })

    res = pd.DataFrame(rows)

    # 排序（每个 output 内部）
    res["rank"] = res.groupby("output")["abs_rho"].rank(
        ascending=False, method="first"
    )

    return res


# =========================
# 提取 Top-K
# =========================

def build_topk_summary(df, k=3):
    rows = []

    for out in PRIMARY_OUTPUTS:
        sub = df[df["output"] == out].sort_values("abs_rho", ascending=False)

        top_inputs = sub["input"].values[:k]
        top_rhos = sub["spearman_rho"].values[:k]

        rows.append({
            "output": out,
            "top_inputs": ",".join(top_inputs),
            "top_rhos": ",".join([f"{v:.3f}" for v in top_rhos]),
        })

    return pd.DataFrame(rows)


# =========================
# 主函数
# =========================

def main():
    df = load_dataset()

    print("\n[INFO] computing spearman...")

    res = compute_spearman(df)

    res.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] saved full table: {SAVE_PATH}")

    topk = build_topk_summary(res, k=3)

    topk_path = os.path.join(OUT_DIR, "dataset_sensitivity_top3.csv")
    topk.to_csv(topk_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved top3 summary: {topk_path}")

    print("\n=== TOP3 SUMMARY ===")
    print(topk.to_string(index=False))


if __name__ == "__main__":
    main()