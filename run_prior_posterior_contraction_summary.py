import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "./experiments_phys_levels"

RUN_TAG = "reduced"   # 当前先做 reduced
RUN_SUFFIX = f"_{RUN_TAG}"

N_CASES = 20
PARAMS = ["E_intercept", "alpha_base", "alpha_slope", "SS316_k_ref"]

OUT_DETAIL_CSV = os.path.join(
    OUT_DIR, f"paper_prior_posterior_contraction{RUN_SUFFIX}.csv"
)
OUT_SUMMARY_CSV = os.path.join(
    OUT_DIR, f"paper_prior_posterior_contraction_summary{RUN_SUFFIX}.csv"
)
OUT_JSON = os.path.join(
    OUT_DIR, f"paper_prior_posterior_contraction_summary{RUN_SUFFIX}.json"
)


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def qwidth(v: np.ndarray, ql=0.05, qh=0.95) -> float:
    return float(np.quantile(v, qh) - np.quantile(v, ql))


def summarize_vector(v: np.ndarray):
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "var": float(np.var(v, ddof=1)) if len(v) > 1 else 0.0,
        "q05": float(np.quantile(v, 0.05)),
        "q25": float(np.quantile(v, 0.25)),
        "q50": float(np.quantile(v, 0.50)),
        "q75": float(np.quantile(v, 0.75)),
        "q95": float(np.quantile(v, 0.95)),
        "width90": qwidth(v, 0.05, 0.95),
        "width50": qwidth(v, 0.25, 0.75),
    }


def main():
    rows = []

    for case_id in range(N_CASES):
        prior_path = os.path.join(
            OUT_DIR, f"benchmark_case{case_id:03d}_prior_samples{RUN_SUFFIX}.csv"
        )
        post_path = os.path.join(
            OUT_DIR, f"benchmark_case{case_id:03d}_posterior_samples{RUN_SUFFIX}.csv"
        )

        require_file(prior_path)
        require_file(post_path)

        df_prior = pd.read_csv(prior_path)
        df_post = pd.read_csv(post_path)

        for p in PARAMS:
            if p not in df_prior.columns or p not in df_post.columns:
                continue

            vp = df_prior[p].dropna().to_numpy(dtype=float)
            vq = df_post[p].dropna().to_numpy(dtype=float)

            sp = summarize_vector(vp)
            sq = summarize_vector(vq)

            width90_ratio = np.nan
            if sp["width90"] > 0:
                width90_ratio = sq["width90"] / sp["width90"]

            width50_ratio = np.nan
            if sp["width50"] > 0:
                width50_ratio = sq["width50"] / sp["width50"]

            var_ratio = np.nan
            if sp["var"] > 0:
                var_ratio = sq["var"] / sp["var"]

            rows.append({
                "benchmark_case_id": case_id,
                "parameter": p,

                "prior_mean": sp["mean"],
                "prior_std": sp["std"],
                "prior_var": sp["var"],
                "prior_q05": sp["q05"],
                "prior_q50": sp["q50"],
                "prior_q95": sp["q95"],
                "prior_width90": sp["width90"],
                "prior_width50": sp["width50"],

                "posterior_mean": sq["mean"],
                "posterior_std": sq["std"],
                "posterior_var": sq["var"],
                "posterior_q05": sq["q05"],
                "posterior_q50": sq["q50"],
                "posterior_q95": sq["q95"],
                "posterior_width90": sq["width90"],
                "posterior_width50": sq["width50"],

                "mean_shift_abs": abs(sq["mean"] - sp["mean"]),
                "width90_ratio_post_over_prior": width90_ratio,
                "width50_ratio_post_over_prior": width50_ratio,
                "var_ratio_post_over_prior": var_ratio,
                "contraction_flag_width90": bool(width90_ratio < 1.0) if np.isfinite(width90_ratio) else False,
                "contraction_flag_var": bool(var_ratio < 1.0) if np.isfinite(var_ratio) else False,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    # summary by parameter
    summary_rows = []
    for p in PARAMS:
        sub = df[df["parameter"] == p].copy()
        if sub.empty:
            continue

        summary_rows.append({
            "parameter": p,
            "n_cases": int(len(sub)),
            "prior_width90_mean": float(sub["prior_width90"].mean()),
            "posterior_width90_mean": float(sub["posterior_width90"].mean()),
            "width90_ratio_mean": float(sub["width90_ratio_post_over_prior"].mean()),
            "width90_ratio_median": float(sub["width90_ratio_post_over_prior"].median()),
            "var_ratio_mean": float(sub["var_ratio_post_over_prior"].mean()),
            "var_ratio_median": float(sub["var_ratio_post_over_prior"].median()),
            "mean_shift_abs_mean": float(sub["mean_shift_abs"].mean()),
            "fraction_width90_contract": float(sub["contraction_flag_width90"].mean()),
            "fraction_var_contract": float(sub["contraction_flag_var"].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # json summary
    out = {
        "run_tag": RUN_TAG,
        "n_cases": N_CASES,
        "parameters": PARAMS,
        "main_takeaway": (
            "Posterior contraction is supported when posterior width/variance "
            "is consistently smaller than the corresponding prior width/variance "
            "across repeated synthetic benchmarks."
        ),
        "summary": summary_rows,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved:")
    print(" -", OUT_DETAIL_CSV)
    print(" -", OUT_SUMMARY_CSV)
    print(" -", OUT_JSON)


if __name__ == "__main__":
    main()