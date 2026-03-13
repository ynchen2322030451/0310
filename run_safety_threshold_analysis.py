# run_safety_threshold_analysis.py
# ============================================================
# Safety-threshold analysis for paper
# ============================================================

import os
import json
import math
import numpy as np
import pandas as pd

from paper_experiment_config import (
    OUT_DIR,
    PAPER_LEVELS,
    PRIMARY_STRESS_THRESHOLD,
    THRESHOLD_SWEEP,
    PRIMARY_STRESS_OUTPUT,
)

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def compute_failure_metrics(y_true, mu, sigma, output_names, threshold):
    stress_idx = output_names.index("iteration2_max_global_stress")

    y_s = y_true[:, stress_idx]
    mu_s = mu[:, stress_idx]
    sig_s = np.maximum(sigma[:, stress_idx], 1e-12)

    p_fail_each = []
    for m, s in zip(mu_s, sig_s):
        z = (threshold - m) / s
        p_fail_each.append(1.0 - normal_cdf(z))
    p_fail_predictive = float(np.mean(p_fail_each))

    p_fail_mean_only = float(np.mean(mu_s > threshold))
    p_fail_truth = float(np.mean(y_s > threshold))

    return {
        "threshold_MPa": float(threshold),
        "p_fail_predictive": p_fail_predictive,
        "p_fail_mean_only": p_fail_mean_only,
        "p_fail_truth": p_fail_truth,
        "stress_true_mean": float(np.mean(y_s)),
        "stress_true_std": float(np.std(y_s)),
        "stress_pred_mean": float(np.mean(mu_s)),
        "stress_pred_std": float(np.std(mu_s)),
        "stress_unc_mean": float(np.mean(sig_s)),
        "stress_output": PRIMARY_STRESS_OUTPUT,
    }

def main():
    summary_rows = []
    sweep_rows_all = []

    for level in PAPER_LEVELS:
        pred_path = os.path.join(OUT_DIR, f"test_predictions_level{level}.json")
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing file: {pred_path}")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        y_true = np.asarray(obj["y_true"], dtype=float)
        mu = np.asarray(obj["mu"], dtype=float)
        sigma = np.asarray(obj["sigma"], dtype=float)
        output_names = obj["output_names"]

        main_res = compute_failure_metrics(
            y_true, mu, sigma, output_names, PRIMARY_STRESS_THRESHOLD
        )

        sweep_res = []
        for thr in THRESHOLD_SWEEP:
            one = compute_failure_metrics(y_true, mu, sigma, output_names, thr)
            sweep_res.append(one)

            sweep_rows_all.append({
                "level": level,
                **one
            })

        out_json = {
            "level": level,
            "main_threshold_result": main_res,
            "threshold_sweep_results": sweep_res,
        }

        with open(os.path.join(OUT_DIR, f"safety_threshold_analysis_level{level}.json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False)

        pd.DataFrame(sweep_res).to_csv(
            os.path.join(OUT_DIR, f"safety_threshold_sweep_level{level}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        summary_rows.append({
            "level": level,
            "main_threshold_MPa": main_res["threshold_MPa"],
            "p_fail_predictive_main": main_res["p_fail_predictive"],
            "p_fail_mean_only_main": main_res["p_fail_mean_only"],
            "p_fail_truth_main": main_res["p_fail_truth"],
            "stress_pred_mean": main_res["stress_pred_mean"],
            "stress_unc_mean": main_res["stress_unc_mean"],
        })

        print(f"[OK] Saved safety threshold analysis for level {level}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUT_DIR, "paper_safety_threshold_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if sweep_rows_all:
        pd.DataFrame(sweep_rows_all).to_csv(
            os.path.join(OUT_DIR, "paper_safety_threshold_sweep_all_levels.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("[DONE] Safety-threshold analysis completed.")

if __name__ == "__main__":
    main()