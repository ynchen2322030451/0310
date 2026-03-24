import os
import json
from pathlib import Path

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_results.txt"

FILES = [
    "experiments_phys_levels/paper_metrics_table.csv",
    "experiments_phys_levels/paper_forward_uq_summary.csv",
    "experiments_phys_levels/paper_safety_threshold_summary.csv",
    "experiments_phys_levels/paper_sobol_results_with_ci.csv",
    "experiments_phys_levels/paper_sobol_methods_ready_summary.csv",
    "experiments_phys_levels/paper_sobol_results_ready_top_factors.csv",
    "experiments_phys_levels/calibration_benchmark_case_summary_reduced.csv",
    "experiments_phys_levels/calibration_benchmark_parameter_recovery_summary_reduced.csv",
    "experiments_phys_levels/calibration_benchmark_observation_fit_summary_reduced.csv",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_summary.csv",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_parameter_table.csv",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_observable_table.csv",
    "experiments_phys_levels/paper_prior_posterior_contraction_summary_reduced.csv",
    "experiments_phys_levels/paper_speed_benchmark_detailed.json",
    "experiments_phys_levels/paper_ood_results.csv",
]

FILE_PURPOSE = {
    "experiments_phys_levels/paper_metrics_table.csv": "主模型性能总表，正文主表候选。",
    "experiments_phys_levels/paper_forward_uq_summary.csv": "正向不确定性传播总表，包含 stress/keff 统计、失效概率与 CVR。",
    "experiments_phys_levels/paper_safety_threshold_summary.csv": "主阈值分析汇总表，用于 test-set threshold analysis。",
    "experiments_phys_levels/paper_sobol_results_with_ci.csv": "Sobol 全局分析原始主结果，带 CI。",
    "experiments_phys_levels/paper_sobol_methods_ready_summary.csv": "Sobol+CI 的 Methods-ready 汇总，标出跨 0 项与稳定主导项。",
    "experiments_phys_levels/paper_sobol_results_ready_top_factors.csv": "Results-ready Sobol 主导因子汇总表。",
    "experiments_phys_levels/calibration_benchmark_case_summary_reduced.csv": "reduced inverse 的 case 级 benchmark 汇总。",
    "experiments_phys_levels/calibration_benchmark_parameter_recovery_summary_reduced.csv": "reduced inverse 参数恢复性总结。",
    "experiments_phys_levels/calibration_benchmark_observation_fit_summary_reduced.csv": "reduced inverse 观测拟合总结。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_summary.csv": "full inverse 与 reduced inverse 的总对比表。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_parameter_table.csv": "full vs reduced 的参数层面对比表。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_observable_table.csv": "full vs reduced 的观测量层面对比表。",
    "experiments_phys_levels/paper_prior_posterior_contraction_summary_reduced.csv": "reduced inverse 下 prior–posterior 收缩程度总结。",
    "experiments_phys_levels/paper_speed_benchmark_detailed.json": "高保真 CPU 与代理 GPU 工作流的实际 wall-clock 对比记录。",
    "experiments_phys_levels/paper_ood_results.csv": "OOD 汇总结果，适合附录/讨论。",
}

def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def main():
    lines = []
    lines.append("0310 项目结果汇总\n")
    lines.append("=" * 100 + "\n\n")

    for rel in FILES:
        path = ROOT / rel
        lines.append("#" * 100 + "\n")
        lines.append(f"文件: {rel}\n")
        lines.append(f"用途: {FILE_PURPOSE.get(rel, '待补充说明')}\n")
        lines.append(f"存在: {'是' if path.exists() else '否'}\n")
        lines.append("-" * 100 + "\n")

        if path.exists():
            try:
                content = read_text_file(path)
                lines.append(content)
            except Exception as e:
                lines.append(f"[读取失败] {e}\n")
        else:
            lines.append("[文件不存在]\n")

        lines.append("\n\n")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[DONE] 已生成: {OUT_FILE}")

if __name__ == "__main__":
    main()