import json
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_results.txt"

FILE_GROUPS = {
    "A_model_performance": [
        "experiments_phys_levels/paper_metrics_table.csv",
        "experiments_phys_levels/paper_focus_metrics_level0.csv",
        "experiments_phys_levels/paper_focus_metrics_level2.csv",
    ],
    "B_forward_uq_and_risk": [
        "experiments_phys_levels/paper_forward_uq_summary.csv",
        "experiments_phys_levels/paper_safety_threshold_summary.csv",
    ],
    "C_sobol_and_ci": [
        "experiments_phys_levels/paper_sobol_results_with_ci.csv",
        "experiments_phys_levels/paper_sobol_methods_ready_summary.csv",
        "experiments_phys_levels/paper_sobol_results_ready_top_factors.csv",
    ],
    "D_inverse_benchmark_maintext": [
        "experiments_phys_levels/inverse_benchmark_meta_reduced_maintext.json",
        "experiments_phys_levels/inverse_benchmark_case_summary_reduced_maintext.csv",
        "experiments_phys_levels/inverse_benchmark_parameter_recovery_summary_reduced_maintext.csv",
        "experiments_phys_levels/inverse_benchmark_observation_fit_summary_reduced_maintext.csv",
    ],
    "E_inverse_comparison_and_contraction": [
        "experiments_phys_levels/paper_inverse_full_vs_reduced_summary.csv",
        "experiments_phys_levels/paper_inverse_full_vs_reduced_parameter_table.csv",
        "experiments_phys_levels/paper_inverse_full_vs_reduced_observable_table.csv",
        "experiments_phys_levels/paper_prior_posterior_contraction_summary_reduced.csv",
    ],
    "F_speed_ood_and_supporting_results": [
        "experiments_phys_levels/paper_speed_benchmark_detailed.json",
        "experiments_phys_levels/paper_ood_results.csv",
        "experiments_phys_levels/paper_iter1_iter2_forward_compare.csv",
        "experiments_phys_levels/paper_iter1_iter2_sobol_compare.csv",
    ],
}

FILE_PURPOSE = {
    "experiments_phys_levels/paper_metrics_table.csv": "主模型性能总表，是正文模型对比与最终模型选择的核心表。",
    "experiments_phys_levels/paper_focus_metrics_level0.csv": "baseline 模型在主输出上的聚焦性能表。",
    "experiments_phys_levels/paper_focus_metrics_level2.csv": "regularized 主模型在主输出上的聚焦性能表。",
    "experiments_phys_levels/paper_forward_uq_summary.csv": "正向不确定性传播总表，包含 stress / keff 统计、阈值失效概率和 overall CVR。",
    "experiments_phys_levels/paper_safety_threshold_summary.csv": "test-set 主阈值风险分析表，用于比较 predictive risk、mean-only risk 与真值风险。",
    "experiments_phys_levels/paper_sobol_results_with_ci.csv": "Sobol 原始主结果，包含 S1 / ST 及其 CI，是结果解释的底层表。",
    "experiments_phys_levels/paper_sobol_methods_ready_summary.csv": "Sobol+CI 的方法学整理表，标记稳定主导项和跨零项。",
    "experiments_phys_levels/paper_sobol_results_ready_top_factors.csv": "可直接写入结果部分的 Sobol 主导因子摘要表。",
    "experiments_phys_levels/inverse_benchmark_meta_reduced_maintext.json": "maintext 版本 reduced inverse benchmark 的配置与运行元数据。",
    "experiments_phys_levels/inverse_benchmark_case_summary_reduced_maintext.csv": "maintext 版本 reduced inverse 的 case 级 benchmark 汇总表。",
    "experiments_phys_levels/inverse_benchmark_parameter_recovery_summary_reduced_maintext.csv": "maintext 版本 reduced inverse 的参数恢复性总结。",
    "experiments_phys_levels/inverse_benchmark_observation_fit_summary_reduced_maintext.csv": "maintext 版本 reduced inverse 的观测拟合总结。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_summary.csv": "full inverse 与 reduced inverse 的总体对比表。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_parameter_table.csv": "full vs reduced 在参数恢复上的对比表。",
    "experiments_phys_levels/paper_inverse_full_vs_reduced_observable_table.csv": "full vs reduced 在观测拟合上的对比表。",
    "experiments_phys_levels/paper_prior_posterior_contraction_summary_reduced.csv": "reduced inverse 的 prior-posterior 收缩量化结果。",
    "experiments_phys_levels/paper_speed_benchmark_detailed.json": "高保真 CPU 与代理 GPU 工作流实际 wall-clock 对比记录。",
    "experiments_phys_levels/paper_ood_results.csv": "OOD 汇总结果，适合附录和局限性讨论。",
    "experiments_phys_levels/paper_iter1_iter2_forward_compare.csv": "iter1 与 iter2 的 forward UQ 对比结果。",
    "experiments_phys_levels/paper_iter1_iter2_sobol_compare.csv": "iter1 与 iter2 的 Sobol 主导因子变化结果。",
}

GROUP_PURPOSE = {
    "A_model_performance": "模型训练后在测试集上的核心性能结果，用于模型选择。",
    "B_forward_uq_and_risk": "主文 forward uncertainty-to-risk 分析链。",
    "C_sobol_and_ci": "主文/方法中关于敏感性与置信区间解释的核心结果。",
    "D_inverse_benchmark_maintext": "当前主文采用的 reduced maintext inverse benchmark 结果。",
    "E_inverse_comparison_and_contraction": "full/reduced 对照与 posterior contraction 结果。",
    "F_speed_ood_and_supporting_results": "速度、OOD 与 iter1/iter2 差异等补充性结果。",
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


def file_size_kb(path: Path) -> str:
    if not path.exists():
        return "N/A"
    return f"{path.stat().st_size / 1024:.1f} KB"


def main():
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("0310 项目结果汇总（精选版）\n")
    lines.append("=" * 120 + "\n")
    lines.append(f"生成时间: {now}\n")
    lines.append(f"项目根目录: {ROOT}\n\n")

    lines.append("【目录索引】\n")
    for group, files in FILE_GROUPS.items():
        lines.append(f"\n- {group}\n")
        lines.append(f"  说明: {GROUP_PURPOSE.get(group, '待补充')}\n")
        for rel in files:
            lines.append(f"    * {rel} -- {FILE_PURPOSE.get(rel, '待补充说明')}\n")
    lines.append("\n\n")

    for group, files in FILE_GROUPS.items():
        lines.append("=" * 120 + "\n")
        lines.append(f"[分组] {group}\n")
        lines.append(f"[分组说明] {GROUP_PURPOSE.get(group, '待补充')}\n")
        lines.append("=" * 120 + "\n\n")

        for rel in files:
            path = ROOT / rel
            exists = path.exists()

            lines.append("#" * 120 + "\n")
            lines.append(f"文件: {rel}\n")
            lines.append(f"类别: {group}\n")
            lines.append(f"用途: {FILE_PURPOSE.get(rel, '待补充说明')}\n")
            lines.append(f"存在: {'是' if exists else '否'}\n")
            lines.append(f"大小: {file_size_kb(path)}\n")
            lines.append("-" * 120 + "\n")

            if exists:
                try:
                    lines.append(read_text_file(path))
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