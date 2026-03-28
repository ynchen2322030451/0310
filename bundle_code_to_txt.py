import os
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_code.txt"

FILE_GROUPS = {
    "A_core_config_and_training": [
        "paper_experiment_config.py",
        "run_phys_levels_main.py",
        "run_prepare_fixed_surrogate.py",
        "run_fixed_surrogate_train_base.py",
    ],
    "B_forward_risk_sensitivity": [
        "run_forward_uq_analysis.py",
        "run_safety_threshold_analysis.py",
        "run_sobol_analysis.py",
        "run_sobol_ci_methods_summary.py",
    ],
    "C_inverse_pipeline": [
        "run_inverse_benchmark_fixed_surrogate.py",
        "run_inverse_diagnostics.py",
        "run_inverse_full_vs_reduced_compare.py",
        "run_prior_posterior_contraction_summary.py",
    ],
    "D_feasible_region_and_plotting": [
        "run_export_2d_feasible_region.py",
        "plot_2d_inverse_feasible_region_final.py",
        "plot_forward_uq_and_sobol_figures.py",
        "plot_inverse_figures.py",
    ],
    "E_speed_ood_and_supporting_analysis": [
        "run_speedup_benchmark.py",
        "run_practical_speed_benchmark.py",
        "run_ood_evaluation.py",
        "run_iter1_iter2_forward_compare.py",
        "run_iter1_iter2_sobol_compare.py",
    ],
}

FILE_PURPOSE = {
    "paper_experiment_config.py": "统一管理论文实验配置，包括输入/输出、主阈值、主输出、层级定义、路径与随机种子。",
    "run_phys_levels_main.py": "主训练与主评估脚本，负责 baseline / regularized surrogate 的训练、测试和指标输出。",
    "run_prepare_fixed_surrogate.py": "生成固定 train/val/test 划分并保存索引与数据文件，保证后续所有实验共享同一数据拆分。",
    "run_fixed_surrogate_train_base.py": "在固定数据划分上训练 baseline surrogate，作为与正则化模型对照的统一基线。",
    "run_forward_uq_analysis.py": "基于固定 surrogate 进行正向不确定性传播，输出分布统计、CVR 和失效概率。",
    "run_safety_threshold_analysis.py": "针对主应力阈值进行 test-set 风险分析，比较 predictive probability、mean-only 判断与真实标签。",
    "run_sobol_analysis.py": "执行 Sobol 全局敏感性分析，输出 S1/ST 指标及其不确定性统计。",
    "run_sobol_ci_methods_summary.py": "将 Sobol 原始结果进一步整理为带 CI 的 methods-ready 与 results-ready 汇总表。",
    "run_inverse_benchmark_fixed_surrogate.py": "在固定 surrogate 条件下执行 inverse benchmark，是当前主文 inverse UQ 的核心脚本。",
    "run_inverse_diagnostics.py": "汇总 inverse benchmark 结果，生成参数恢复、观测拟合、接受率和可行域等诊断结果。",
    "run_inverse_full_vs_reduced_compare.py": "比较 full inverse 与 reduced inverse 的观测层面和参数层面差异。",
    "run_prior_posterior_contraction_summary.py": "量化 prior 到 posterior 的宽度收缩、方差收缩和均值偏移。",
    "run_export_2d_feasible_region.py": "导出二维主导参数平面上的 prior / posterior / feasible region 数据。",
    "plot_2d_inverse_feasible_region_final.py": "绘制最终版二维 feasible-region 与 posterior contraction 图。",
    "plot_forward_uq_and_sobol_figures.py": "绘制正向 UQ、risk curve、Sobol、CVR 等主文图。",
    "plot_inverse_figures.py": "绘制 inverse benchmark 主图，包括参数恢复、观测拟合和代表性 case 图。",
    "run_speedup_benchmark.py": "执行代理模型与高保真流程的速度对比基准测试。",
    "run_practical_speed_benchmark.py": "记录实际运行环境下的 HF CPU 与 surrogate GPU 的 wall-clock 对比。",
    "run_ood_evaluation.py": "对选定参数进行 OOD 外推测试，作为附录/讨论支撑。",
    "run_iter1_iter2_forward_compare.py": "比较 iter1 与 iter2 输出在 forward UQ 下的统计差异。",
    "run_iter1_iter2_sobol_compare.py": "比较 iter1 与 iter2 输出的 Sobol 主导因子变化。",
}

GROUP_PURPOSE = {
    "A_core_config_and_training": "核心配置与统一训练入口，支撑所有下游分析。",
    "B_forward_risk_sensitivity": "主文的正向不确定性传播、风险量化与全局敏感性主链。",
    "C_inverse_pipeline": "主文 inverse benchmark、收缩分析与 full/reduced 对照主链。",
    "D_feasible_region_and_plotting": "二维可行域导出与主文图绘制脚本。",
    "E_speed_ood_and_supporting_analysis": "速度、OOD 与 iter1/iter2 对比等补充分析。",
}


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def file_size_kb(path: Path) -> str:
    if not path.exists():
        return "N/A"
    return f"{path.stat().st_size / 1024:.1f} KB"


def main():
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("0310 项目代码汇总（精选版）\n")
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
                    lines.append(read_text(path))
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