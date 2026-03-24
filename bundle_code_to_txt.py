import os
from pathlib import Path

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_code.txt"

FILES = [
    "paper_experiment_config.py",
    "run_phys_levels_main.py",
    "run_forward_uq_analysis.py",
    "run_sobol_analysis.py",
    "run_sobol_ci_methods_summary.py",
    "run_calibration_benchmark.py",
    "run_inverse_diagnostics.py",
    "run_inverse_full_vs_reduced_compare.py",
    "run_prior_posterior_contraction_summary.py",
    "run_export_2d_feasible_region.py",
    "plot_2d_inverse_feasible_region_final.py",
    "run_safety_threshold_analysis.py",
    "run_speedup_benchmark.py",
    "run_practical_speed_benchmark.py",
    "plot_forward_uq_and_sobol_figures.py",
    "plot_inverse_figures.py",
]

FILE_PURPOSE = {
    "paper_experiment_config.py": "统一管理论文实验配置，包括输入/输出、主阈值、主输出、主文模型层级等。",
    "run_phys_levels_main.py": "主训练与主评估流程，生成 baseline / regularized 模型结果。",
    "run_forward_uq_analysis.py": "正向不确定性传播、输出分布、CVR 与失效概率分析主脚本。",
    "run_sobol_analysis.py": "Sobol 全局分析主脚本。",
    "run_sobol_ci_methods_summary.py": "将 Sobol+CI 结果整理成论文 Methods/Results 可直接引用的 summary。",
    "run_calibration_benchmark.py": "重复 synthetic benchmark 的 inverse UQ 主脚本。",
    "run_inverse_diagnostics.py": "inverse benchmark 结果汇总、诊断与图输入整理。",
    "run_inverse_full_vs_reduced_compare.py": "full inverse 与 reduced inverse 的结果对照整理脚本。",
    "run_prior_posterior_contraction_summary.py": "prior–posterior contraction 定量汇总脚本。",
    "run_export_2d_feasible_region.py": "导出二维 dominant-parameter plane 的 prior/posterior/feasible 数据。",
    "plot_2d_inverse_feasible_region_final.py": "二维 feasible-region 与 posterior contraction 的最终版画图脚本。",
    "run_safety_threshold_analysis.py": "131 MPa 及扩展阈值风险统计分析。",
    "run_speedup_benchmark.py": "代理模型速度测试主脚本。",
    "run_practical_speed_benchmark.py": "记录高保真 CPU 与代理 GPU 工作流实际 wall-clock 对比。",
    "plot_forward_uq_and_sobol_figures.py": "forward UQ、threshold risk、Sobol、CVR 等主文图脚本。",
    "plot_inverse_figures.py": "inverse benchmark 主图脚本。",
}

def main():
    lines = []
    lines.append("0310 项目代码汇总\n")
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
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines.append(f.read())
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