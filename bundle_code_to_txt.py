import json
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_code.txt"

# ============================================================
# 每个逻辑代码项允许配置多个候选路径
# 脚本会自动选择第一个存在的文件
# ============================================================

FILE_GROUPS = {
    "A_core_config_and_training": [
        {
            "name": "paper_experiment_config",
            "candidates": [
                "paper_experiment_config.py",
            ],
            "purpose": "统一管理论文实验配置，包括输入/输出、主输出、阈值、seed、路径等。",
        },
        {
            "name": "run_phys_levels_main",
            "candidates": [
                "run_phys_levels_main.py",
            ],
            "purpose": "核心模型定义、训练函数、基础评估函数所在主脚本，是下游所有实验的底层依赖。",
        },
        {
            "name": "run_fixed_surrogate_train_base",
            "candidates": [
                "run_fixed_surrogate_train_base.py",
            ],
            "purpose": "固定数据划分上的 baseline/base 模型训练脚本。",
        },
        {
            "name": "run_prepare_fixed_surrogate",
            "candidates": [
                "run_prepare_fixed_surrogate.py",
            ],
            "purpose": "固定 split 的准备脚本，用于冻结 train/val/test 划分。",
        },
    ],

    "B_forward_uq_and_risk": [
        {
            "name": "run_forward_uq_analysis",
            "candidates": [
                "run_forward_uq_analysis.py",
                "new_run_forward_uq_analysis.py",
            ],
            "purpose": "正向不确定性传播主脚本，导出输出分布、失效概率、联合样本和 CVR。",
        },
        {
            "name": "run_safety_threshold_analysis",
            "candidates": [
                "run_safety_threshold_analysis.py",
            ],
            "purpose": "测试集阈值风险分析脚本，用于比较 truth / mean / predictive risk。",
        },
        {
            "name": "plot_forward_uq_and_sobol_figures",
            "candidates": [
                "plot_forward_uq_and_sobol_figures.py",
            ],
            "purpose": "forward UQ、threshold risk、Sobol、CVR 等主文图绘图脚本。",
        },
    ],

    "C_sensitivity_and_comparison": [
        {
            "name": "run_sobol_analysis",
            "candidates": [
                "run_sobol_analysis.py",
            ],
            "purpose": "Sobol 全局敏感性分析主脚本。",
        },
        {
            "name": "run_sobol_ci_methods_summary",
            "candidates": [
                "run_sobol_ci_methods_summary.py",
            ],
            "purpose": "将 Sobol+CI 结果整理成 methods/results 可直接使用的 summary。",
        },
        {
            "name": "run_dataset_sensitivity_analysis",
            "candidates": [
                "run_dataset_sensitivity_analysis.py",
            ],
            "purpose": "直接基于固定数据集做 Spearman 敏感性分析，用于与 surrogate Sobol 对照。",
        },
        {
            "name": "run_compare_fixed_models",
            "candidates": [
                "run_compare_fixed_models.py",
            ],
            "purpose": "固定 baseline 与 fixed level2 的测试集性能对比脚本。",
        },
        {
            "name": "run_iter1_iter2_forward_compare",
            "candidates": [
                "run_iter1_iter2_forward_compare.py",
            ],
            "purpose": "iter1 与 iter2 的 forward UQ 结果比较脚本。",
        },
        {
            "name": "run_iter1_iter2_sobol_compare",
            "candidates": [
                "run_iter1_iter2_sobol_compare.py",
            ],
            "purpose": "iter1 与 iter2 的 Sobol 主导因子比较脚本。",
        },
    ],

    "D_inverse_uq_main": [
        {
            "name": "run_calibration_benchmark",
            "candidates": [
                "run_calibration_benchmark.py",
            ],
            "purpose": "reduced/full inverse benchmark 的主脚本。",
        },
        {
            "name": "run_inverse_benchmark_fixed_surrogate",
            "candidates": [
                "run_inverse_benchmark_fixed_surrogate.py",
            ],
            "purpose": "固定 surrogate 条件下的 inverse benchmark 版本脚本。",
        },
        {
            "name": "run_inverse_diagnostics",
            "candidates": [
                "run_inverse_diagnostics.py",
            ],
            "purpose": "inverse benchmark 结果汇总、诊断与图输入整理脚本。",
        },
        {
            "name": "run_inverse_full_vs_reduced_compare",
            "candidates": [
                "run_inverse_full_vs_reduced_compare.py",
            ],
            "purpose": "full inverse 与 reduced inverse 的结果对比脚本。",
        },
        {
            "name": "run_prior_posterior_contraction_summary",
            "candidates": [
                "run_prior_posterior_contraction_summary.py",
            ],
            "purpose": "prior-posterior contraction 定量汇总脚本。",
        },
        {
            "name": "run_export_2d_feasible_region",
            "candidates": [
                "run_export_2d_feasible_region.py",
            ],
            "purpose": "导出二维 dominant-parameter plane 的 prior/posterior/feasible region 数据。",
        },
        {
            "name": "plot_2d_inverse_feasible_region_final",
            "candidates": [
                "plot_2d_inverse_feasible_region_final.py",
            ],
            "purpose": "二维 feasible-region 与 posterior contraction 最终版绘图脚本。",
        },
        {
            "name": "plot_inverse_figures",
            "candidates": [
                "plot_inverse_figures.py",
            ],
            "purpose": "inverse benchmark 主图绘图脚本。",
        },
    ],

    "E_speed_ood_and_supporting": [
        {
            "name": "run_speedup_benchmark",
            "candidates": [
                "run_speedup_benchmark.py",
            ],
            "purpose": "代理模型与高保真程序的速度对比脚本。",
        },
        {
            "name": "run_practical_speed_benchmark",
            "candidates": [
                "run_practical_speed_benchmark.py",
            ],
            "purpose": "记录实际 wall-clock 的实用速度基准脚本。",
        },
        {
            "name": "run_ood_evaluation",
            "candidates": [
                "run_ood_evaluation.py",
            ],
            "purpose": "OOD 泛化评估脚本。",
        },
    ],

    "F_project_maintenance": [
        {
            "name": "bundle_code_to_txt",
            "candidates": [
                "bundle_code_to_txt.py",
            ],
            "purpose": "代码打包脚本本体。",
        },
        {
            "name": "bundle_results_to_txt",
            "candidates": [
                "bundle_results_to_txt.py",
            ],
            "purpose": "结果打包脚本本体。",
        },
        {
            "name": "cleanup_legacy_files",
            "candidates": [
                "cleanup_legacy_files.py",
            ],
            "purpose": "历史遗留文件整理脚本。",
        },
    ],
}

GROUP_PURPOSE = {
    "A_core_config_and_training": "模型定义、固定数据划分与训练主链。",
    "B_forward_uq_and_risk": "正向不确定性传播与风险分析主链。",
    "C_sensitivity_and_comparison": "Sobol、数据集敏感性与模型对比分析主链。",
    "D_inverse_uq_main": "inverse UQ、benchmark、收缩与二维可行域分析主链。",
    "E_speed_ood_and_supporting": "速度、OOD 等补充性实验代码。",
    "F_project_maintenance": "项目维护和打包相关工具代码。",
}


def resolve_existing_path(candidates):
    for rel in candidates:
        path = ROOT / rel
        if path.exists():
            return rel, path
    return None, None


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def file_size_kb(path: Path) -> str:
    if path is None or (not path.exists()):
        return "N/A"
    return f"{path.stat().st_size / 1024:.1f} KB"


def modified_time(path: Path) -> str:
    if path is None or (not path.exists()):
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def main():
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing_items = []

    lines.append("0310 项目代码汇总（精选版，自动候选路径解析）\n")
    lines.append("=" * 120 + "\n")
    lines.append(f"生成时间: {now}\n")
    lines.append(f"项目根目录: {ROOT}\n\n")

    lines.append("【目录索引】\n")
    for group, items in FILE_GROUPS.items():
        lines.append(f"\n- {group}\n")
        lines.append(f"  说明: {GROUP_PURPOSE.get(group, '待补充')}\n")
        for item in items:
            rel, path = resolve_existing_path(item["candidates"])
            chosen = rel if rel is not None else "未找到"
            lines.append(f"    * {item['name']}\n")
            lines.append(f"      当前文件: {chosen}\n")
            lines.append(f"      用途: {item['purpose']}\n")
    lines.append("\n\n")

    for group, items in FILE_GROUPS.items():
        lines.append("=" * 120 + "\n")
        lines.append(f"[分组] {group}\n")
        lines.append(f"[分组说明] {GROUP_PURPOSE.get(group, '待补充')}\n")
        lines.append("=" * 120 + "\n\n")

        for item in items:
            rel, path = resolve_existing_path(item["candidates"])
            exists = path is not None and path.exists()

            lines.append("#" * 120 + "\n")
            lines.append(f"逻辑名称: {item['name']}\n")
            lines.append(f"类别: {group}\n")
            lines.append(f"用途: {item['purpose']}\n")
            lines.append(f"实际文件: {rel if rel is not None else '未找到'}\n")
            lines.append(f"存在: {'是' if exists else '否'}\n")
            lines.append(f"大小: {file_size_kb(path)}\n")
            lines.append(f"修改时间: {modified_time(path)}\n")
            lines.append("候选路径:\n")
            for c in item["candidates"]:
                lines.append(f"  - {c}\n")
            lines.append("-" * 120 + "\n")

            if exists:
                try:
                    lines.append(read_text_file(path))
                except Exception as e:
                    lines.append(f"[读取失败] {e}\n")
            else:
                missing_items.append({
                    "group": group,
                    "name": item["name"],
                    "candidates": item["candidates"],
                })
                lines.append("[文件不存在]\n")

            lines.append("\n\n")

    lines.append("=" * 120 + "\n")
    lines.append("[缺失代码项汇总]\n")
    lines.append("=" * 120 + "\n")

    if len(missing_items) == 0:
        lines.append("所有逻辑代码项均已找到对应文件。\n")
    else:
        for miss in missing_items:
            lines.append(f"- 分组: {miss['group']}\n")
            lines.append(f"  逻辑名称: {miss['name']}\n")
            lines.append("  候选路径:\n")
            for c in miss["candidates"]:
                lines.append(f"    * {c}\n")
            lines.append("\n")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[DONE] 已生成: {OUT_FILE}")
    if len(missing_items) > 0:
        print(f"[WARN] 有 {len(missing_items)} 个逻辑代码项未找到，详情见文件末尾的“缺失代码项汇总”。")
    else:
        print("[OK] 所有逻辑代码项都已成功解析。")


if __name__ == "__main__":
    main()