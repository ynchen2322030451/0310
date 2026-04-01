import json
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/tjzs/Documents/0310")
OUT_FILE = ROOT / "upload_bundle_results.txt"

# ============================================================
# 每个逻辑结果项允许配置多个候选路径
# 脚本会自动选择第一个存在的文件
# ============================================================

FILE_GROUPS = {
    "A_model_performance": [
        {
            "name": "paper_metrics_table",
            "candidates": [
                "experiments_phys_levels/paper_metrics_table.csv",
            ],
            "purpose": "主模型性能总表，是正文模型对比与最终模型选择的核心表。",
        },
        {
            "name": "focus_metrics_level0",
            "candidates": [
                "experiments_phys_levels/fixed_surrogate_fixed_base/paper_focus_metrics_level0.csv",
                "experiments_phys_levels/paper_focus_metrics_level0.csv",
            ],
            "purpose": "baseline 模型在主输出上的聚焦性能表。",
        },
        {
            "name": "focus_metrics_level2",
            "candidates": [
                "experiments_phys_levels/fixed_surrogate_fixed_level2/paper_focus_metrics_level2.csv",
                "experiments_phys_levels/paper_focus_metrics_level2.csv",
                "experiments_phys_levels/paper_fixed_model_compare_primary.csv",
            ],
            "purpose": "regularized 主模型在主输出上的聚焦性能表；若单独 focus 文件不存在，则回退到模型主输出对比表。",
        },
        {
            "name": "fixed_model_compare_summary",
            "candidates": [
                "experiments_phys_levels/paper_fixed_model_compare_summary.csv",
            ],
            "purpose": "fixed baseline 与 fixed level2 的总体对比表。",
        },
        {
            "name": "fixed_model_compare_per_output",
            "candidates": [
                "experiments_phys_levels/paper_fixed_model_compare_per_output.csv",
            ],
            "purpose": "fixed baseline 与 fixed level2 的逐输出对比表。",
        },
    ],

    "B_forward_uq_and_risk": [
        {
            "name": "paper_forward_uq_summary",
            "candidates": [
                "experiments_phys_levels/paper_forward_uq_summary.csv",
            ],
            "purpose": "正向不确定性传播总表，包含 stress / keff 统计、阈值失效概率和 overall CVR。",
        },
        {
            "name": "paper_safety_threshold_summary",
            "candidates": [
                "experiments_phys_levels/paper_safety_threshold_summary.csv",
            ],
            "purpose": "test-set 主阈值风险分析表，用于比较 predictive risk、mean-only risk 与真值风险。",
        },
        {
            "name": "forward_failure_prob_level0",
            "candidates": [
                "experiments_phys_levels/forward_uq_failure_prob_level0.csv",
                "experiments_phys_levels/forward_uq_failure_prob_pred_level0.csv",
            ],
            "purpose": "baseline 的 forward UQ 阈值失效概率结果。",
        },
        {
            "name": "forward_failure_prob_level2",
            "candidates": [
                "experiments_phys_levels/forward_uq_failure_prob_level2.csv",
                "experiments_phys_levels/forward_uq_failure_prob_pred_level2.csv",
            ],
            "purpose": "level2 的 forward UQ 阈值失效概率结果。",
        },
    ],

    "C_sobol_and_ci": [
        {
            "name": "paper_sobol_results_with_ci",
            "candidates": [
                "experiments_phys_levels/paper_sobol_results_with_ci.csv",
            ],
            "purpose": "Sobol 原始主结果，包含 S1 / ST 及其 CI，是结果解释的底层表。",
        },
        {
            "name": "paper_sobol_methods_ready_summary",
            "candidates": [
                "experiments_phys_levels/paper_sobol_methods_ready_summary.csv",
            ],
            "purpose": "Sobol+CI 的方法学整理表，标记稳定主导项和跨零项。",
        },
        {
            "name": "paper_sobol_results_ready_top_factors",
            "candidates": [
                "experiments_phys_levels/paper_sobol_results_ready_top_factors.csv",
            ],
            "purpose": "可直接写入结果部分的 Sobol 主导因子摘要表。",
        },
        {
            "name": "dataset_sensitivity_top3",
            "candidates": [
                "experiments_phys_levels/dataset_sensitivity_top3.csv",
            ],
            "purpose": "基于固定数据集的 Spearman top3 因子汇总，用于与 surrogate Sobol 结果对照。",
        },
        {
            "name": "dataset_sensitivity_spearman",
            "candidates": [
                "experiments_phys_levels/dataset_sensitivity_spearman.csv",
            ],
            "purpose": "固定数据集上的 Spearman 敏感性全表。",
        },
    ],

    "D_inverse_benchmark_maintext": [
        {
            "name": "inverse_benchmark_meta_reduced_maintext",
            "candidates": [
                "experiments_phys_levels/inverse_benchmark_meta_reduced_maintext.json",
                "experiments_phys_levels/_legacy_unused_20260325_161538/inverse_benchmark_meta_reduced_maintext.json",
            ],
            "purpose": "maintext 版本 reduced inverse benchmark 的配置与运行元数据。",
        },
        {
            "name": "inverse_benchmark_case_summary_reduced_maintext",
            "candidates": [
                "experiments_phys_levels/inverse_benchmark_case_summary_reduced_maintext.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/inverse_benchmark_case_summary_reduced_maintext.csv",
            ],
            "purpose": "maintext 版本 reduced inverse 的 case 级 benchmark 汇总表。",
        },
        {
            "name": "inverse_benchmark_parameter_recovery_summary_reduced_maintext",
            "candidates": [
                "experiments_phys_levels/inverse_benchmark_parameter_recovery_summary_reduced_maintext.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/inverse_benchmark_parameter_recovery_summary_reduced_maintext.csv",
            ],
            "purpose": "maintext 版本 reduced inverse 的参数恢复性总结。",
        },
        {
            "name": "inverse_benchmark_observation_fit_summary_reduced_maintext",
            "candidates": [
                "experiments_phys_levels/inverse_benchmark_observation_fit_summary_reduced_maintext.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/inverse_benchmark_observation_fit_summary_reduced_maintext.csv",
            ],
            "purpose": "maintext 版本 reduced inverse 的观测拟合总结。",
        },
    ],

    "E_inverse_comparison_and_contraction": [
        {
            "name": "paper_inverse_full_vs_reduced_summary",
            "candidates": [
                "experiments_phys_levels/paper_inverse_full_vs_reduced_summary.csv",
            ],
            "purpose": "full inverse 与 reduced inverse 的总体对比表。",
        },
        {
            "name": "paper_inverse_full_vs_reduced_parameter_table",
            "candidates": [
                "experiments_phys_levels/paper_inverse_full_vs_reduced_parameter_table.csv",
            ],
            "purpose": "full vs reduced 在参数恢复上的对比表。",
        },
        {
            "name": "paper_inverse_full_vs_reduced_observable_table",
            "candidates": [
                "experiments_phys_levels/paper_inverse_full_vs_reduced_observable_table.csv",
            ],
            "purpose": "full vs reduced 在观测拟合上的对比表。",
        },
        {
            "name": "paper_prior_posterior_contraction_summary_reduced",
            "candidates": [
                "experiments_phys_levels/paper_prior_posterior_contraction_summary_reduced.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/paper_prior_posterior_contraction_summary_reduced.csv",
            ],
            "purpose": "reduced inverse 的 prior-posterior 收缩量化结果。",
        },
    ],

    "F_speed_ood_and_supporting_results": [
        {
            "name": "paper_speed_benchmark_detailed",
            "candidates": [
                "experiments_phys_levels/paper_speed_benchmark_detailed.json",
            ],
            "purpose": "高保真 CPU 与代理 GPU 工作流实际 wall-clock 对比记录。",
        },
        {
            "name": "paper_ood_results",
            "candidates": [
                "experiments_phys_levels/paper_ood_results.csv",
            ],
            "purpose": "OOD 汇总结果，适合附录和局限性讨论。",
        },
        {
            "name": "paper_iter1_iter2_forward_compare",
            "candidates": [
                "experiments_phys_levels/paper_iter1_iter2_forward_compare.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/paper_iter1_iter2_forward_compare.csv",
            ],
            "purpose": "iter1 与 iter2 的 forward UQ 对比结果。",
        },
        {
            "name": "paper_iter1_iter2_sobol_compare",
            "candidates": [
                "experiments_phys_levels/paper_iter1_iter2_sobol_compare.csv",
                "experiments_phys_levels/_legacy_unused_20260325_161538/paper_iter1_iter2_sobol_compare.csv",
            ],
            "purpose": "iter1 与 iter2 的 Sobol 主导因子变化结果。",
        },
    ],
}

GROUP_PURPOSE = {
    "A_model_performance": "模型训练后在测试集上的核心性能结果，用于模型选择。",
    "B_forward_uq_and_risk": "主文 forward uncertainty-to-risk 分析链。",
    "C_sobol_and_ci": "主文/方法中关于敏感性与置信区间解释的核心结果。",
    "D_inverse_benchmark_maintext": "当前主文采用的 reduced maintext inverse benchmark 结果。",
    "E_inverse_comparison_and_contraction": "full/reduced 对照与 posterior contraction 结果。",
    "F_speed_ood_and_supporting_results": "速度、OOD 与 iter1/iter2 差异等补充性结果。",
}


def resolve_existing_path(candidates):
    for rel in candidates:
        path = ROOT / rel
        if path.exists():
            return rel, path
    return None, None


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

    lines.append("0310 项目结果汇总（精选版，自动候选路径解析）\n")
    lines.append("=" * 120 + "\n")
    lines.append(f"生成时间: {now}\n")
    lines.append(f"项目根目录: {ROOT}\n\n")

    # ========================================================
    # 目录索引
    # ========================================================
    lines.append("【目录索引】\n")
    for group, items in FILE_GROUPS.items():
        lines.append(f"\n- {group}\n")
        lines.append(f"  说明: {GROUP_PURPOSE.get(group, '待补充')}\n")
        for item in items:
            rel, path = resolve_existing_path(item["candidates"])
            if rel is None:
                chosen = "未找到"
            else:
                chosen = rel
            lines.append(f"    * {item['name']}\n")
            lines.append(f"      当前文件: {chosen}\n")
            lines.append(f"      用途: {item['purpose']}\n")
    lines.append("\n\n")

    # ========================================================
    # 逐组写入内容
    # ========================================================
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

    # ========================================================
    # 缺失文件汇总
    # ========================================================
    lines.append("=" * 120 + "\n")
    lines.append("[缺失文件汇总]\n")
    lines.append("=" * 120 + "\n")

    if len(missing_items) == 0:
        lines.append("所有逻辑结果项均已找到对应文件。\n")
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
        print(f"[WARN] 有 {len(missing_items)} 个逻辑结果项未找到，详情见文件末尾的“缺失文件汇总”。")
    else:
        print("[OK] 所有逻辑结果项都已成功解析。")


if __name__ == "__main__":
    main()