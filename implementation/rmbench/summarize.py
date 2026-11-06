"""Summarize completed RMBench checkpoint rollouts and source evidence."""

import argparse
import csv
import json
from pathlib import Path


MODELS = ("M0", "M3")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def checkpoint_result(evaluation, model, epoch, seeds, count):
    directory = evaluation / model / f"epoch_{epoch:03d}"
    summary = read_json(directory / "summary.json")
    if summary is None or not summary["complete"]:
        return None
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()
            if line.strip()]
    if (summary["model"] != model or summary["epoch"] != epoch
            or summary["episodes"] != count or len(rows) != count
            or [row["seed"] for row in rows] != seeds):
        raise ValueError(f"Completed result does not match the fixed reset protocol: {directory}")
    successes = sum(row["success"] for row in rows)
    if summary["successes"] != successes or summary["success_rate"] != successes / count:
        raise ValueError(f"Checkpoint success count disagrees with its episodes: {directory}")
    return {"epoch": epoch, "path": directory, "summary": summary, "rows": rows,
            "successes": successes, "rate": successes / count}


def score_cells(result, count):
    if result is None:
        return "待完成", "待完成"
    return f"{result['successes']}/{count}", f"{result['rate']:.1%}"


def best_result(results, epochs):
    if not all(epoch in results for epoch in epochs):
        return None
    return max(results.values(), key=lambda result: (result["successes"], -result["epoch"]))


def training_progress(root, model):
    path = root / "logs" / f"{model}_epochs.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def number(value):
    return f"{value:,}" if value is not None else "待记录"


def parameters(model_config, results):
    if model_config and "trainable_parameters" in model_config:
        return model_config["trainable_parameters"], model_config["inference_parameters"]
    if results:
        summary = next(iter(results.values()))["summary"]
        return summary.get("training_parameters"), summary.get("inference_parameters")
    return None, None


def report_timing(result):
    summary = result["summary"]
    replans = summary.get("replans", 0)
    if not replans:
        return None
    prediction = 1000 * summary["predict_seconds"] / replans
    observation = 1000 * summary["observe_seconds"] / replans
    return f"{prediction:.2f}", f"{observation:.2f}", f"{prediction + observation:.2f}"


def representative(results, best, success, evaluation):
    ordered = ([best] if best is not None else []) + [
        result for result in results.values() if result is not best
    ]
    for result in ordered:
        for row in result["rows"]:
            if bool(row["success"]) != success or not row.get("video"):
                continue
            path = result["path"] / row["video"]
            if not path.is_file():
                continue
            final = row.get("final", {})
            stage = final.get("stage", "未记录")
            error = final.get("xy_return_error_m")
            error_text = "" if error is None else f"，末态方块平面回位误差 {100 * error:.2f} cm"
            return (
                f"第 {result['epoch']} 轮，环境种子 {row['seed']}，执行 {row['steps']} 步，"
                f"最终阶段 {stage}{error_text}。"
                f"[视频]({path.relative_to(evaluation).as_posix()})"
            )
    return "待记录"


def report(root):
    config_dir = root / "configs"
    protocol = read_json(config_dir / "protocol.json")
    evaluation = root / "evaluation"
    epochs, count = protocol["eval_epochs"], protocol["eval_episodes"]
    seed_manifest = read_json(evaluation / "reset_seeds.json")
    seeds = []
    if seed_manifest and seed_manifest["complete"]:
        seeds = seed_manifest["seeds"]
        if (len(seeds) != count or len(set(seeds)) != count
                or seed_manifest["task"] != protocol["task"]
                or seed_manifest["task_config"] != protocol["task_config"]):
            raise ValueError("The completed reset manifest does not match the protocol")

    results = {model: {} for model in MODELS}
    for model in MODELS:
        for epoch in epochs:
            result = checkpoint_result(evaluation, model, epoch, seeds, count)
            if result is not None:
                results[model][epoch] = result
    best = {model: best_result(results[model], epochs) for model in MODELS}

    lines = [
        "# RMBench 放回方块单种子闭环验证",
        "",
        f"任务为 {protocol['task']}，配置为 {protocol['task_config']}，训练种子为 {protocol['seed']}。"
        f"M0 和 M3 各计划训练 {protocol['epochs']} 轮，在同一组 {count} 个官方环境初始条件上评测五个检查点。"
        "两组成绩取各自五个检查点中的最高成功率，同分取最早检查点。",
        "",
    ]
    if seeds:
        lines.append(f"已固定 {len(seeds)} 个有效测试初始条件。[初始条件清单](reset_seeds.json)")
    else:
        lines.append("固定测试初始条件待生成。")

    lines += [
        "",
        "## 闭环成功率",
        "",
        f"| 轮次 | M0 成功数/{count} | M0 成功率 | M3 成功数/{count} | M3 成功率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for epoch in epochs:
        cells = [cell for model in MODELS for cell in score_cells(results[model].get(epoch), count)]
        lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
    cells = [cell for model in MODELS for cell in score_cells(best[model], count)]
    lines.append("| 最佳 | " + " | ".join(cells) + " |")
    lines.append("")
    for model in MODELS:
        if best[model] is not None:
            epoch = best[model]["epoch"]
            path = f"../checkpoints/{model}/epoch_{epoch:03d}.pt"
            lines.append(f"{model} 的最佳检查点为第 {epoch} 轮。[检查点]({path})")
    if all(best.values()):
        difference = best["M3"]["rate"] - best["M0"]["rate"]
        if difference > 0:
            lines.append(f"M3 的最佳成功率比 M0 高 {100 * difference:.1f} 个百分点。")
        elif difference < 0:
            lines.append(f"M3 的最佳成功率比 M0 低 {-100 * difference:.1f} 个百分点。")
        else:
            lines.append("M3 与 M0 的最佳成功率相同。")
    else:
        lines.append("单种子比较待五个检查点全部完成。")

    lines += ["", "## 训练预算与参数量", ""]
    model_configs = {model: read_json(config_dir / f"{model}.json") for model in MODELS}
    config = next((value for value in model_configs.values() if value is not None), None)
    if config is not None:
        samples, updates = config["samples_per_epoch"], config["updates_per_epoch"]
        lines += [
            f"每种模型每轮使用 {samples:,} 个相同监督样本和 {updates:,} 次优化更新，"
            f"{protocol['epochs']} 轮共 {samples * protocol['epochs']:,} 个样本和 "
            f"{updates * protocol['epochs']:,} 次更新。",
            "",
        ]
    lines += [
        "| 模型 | 已完成轮次 | 训练参数 | 推理参与计算参数 |",
        "|---|---:|---:|---:|",
    ]
    counts = {}
    for model in MODELS:
        progress = training_progress(root, model)
        counts[model] = parameters(model_configs[model], results[model])
        completed = progress["epoch"] if progress else "待记录"
        lines.append(f"| {model} | {completed} | {number(counts[model][0])} | {number(counts[model][1])} |")
    if all(value is not None for counts_for_model in counts.values() for value in counts_for_model):
        train_delta = counts["M3"][0] - counts["M0"][0]
        inference_delta = counts["M3"][1] - counts["M0"][1]
        lines += ["", f"M3 新增 {train_delta:,} 个训练参数和 {inference_delta:,} 个推理参数。"]

    lines += [
        "",
        "## 策略推理耗时",
        "",
        "每个动作块的平均毫秒数由该节点的总耗时除以所有回合的动作块数得到。"
        "预测包含扩散采样，观测包含 M3 的逐步状态更新。",
        "",
        "| 模型 | 轮次 | 预测 ms/块 | 观测更新 ms/块 | 合计 ms/块 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for result in results[model].values():
            timing = report_timing(result)
            if timing is not None:
                lines.append(f"| {model} | {result['epoch']} | " + " | ".join(timing) + " |")
    common = [epoch for epoch in epochs if all(epoch in results[model] for model in MODELS)]
    if common:
        epoch = common[-1]
        m0, m3 = (report_timing(results[model][epoch]) for model in MODELS)
        if m0 is not None and m3 is not None and float(m0[-1]) > 0:
            ratio = float(m3[-1]) / float(m0[-1])
            lines += ["", f"第 {epoch} 轮共同评测节点，M3 每块策略耗时为 M0 的 {ratio:.2f} 倍。"]

    lines += ["", "## 连续状态和代表轨迹", ""]
    verification = read_json(evaluation / "verification/real_closed_loop.json")
    if verification is not None:
        lines.append(
            f"真实环境验证执行了 {verification['steps']} 步，产生 {verification['observations']} 帧观测和 "
            f"{verification['replans']} 次动作块预测。"
            "[验证记录](verification/real_closed_loop.json)"
        )
    calibration = read_json(config_dir / "event_threshold_calibration.json")
    if calibration is not None:
        thresholds = calibration["event_thresholds"]
        lines.append(
            f"训练数据校准后的视觉余弦阈值为 {thresholds['visual']:.9f}，"
            f"历史余弦阈值为 {thresholds['history']:.9f}。"
            "[阈值校准记录](../configs/event_threshold_calibration.json)"
        )
    memory_rows = [row["memory"] for result in results["M3"].values() for row in result["rows"]
                   if row.get("memory") is not None]
    if memory_rows:
        writes = [row["write_count"] for row in memory_rows]
        replacements = sum(row["replacement_count"] for row in memory_rows)
        retained = sum(row["initial_retained"] for row in memory_rows)
        lines.append(
            f"已完成 M3 回合的写入次数范围为 {min(writes)}–{max(writes)}，"
            f"累计遗忘 {replacements} 次，{retained}/{len(memory_rows)} 回合保留初始事件。"
        )
    lines.append("")
    for model in MODELS:
        for success, label in ((True, "成功"), (False, "失败")):
            example = representative(results[model], best[model], success, evaluation)
            lines.append(f"- {model} {label}轨迹。{example}")

    lines += [
        "",
        "## 数据和代码来源",
        "",
        "- 官方 50 条放回方块专家示范。"
        "[RMBench 数据归档](https://huggingface.co/datasets/TianxingChen/RMBench/resolve/main/data/put_back_block/demo_clean.zip)",
        "- 环境和官方成功判定。[RMBench 7d1cdfc](https://github.com/RoboTwin-Platform/RMBench/commit/7d1cdfc)",
        "- 原始扩散策略模型。[Stanford Diffusion Policy 5ba07ac](https://github.com/real-stanford/diffusion_policy/commit/5ba07ac)",
        "- 历史模块与过去动作监督参数。[Memory 3.2 d4e1749](https://github.com/Lin-zh-handsome/memory/commit/d4e1749)",
    ]
    source = read_json(config_dir / "source.json")
    if source and source.get("experiment_commit"):
        lines.append(f"- 本实验实现提交为 {source['experiment_commit']}。")
    if source and source.get("evaluation_commit"):
        lines.append(f"- 正式闭环调度与续跑提交为 {source['evaluation_commit']}。")

    evaluation.mkdir(exist_ok=True)
    target = evaluation / "report.md"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(report(args.root))


if __name__ == "__main__":
    main()
