"""Summarize only complete M5 closed-loop evaluations against the paired M4 baselines."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

EPOCHS = (50, 100, 150, 200, 250)
ENV_SEEDS = set(range(100000, 100050))
POSITIONS = ("左", "中", "右")


def read_results(root, model):
    results = {}
    for epoch in EPOCHS:
        directory = root / f"evaluation/{model}/epoch_{epoch:03d}"
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(row) for row in (directory / "episodes.jsonl").read_text().splitlines()]
        if (summary["model"] != model or summary["epoch"] != epoch or not summary["complete"]
                or summary["episodes"] != 50 or len(rows) != 50 or {r["seed"] for r in rows} != ENV_SEEDS
                or summary["successes"] != sum(row["success"] for row in rows)):
            raise ValueError(f"Unsupported or incomplete {model} result at {directory}")
        results[epoch] = rows
    return results


def stage_metrics(rows):
    stages = Counter()
    layouts = defaultdict(Counter)
    failures = Counter()
    for row in rows:
        order = row["initial_state"]["uncover_order"]
        if sorted(order) != [0, 1, 2]:
            raise ValueError(f"Invalid uncover order for {row['seed']}: {order}")
        layout = "".join(POSITIONS[i] for i in order)
        stage = max([row["final"]["stage"], *(item["stage"] for item in row["stage_transitions"])])
        metrics = {"episodes": 1, "covered": stage >= 3, "first": stage >= 4,
                   "second": stage >= 5, "third": stage >= 6, "success": bool(row["success"]),
                   "first_correct_later_failed": stage >= 4 and not row["success"]}
        stages.update(metrics)
        layouts[layout].update(metrics)
        if not row["success"]:
            failures[row["failure_category"]] += 1
    return {**stages, "layouts": layouts, "failures": failures}


def best_epoch(metrics):
    return max(EPOCHS, key=lambda epoch: (metrics[epoch]["success"], -epoch))


def loss_summary(root):
    with (root / "logs/M5_epochs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 251)):
        raise ValueError("M5 training history doesn't cover 250 epochs")
    fields = ("total_loss", "ddpm_loss", "goal_loss", "effective_goal_weight", "valid_goal_samples")
    sums = {}
    for row in rows:
        epoch = int(row["epoch"])
        parts = {key: float(row[key]) for key in fields}
        if (not all(math.isfinite(value) for value in parts.values())
                or not 0 <= parts["goal_loss"] <= 2.000001 or parts["valid_goal_samples"] <= 0):
            raise ValueError(f"Unexpected M5 loss or label count in epoch {epoch}: {parts}")
        if not math.isclose(parts["effective_goal_weight"], 0.02 * min((epoch - 1) / 9, 1)):
            raise ValueError(f"Goal weight mismatch at epoch {epoch}")
        if epoch in (1, 10, *EPOCHS):
            sums[epoch] = parts
    return sums


def report(root, baselines, seed_roots, defer_seed1=False):
    records = {}
    sections = []
    for seed, path in seed_roots.items():
        reference = read_results(baselines[seed], "M4")
        candidate = read_results(path, "M5")
        for epoch in EPOCHS:
            layout_by_seed = {r["seed"]: r["initial_state"]["uncover_order"] for r in reference[epoch]}
            if any(r["initial_state"]["uncover_order"] != layout_by_seed[r["seed"]]
                   for r in candidate[epoch]):
                raise ValueError(f"M4/M5 layout mismatch for seed {seed} epoch {epoch}")
        m4 = {epoch: stage_metrics(rows) for epoch, rows in reference.items()}
        m5 = {epoch: stage_metrics(rows) for epoch, rows in candidate.items()}
        m4_best, m5_best = best_epoch(m4), best_epoch(m5)
        losses = loss_summary(path)
        readout_path = path / "goal_readout.json"
        readout = json.loads(readout_path.read_text()) if readout_path.is_file() else None
        records[seed] = {"causal_goal_readout": readout, "m4": m4, "m5": m5, "m4_best_epoch": m4_best, "m5_best_epoch": m5_best,
                         "losses": losses}
        diff = 2 * (m5[m5_best]["success"] - m4[m4_best]["success"])
        lines = [f"## 训练种子 {seed}", "",
                 f"M5 最佳 {m5[m5_best]['success'] * 2}%（第 {m5_best} 轮），"
                 f"M4 最佳 {m4[m4_best]['success'] * 2}%（第 {m4_best} 轮），相差 {diff:+d} 个百分点。",
                 "",
                 "| 轮次 | M4 最终成功 | M5 最终成功 | 盖齐 M4 / M5 | 首揭对 M4 / M5 | 第二次揭对 M4 / M5 | 第三次揭对 M4 / M5 | 首揭对后失败 M4 / M5 |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for epoch in EPOCHS:
            a, b = m4[epoch], m5[epoch]
            pair = " | ".join(f"{a[k]} / {b[k]}" for k in
                              ("covered", "first", "second", "third", "first_correct_later_failed"))
            lines.append(f"| {epoch} | {a['success']}/50 | {b['success']}/50 | {pair} |")
        lines += ["", "第二、第三次揭对按同一回合依次通过官方阶段统计。各列分母均为 50。",
                  "", "### 同轮阶段变化", ""]
        for epoch in dict.fromkeys((m5_best, 250)):
            a, b = m4[epoch], m5[epoch]
            lines.append(f"- 第 {epoch} 轮，M5 相比 M4 首揭对 {b['first'] - a['first']:+d} 回合，"
                         f"首揭对后仍失败 {b['first_correct_later_failed'] - a['first_correct_later_failed']:+d} 回合，"
                         f"最终成功 {b['success'] - a['success']:+d} 回合。")
        lines += ["", "### M5 最佳轮次的颜色布局", "",
                  "| 红、绿、蓝位置 | 回合 | M4 首揭对 | M5 首揭对 | M4 成功 | M5 成功 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for layout, b in sorted(m5[m5_best]["layouts"].items()):
            a = m4[m5_best]["layouts"][layout]
            lines.append(f"| {layout} | {b['episodes']} | {a['first']} | {b['first']} | {a['success']} | {b['success']} |")
        lines += ["", "### 训练损失", "",
                  "| 轮次 | 总损失 | 扩散损失 | 目标损失 | 目标权重 | 有效锚点 |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for epoch, values in losses.items():
            lines.append(f"| {epoch} | {values['total_loss']:.5f} | {values['ddpm_loss']:.5f} | "
                         f"{values['goal_loss']:.5f} | {values['effective_goal_weight']:.4f} | "
                         f"{values['valid_goal_samples']:.0f} |")
        lines += ["", f"训练目标损失从第 10 轮 {losses[10]['goal_loss']:.5f} 变化到第 250 轮 "
                  f"{losses[250]['goal_loss']:.5f}，所有轮次均为有限值。"
                  "这衡量模型对训练目标表征的拟合，闭环结果见上表。", ""]
        if readout is not None:
            lines += ["### 共享特征可读性", "",
                      "| 轮次 | 匹配位置余弦相似度 | 其他回合均值对照 | 完成状态检索 | 揭盖位置检索 |",
                      "|---:|---:|---:|---:|---:|"]
            for epoch, m in readout["checkpoints"].items():
                lines.append(f"| {epoch} | {m['mean_matched_cosine']:.4f} | "
                             f"{m['mean_other_episode_prototype_cosine']:.4f} | "
                             f"{m['cross_episode_state_top1']:.1%} | {m['cross_episode_uncover_position_top1']:.1%} |")
            lines += ["", "从因果观测序列得到 z_t，再只为离线诊断调用辅助头。"
                      "查询回合不进入候选库，但策略曾用这些示范训练。闭环决策能力由独立环境评测衡量。", ""]
        sections.extend(lines)
    reached = records[0]["m5"][records[0]["m5_best_epoch"]]["success"] >= 20
    if (reached and not defer_seed1) != (1 in records):
        raise ValueError("Seed 1 presence disagrees with the 40% gate")
    spatial = (root / "annotations/feature_quality.json").is_file()
    teacher = ("冻结官方 R3M-ResNet18 从 layer4 生成 3×3 个 512 维位置表征，每个位置分别归一化。"
               if spatial else "冻结 ImageNet ResNet18（残差视觉网络）离线生成 512 维归一化标签。")
    seed_gate_note = "种子 0 的最佳成功率低于 40%，本轮保留种子 0 的完整结果。"
    if reached:
        seed_gate_note = ("种子 0 达到 40% 门槛。按用户安排，种子 1 暂缓。"
                          if defer_seed1 else "种子 0 达到 40% 门槛，已执行种子 1。")
    intro = ["# M5 下一关键状态辅助监督 · RMBench 方块覆盖", "",
             "M5 保留 M4 的连续历史、事件写入淘汰、交叉注意力和扩散动作网络。"
             "用于重构过去动作的辅助头改为预测下一关键帧视觉表征；预测使用的潜在向量也直接送给动作头。",
             "", "目标来自官方语言标注，同一事件内的相邻动作描述先合并。已核对 50 条示范的"
             "关节和动作数组以及各事件起止图像。最后一张原始采集图像依照官方转换规则作为前一帧的动作来源。"
             "每个事件用最后一张有效 RGB 形成目标，只有前半段的锚点使用目标损失。",
             teacher,
             "参见 [标注统计](annotations/statistics.json) 和 [复现步骤](README.md)。",
             "", seed_gate_note, ""]
    comparison = records[0]
    delta = (comparison["m5"][comparison["m5_best_epoch"]]["success"]
             - comparison["m4"][comparison["m4_best_epoch"]]["success"])
    transfer = ("固定初态上的最佳成功率上升，下一步可以用小规模独立实验验证标注规则能否迁移到电池试插。"
                if delta > 0 else "方块覆盖的结果未提供优先扩展到电池试插的性能依据。")
    if spatial:
        intro += ["[关键帧审查](KEYFRAME_AUDIT.md) · [教师特征质量](FEATURE_QUALITY.md)", ""]
    ending = ["## 迁移判断", "",
              transfer + " 电池试插的操作边界与重复尝试需要另行对齐，迁移效果仍待闭环评测。", "",
              "[各轮阶段及布局明细](comparison.json) · [服务器视频索引](videos.md)", ""]
    (root / "REPORT.md").write_text("\n".join(intro + sections + ending))
    (root / "comparison.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    videos = ["# 服务器视频索引", "",
              "路径相对于服务器 M5 实验根目录。", ""]
    for seed, path in seed_roots.items():
        for video in sorted(path.glob("evaluation/M5/epoch_*/*.mp4")):
            videos.append(f"- 种子 {seed} · {video.relative_to(root)}")
    (root / "videos.md").write_text("\n".join(videos) + "\n")
