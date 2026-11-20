"""Report the four-group, three-seed RMBench campaign from complete rollouts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time


TASKS = ("put_back_block", "swap_blocks", "battery_try", "cover_blocks",
         "observe_and_pickup", "rearrange_blocks", "swap_T", "blocks_ranking_try", "press_button")
MODELS = ("M0", "mamba_only", "memory_only", "M5")
EPOCHS = (50, 100, 150, 200, 250)
REUSED = {
    ("put_back_block", "M0"): "rmbench_put_back_block_m0_m3_seed0",
    ("swap_blocks", "M0"): "rmbench_memory_validation/swap_blocks",
    ("battery_try", "M0"): "rmbench_mn_validation/battery_try",
    ("cover_blocks", "M0"): "rmbench_mn_validation/cover_blocks",
    ("cover_blocks", "M5"): "rmbench/m5_goal_spatial_r3m",
}


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render(root):
    done = 0
    title = ("任务", "训练种子", "M0 原扩散", "纯 Mamba", "八槽情景记忆", "完整 M5")
    rows = ["# RMBench 九任务递进消融", "", f"更新时间 {datetime.now().astimezone().isoformat(timespec='seconds')}",
            "", "每格为五个完整检查点的最佳成功率，括号内为检查点轮次。", "",
            "| " + " | ".join(title) + " |", "|---|---|---|---|---|---|"]
    result = {}
    for task in TASKS:
        for seed in (0, 1, 2):
            display = [task, str(seed)]
            for model in MODELS:
                old = REUSED.get((task, model)) if seed == 0 else None
                experiment = root.parents[1] / old if old else root / task / f"seed{seed}"
                summaries = [read(experiment / f"evaluation/{model}/epoch_{ep:03d}/summary.json")
                             for ep in EPOCHS]
                counts = [s["successes"] if (s.get("complete") is True and s.get("episodes") == 50
                          and s.get("model") == model and s.get("epoch") == ep) else None
                          for ep, s in zip(EPOCHS, summaries)]
                if all(n is not None for n in counts):
                    best = max(counts)
                    epoch = EPOCHS[counts.index(best)]
                    result[(task, seed, model)] = best * 2
                    display.append(f"{best * 2}%（{epoch}）")
                    done += 1
                else:
                    completed = sum(n is not None for n in counts)
                    display.append(f"{completed}/5 已评" if completed else "待完成")
            rows.append("| " + " | ".join(display) + " |")
    rows.insert(4, f"已完成 {done}/108 组。")
    rows.extend(("", "## 三个训练种子的均值与样本标准差", "",
                 "| 任务 | M0 原扩散 | 纯 Mamba | 八槽情景记忆 | 完整 M5 |",
                 "|---|---|---|---|---|"))
    for task in TASKS:
        display = [task]
        for model in MODELS:
            values = [result.get((task, seed, model)) for seed in (0, 1, 2)]
            display.append(f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f} 个百分点"
                           if all(value is not None for value in values) else "待完成")
        rows.append("| " + " | ".join(display) + " |")
    groups = {"M(1) 五项": (TASKS[0], TASKS[1], *TASKS[4:7]),
              "M(n) 四项": (TASKS[2], TASKS[3], *TASKS[7:]), "九任务": TASKS}
    rows.extend(("", "## 任务等权平均成功率", "",
                 "各种子分别对组内任务的五点最佳成功率等权平均。三种子汇总给出均值和样本标准差。", "",
                 "| 范围 | 训练种子 | M0 原扩散 | 纯 Mamba | 八槽情景记忆 | 完整 M5 |",
                 "|---|---|---|---|---|---|"))
    for label, tasks in groups.items():
        by_model = {model: [] for model in MODELS}
        for seed in (0, 1, 2):
            display = [label, str(seed)]
            for model in MODELS:
                if all((task, seed, model) in result for task in tasks):
                    mean = statistics.mean(result[task, seed, model] for task in tasks)
                    by_model[model].append(mean)
                    display.append(f"{mean:.1f}%")
                else:
                    display.append("待完成")
            rows.append("| " + " | ".join(display) + " |")
        display = [label, "三种子汇总"]
        for model in MODELS:
            means = by_model[model]
            display.append(f"{statistics.mean(means):.1f}% ± {statistics.stdev(means):.1f} 个百分点"
                           if len(means) == 3 else "待完成")
        rows.append("| " + " | ".join(display) + " |")
    tmp = root / "REPORT.tmp"
    tmp.write_text("\n".join(rows) + "\n")
    tmp.replace(root / "REPORT.md")
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    while True:
        done = render(args.root)
        if done == 108 or not args.follow:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
