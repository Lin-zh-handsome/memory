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
    if (root / "result_sources.json").is_file():
        return render_m5_seed0(root)
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


def render_m5_seed0(root):
    import csv

    sources = json.loads((root / "result_sources.json").read_text())["tasks"]
    titles = {"put_back_block": "方块归位", "swap_blocks": "方块交换", "battery_try": "电池试插",
              "cover_blocks": "方块覆盖", "observe_and_pickup": "观察后抓取",
              "rearrange_blocks": "方块重排", "swap_T": "T形块交换",
              "blocks_ranking_try": "方块排序试验", "press_button": "按按钮"}
    lines = ["# M5 种子 0 九任务进度", "",
             f"更新时间 {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "已完成任务复用旧版结果。其余任务从头训练，RGB 保持 0～1 后进行 ImageNet 标准化。",
             "各任务使用原来的 50 个初态，五节点全部评完后取最佳，同分取较早节点。", "",
             "| 任务 | 预处理版本 | 训练进度 | 50轮 | 100轮 | 150轮 | 200轮 | 250轮 | 最佳 |",
             "|---|---|---|---|---|---|---|---|---|"]
    done = 0
    records = []
    for item in sources:
        experiment = Path(item["root"])
        summaries = [read(experiment / f"evaluation/M5/epoch_{epoch:03d}/summary.json") for epoch in EPOCHS]
        counts = [data["successes"] if data.get("complete") and data.get("episodes") == 50
                  and data.get("model") == "M5" and data.get("epoch") == epoch else None
                  for epoch, data in zip(EPOCHS, summaries)]
        trained = read(experiment / "logs/M5_complete.json").get("epoch") == 250
        epoch_file = experiment / "logs/M5_epochs.csv"
        epoch = None
        if epoch_file.exists():
            with epoch_file.open() as handle:
                history = list(csv.DictReader(handle))
            if history:
                epoch = int(history[-1]["epoch"])
        training = "250轮完成" if trained else f"已完成{epoch}轮" if epoch else "待训练"
        version = "原版RGB" if item["reused_complete"] else "修正RGB"
        display = [titles[item["task"]], version, training]
        for count, data in zip(counts, summaries):
            display.append(f"{count}/50（{count*2}%）" if count is not None
                           else f"已评{data['episodes']}/50，成功{data['successes']}" if data else "待评")
        complete = all(count is not None for count in counts)
        best = None
        if complete:
            best = max(counts)
            display.append(f"{best*2}%（{EPOCHS[counts.index(best)]}轮）")
            done += 1
        else:
            display.append(f"{sum(count is not None for count in counts)}/5节点完成")
        lines.append("| " + " | ".join(display) + " |")
        records.append({**item, "training_complete": trained, "epoch_complete": epoch,
                        "evaluation_complete": complete, "counts": counts,
                        "best_success_rate": best/50 if best is not None else None})
    lines.insert(4, f"完整结果 {done}/9 项。")
    for name, value in (("REPORT.md", "\n".join(lines)+"\n"),
                        ("RESULTS.json", json.dumps({"completed": done, "tasks": records}, indent=2, ensure_ascii=False)+"\n")):
        temp = root / (name + ".tmp")
        temp.write_text(value)
        temp.replace(root / name)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    while True:
        done = render(args.root)
        target = 9 if (args.root / "result_sources.json").is_file() else 108
        if done == target or not args.follow:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
