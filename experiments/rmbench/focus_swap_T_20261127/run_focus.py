"""Resume only corrected-RGB M5 seed-0 swap_T and finish its five evaluations."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

BASE = Path("/home/linzhenghong/project/rmbench_m0_m3")
PARENT = BASE / "experiments/rmbench/m5_seed0_rgb_corrected"
FOCUS = PARENT / "focus_swap_T_20261127"
PLAN = BASE / ".planning/2026-11-27-focus-swap-t"
CODE = BASE / "source/m5-goal-spatial-r3m/implementation/rmbench"
RUN = PARENT / "swap_T/seed0"
ZONE = ZoneInfo("Asia/Shanghai")
sys.path.insert(0, str(CODE))
from run_m4 import cards, compute_processes


def save(path, value):
    temp = path.with_suffix(".focus.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def progress(state, **details):
    data = {"time": datetime.now(ZONE).isoformat(), "status": state, **details}
    print(json.dumps(data), flush=True)
    save(FOCUS / "logs/supervisor_status.json", data)
    return data


def results():
    values = []
    for epoch in (50, 100, 150, 200, 250):
        directory = RUN / f"evaluation/M5/epoch_{epoch:03d}"
        path = directory / "summary.json"
        marker = json.loads(path.read_text()) if path.exists() else {}
        episodes = {}
        for rows in (directory / "episodes.jsonl", *sorted(directory.glob("shards/*/episodes.jsonl"))):
            if rows.exists():
                for line in rows.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        episodes[row["episode_id"]] = row
        done = marker.get("complete", False) and marker.get("episodes") == 50
        count = len(episodes) if episodes else marker.get("episodes", 0)
        successes = sum(bool(row["success"]) for row in episodes.values()) if episodes else marker.get("successes", 0)
        values.append({"epoch": epoch, "episodes": count, "successes": successes, "complete": bool(done),
                       "success_rate": successes / 50 if done else None})
    marker = RUN / "logs/M5_complete.json"
    trained = marker.exists() and json.loads(marker.read_text()).get("epoch") == 250
    data = {"time": datetime.now(ZONE).isoformat(), "task": "swap_T", "model": "M5", "seed": 0,
            "rgb_normalization": "identity", "training_complete": trained, "evaluations": values,
            "complete": bool(trained and all(row["complete"] for row in values))}
    save(FOCUS / "RESULTS.json", data)
    text = ["# T形块交换 M5 种子0（修正RGB）", "", "更新时间：" + data["time"], "",
            "训练：" + ("250轮完成" if trained else "继续训练至250轮"), "",
            "| 权重轮次 | 已评回合 | 成功回合 | 成功率 | 状态 |", "|---|---:|---:|---:|---|"]
    for row in values:
        rate = str(round(100 * row["success_rate"], 1)) + "%" if row["complete"] else "未完成"
        text.append(f'| {row["epoch"]} | {row["episodes"]}/50 | {row["successes"]} | {rate} | {"完整" if row["complete"] else "待完成"} |')
    text += ["", "只运行该任务。使用一张空闲H200、四个任务槽，先1项训练与3项评测，训练完成后4项评测。",
             "官方规则、模型、训练预算、配对初态和每节点50回合保持不变。", ""]
    (FOCUS / "REPORT.md").write_text("\n".join(text))
    return data


def main():
    cfg = json.loads((RUN / "configs/protocol.json").read_text())
    assert cfg["task"] == "swap_T" and cfg["seed"] == 0 and cfg["epochs"] == 250
    assert cfg["vision"]["rgb_normalization"] == "identity"
    assert (RUN / "checkpoints/M5/latest.pt").is_file()
    busy = compute_processes()
    card = next(card for card in cards() if card.uuid not in busy and card.allocated_mib <= 128)
    env = os.environ.copy()
    reporter = Path("/proc/3683036/environ")
    if reporter.exists():
        env = dict(os.fsdecode(value).split("=", 1) for value in reporter.read_bytes().split(b"\0") if b"=" in value)
    dependencies = [str(BASE.parent / "coffee_dp_official/source/diffusion_policy"),
                    str(BASE.parent / "benchmark/RoboTwin/envs/curobo/src")]
    dependencies[0] = "/home/linzhenghong/project/coffee_dp_official/source/diffusion_policy"
    dependencies[1] = "/home/linzhenghong/project/benchmark/RoboTwin/envs/curobo/src"
    env["PYTHONPATH"] = os.pathsep.join([*dependencies, env.get("PYTHONPATH", "")])
    env["PYTHONNOUSERSITE"] = "1"
    env["RMBENCH_ISOLATE_RENDER_GPU"] = "1"
    command = ["flock", "-n", "-F", str(BASE / "experiments/rmbench/m5_ablation_seeds/queue.lock"),
               "flock", "-n", "-F", str(PARENT / "queue.lock"),
               sys.executable, "-u", str(CODE / "run_ablation.py"), "--queue", str(FOCUS / "queue.json"),
               "--rmbench", str(BASE / "source/RMBench"), "--gpu", card.uuid, "--train-slots", "1",
               "--eval-slots", "3", "--split-evaluations", "--fill-idle-slots"]
    results()
    with (FOCUS / "logs/campaign.log").open("ab") as output:
        job = subprocess.Popen(command, cwd=CODE.parent.parent, env=env, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=output, start_new_session=True)
    launch = {"pid": job.pid, "supervisor_pid": os.getpid(), "gpu_index": card.index, "gpu_uuid": card.uuid,
              "processes": 4, "scope": "swap_T M5 seed0 corrected RGB", "queue": str(FOCUS / "queue.json")}
    save(FOCUS / "logs/launch.json", launch)
    save(PARENT / "logs/campaign_status.json", {"status": "running", "scope_tasks": ["swap_T"],
         "other_tasks_paused": True, "time": datetime.now(ZONE).isoformat(), **launch})
    progress("running", **launch)
    exit_code = job.wait()
    data = results()
    state = "completed" if exit_code == 0 and data["complete"] else "failed"
    outcome = progress(state, exit_code=exit_code, **launch, results=str(FOCUS / "RESULTS.json"))
    save(PARENT / "logs/campaign_status.json", {"status": "paused", "reason": "focused_task_" + state,
         "scope_tasks": ["swap_T"], "other_tasks_paused": True, "focus": outcome})
    with (PLAN / "progress.md").open("a") as handle:
        handle.write("\n## " + outcome["time"] + " " + state + "\n\n" + json.dumps(data, ensure_ascii=False) + "\n")
    if data["complete"]:
        plan = PLAN / "task_plan.md"
        plan.write_text(plan.read_text().replace("- [ ] 完成250轮", "- [x] 完成250轮").replace("- [ ] 完成五节点", "- [x] 完成五节点"))
    subprocess.run([sys.executable, str(CODE / "report_ablation.py"), str(PARENT)], check=False, env=env)
    with (PARENT / "progress.md").open("a") as handle:
        handle.write("\n## T形块交换单任务 " + state + "\n\n" + json.dumps(data, ensure_ascii=False) + "\n")

try:
    main()
except Exception as error:
    progress("failed", error=str(error))
    raise
