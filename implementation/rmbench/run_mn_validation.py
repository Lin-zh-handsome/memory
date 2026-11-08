"""Run Cover Blocks, followed by Battery Try, on one physical GPU."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from campaign import CODE, Campaign, record
from run_memory_validation import (
    atomic_json, prepare_data, stage_configs, validate_trainer, verify_result, wait_for_trainer,
)
from summarize import read_json, report

TASKS = {"cover_blocks": 1500, "battery_try": 1000}
MODELS = ("M0", "M3")


class ExternalDependencyError(RuntimeError):
    pass


def log_progress(root, detail):
    with (root / "progress.md").open("a", encoding="utf-8") as log:
        log.write(f"- {datetime.now().astimezone().isoformat(timespec='seconds')} {detail}\n")
    record("progress", detail=detail)


async def wait_download(destination):
    pidfile = destination / "download.pid"
    if not pidfile.is_file():
        return
    pid = int(pidfile.read_text())
    proc = Path(f"/proc/{pid}")
    while proc.exists():
        try:
            command = (proc / "cmdline").read_bytes()
        except FileNotFoundError:
            break
        if b"scripts/_download_data.sh" not in command or str(destination).encode() not in command:
            break
        await asyncio.sleep(10)


async def prepare_with_retry(args, task, root):
    await wait_download(root)
    download_log = root / "logs/download.log"
    for attempt in range(3):
        try:
            return prepare_data(args, task, root)
        except subprocess.CalledProcessError as error:
            if not any("_download_data.sh" in str(arg) for arg in error.cmd):
                raise
            failure = download_log.read_text(errors="replace")[-5000:]
            if not any(marker in failure for marker in
                       ("ReadTimeout", "ConnectionError", "502", "503", "504", "ConnectTimeout", "429")):
                raise
            log_progress(args.root, f"{task} 下载尝试 {attempt + 1} 遇到外部网络故障。日志 {download_log}。")
            if attempt == 2:
                raise ExternalDependencyError(f"官方示范下载三次未成功。{download_log}") from error
            await asyncio.sleep(20 * (attempt + 1))


def outcome_report(root, task, result):
    failures = {model: Counter() for model in MODELS}
    timeout = Counter()
    batteries = {model: Counter() for model in MODELS}
    cfg = read_json(root / "configs/protocol.json")
    for model in MODELS:
        for epoch in cfg["eval_epochs"]:
            file = root / f"evaluation/{model}/epoch_{epoch:03d}/episodes.jsonl"
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if not row["success"]:
                    failures[model][row["failure_category"]] += 1
                    timeout[model] += row["timed_out"]
                attempt = row.get("battery_attempts")
                if attempt:
                    batteries[model]["all_attempts"] += attempt["attempt_count"]
                    batteries[model]["repeated_attempts"] += attempt["repeated_count"]
                    batteries[model]["episodes_with_repeat"] += bool(attempt["repeated_count"])
                    batteries[model]["gauge_activated"] += attempt["gauge_activated"]
                    batteries[model]["successful_feedback"] += attempt["successful_feedback"]
    result["failure_categories"] = {model: dict(values) for model, values in failures.items()}
    result["timeouts"] = dict(timeout)
    if task == "battery_try":
        result["battery_attempts"] = {model: dict(values) for model, values in batteries.items()}

    path = root / "evaluation/report.md"
    lines = ["", "## 失败类别", ""]
    for model in MODELS:
        lines.append(f"- {model} 阶段与顺序失败计数 {dict(failures[model])}，达到步数上限 {timeout[model]} 回合。")
        if task == "battery_try":
            lines.append(f"- {model} 组合与仪表反馈统计 {dict(batteries[model])}。")
    lines.append("阶段和轨迹可区分可观测的操作失误、顺序错误与超时。内部记忆原因需额外对照。")
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")


class SingleGpuCampaign(Campaign):
    def __init__(self, args):
        super().__init__(args)
        self.groups = set()
        self.launch_lock = asyncio.Lock()

    def check_idle(self, gpu):
        query = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id,memory.free",
             "--format=csv,noheader,nounits"], text=True,
        )
        pci, free = query.strip().split(",")
        self.pcis[gpu] = pci.strip()
        if int(free) < 24000:
            raise RuntimeError(f"Insufficient free memory for a new job on {gpu}: {free.strip()} MiB")
        processes = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
        )
        for line in processes.splitlines():
            try:
                group = os.getpgid(int(line.strip()))
            except ProcessLookupError:
                continue
            if group not in self.groups:
                raise RuntimeError(f"GPU {gpu} is occupied by a different workload (PID {line.strip()})")

    async def start(self, name, gpu, script, *arguments):
        async with self.launch_lock:
            self.check_idle(gpu)
            child = await super().start(name, gpu, script, *arguments)
            self.groups.add(child.pid)
            return child

    async def finish(self, name, child):
        try:
            return await super().finish(name, child)
        finally:
            self.groups.discard(child.pid)


async def run_slots(campaign, gpu, workers, adopted):
    cfg = campaign.config
    pending = [(model, epoch) for epoch in cfg["eval_epochs"] for model in MODELS]
    assigned = [list(MODELS)] if workers == 1 else [[model] for model in MODELS]
    training_failed = asyncio.Event()

    async def slot(models):
        for model in models:
            marker = campaign.logs / f"{model}_complete.json"
            if marker.is_file():
                continue
            try:
                if model in adopted:
                    await wait_for_trainer(adopted[model], model, gpu, campaign)
                else:
                    await campaign.train(model, gpu)
            except Exception:
                training_failed.set()
                raise
        while pending and not training_failed.is_set():
            ready = next((
                (model, epoch) for model, epoch in pending
                if (campaign.args.root / f"checkpoints/{model}/epoch_{epoch:03d}.pt").is_file()
            ), None)
            if ready is None:
                if all((campaign.logs / f"{model}_complete.json").is_file() for model in MODELS):
                    raise RuntimeError("Training ended without all evaluation checkpoints")
                await asyncio.sleep(15)
                continue
            model, epoch = ready
            pending.remove(ready)
            point = read_json(campaign.args.root / f"evaluation/{model}/epoch_{epoch:03d}/summary.json")
            if point is None or not point["complete"]:
                await campaign.evaluate(gpu, model, epoch)

    outcomes = await asyncio.gather(*(slot(models) for models in assigned), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            raise outcome


async def run_task(args, task, expected_limit):
    root = args.root / task
    (root / "logs").mkdir(parents=True, exist_ok=True)
    completed = root / "result.json"
    if completed.is_file():
        return read_json(completed)

    log_progress(args.root, f"开始 {task}。")
    stage_configs(args, task, root)
    data = await prepare_with_retry(args, task, root)
    campaign = SingleGpuCampaign(SimpleNamespace(root=root, data=data, rmbench=args.rmbench,
                                        camera_shader="rt", eval_batch_size=2, switch_at=None))
    config = campaign.config
    if (config["max_episode_steps"] != expected_limit or config["seed"] != 0
            or config["epochs"] != 250 or config["models"] != list(MODELS)
            or config["eval_episodes"] != 50 or config["eval_epochs"] != [50, 100, 150, 200, 250]):
        raise RuntimeError(f"Unexpected experimental protocol for {task}")

    gpu = args.gpus[0]
    adopted = {}
    if task == "cover_blocks" and args.adopt_cover_m0 and not (campaign.logs / "M0_complete.json").is_file():
        pid = args.adopt_cover_m0
        validate_trainer(pid, "M0", gpu, campaign)
        adopted["M0"] = pid
        campaign.groups.add(os.getpgid(pid))
    campaign.check_idle(gpu)
    if "event_thresholds" not in config:
        child = await campaign.start("calibrate", gpu, "calibrate.py", "--root", root, "--data", data)
        await campaign.finish("calibrate", child)
    check = root / "configs/preflight_pass.json"
    if not check.is_file():
        child = await campaign.start("preflight", gpu, "preflight_mn.py", root, args.rmbench)
        await campaign.finish("preflight", child)
        atomic_json(check, {"task": task, "passed": True, "log": "../logs/preflight.log"})
    await campaign.prepare(gpu)
    seeds = read_json(root / "evaluation/reset_seeds.json")["seeds"]
    if not (root / "evaluation/verification/real_closed_loop.json").is_file():
        child = await campaign.start("verify", gpu, "evaluate.py", *campaign.evaluator_args(gpu),
                                     "verify", "--seed", seeds[0])
        await campaign.finish("verify", child)
    log_progress(args.root, f"{task} 的数据、模型梯度、记忆和真实环境接口验证通过。开始单卡正式运行。")

    await run_slots(campaign, gpu, args.workers, adopted)
    result = verify_result(root, task)
    report(root)
    outcome_report(root, task, result)
    atomic_json(completed, result)
    log_progress(args.root, f"{task} 完成两模型训练与 500 回合评测。")
    return result


def summary(root, results):
    lines = [
        "# RMBench 单种子闭环对比",
        "",
        "| 任务 | 记忆类别 | M0 最佳 | M3 最佳 |",
        "|---|---|---:|---:|",
        "| Put Back Block | M(1) | 26% | 100% |",
        "| Swap Blocks | M(1) | 18% | 100% |",
    ]
    decisions = []
    for task, result in results.items():
        if "blocked" in result:
            lines.append(f"| {task} | M(n) | 未完成 | 未完成 |")
            decisions.append(f"- {task}。{result['blocked']}")
            continue
        m0, m3 = (result["models"][model] for model in MODELS)
        lines.append(f"| {task} | M(n) | {m0['rate']:.0%} | {m3['rate']:.0%} |")
        outcome = "优于" if m3["rate"] > m0["rate"] else "持平于" if m3["rate"] == m0["rate"] else "低于"
        decisions.append(f"- {task} 的 M3 {outcome} M0。失败类别为 {result['failure_categories']}。"
                         f"[逐检查点成绩与失败分析](../{task}/evaluation/report.md)")
    lines += ["", "前两项引用已发布结果。新增任务采用用户指定的训练种子 0、每检查点 50 回合和五检查点最佳成绩。",
              "", *decisions, "",
              "顺序错误和动作执行受阻根据可观测阶段与轨迹统计。完整 M3 的结果用于判断初步优势，记忆模块的独立贡献需要对照。"]
    (root / "summary/four_task_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--adopt-cover-m0", type=int)
    args = parser.parse_args()
    args.root, args.rmbench = args.root.resolve(), args.rmbench.resolve()
    args.gpus = [args.gpu]
    args.protocol_template = CODE / "configs/protocol.json"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    with (args.root / "mn_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                              "--rmbench", str(args.rmbench), "--gpu", args.gpu, "--workers", str(args.workers)])
        (args.root / "summary/reproduce.txt").write_text(
            f"export PYTHONNOUSERSITE=1\nexport PYTHONPATH={shlex.quote(os.environ['PYTHONPATH'])}\n{command}\n"
        )
        results = {}
        for task, limit in TASKS.items():
            try:
                results[task] = await run_task(args, task, limit)
            except ExternalDependencyError as error:
                if task != "cover_blocks":
                    raise
                results[task] = {"blocked": str(error)}
                log_progress(args.root, f"Cover Blocks 外部依赖未解除，继续 Battery Try。{error}")
            summary(args.root, results)
        log_progress(args.root, "P1 调度结束。查看四任务汇总。")


if __name__ == "__main__":
    asyncio.run(main())
