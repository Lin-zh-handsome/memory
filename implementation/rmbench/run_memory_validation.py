"""Run Swap Blocks, then Cover Blocks, using the audited RMBench policies."""

import argparse
import asyncio
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from campaign import CODE, Campaign, record
from summarize import best_result, checkpoint_result, read_json, report


TASKS = ("swap_blocks", "cover_blocks")
MODEL_NAMES = ("M0", "M3")
ASSETS = {"swap_blocks": ("002_breadbasket",), "cover_blocks": ("003_cover",),
          "battery_try": ("017_battery_slot_gauge", "018_battery")}
DATASET = "TianxingChen/RMBench"


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_result(root, task):
    cfg = read_json(root / "configs/protocol.json")
    evaluation = root / "evaluation"
    seeds = read_json(evaluation / "reset_seeds.json")["seeds"]
    results, failures, histories = {}, {}, []
    reference = {}
    for model in MODEL_NAMES:
        points = {}
        stages = Counter()
        for epoch in cfg["eval_epochs"]:
            result = checkpoint_result(evaluation, model, epoch, seeds, cfg["eval_episodes"])
            if result is None:
                raise RuntimeError(f"{task} {model} epoch {epoch} has not completed")
            points[epoch] = result
            for row in result["rows"]:
                initial = row["initial_state"]
                former = reference.setdefault(row["seed"], initial)
                same_keys = initial.keys() == former.keys()
                same_poses = same_keys and all(
                    np.allclose(initial[key], former[key], rtol=0, atol=1e-5) if key == "qpos"
                    else initial[key] == former[key] if key in ("occupied_baskets", "empty_basket", "uncover_order", "target_combination")
                    else all(np.allclose(a[name], b[name], rtol=0, atol=5e-4)
                             for a, b in zip(initial[key], former[key], strict=True) for name in ("p", "q"))
                    for key in initial
                )
                if not same_poses:
                    raise RuntimeError(f"Initial state changed for {task} seed {row['seed']}")
                if not row["success"]:
                    stages[max(state["stage"] for state in [*row["stage_transitions"], row["final"]])] += 1
                memory = row["memory"]
                if model == "M3":
                    times = memory["final_event_times"]
                    if (not memory["initial_retained"] or times[0] != 0
                            or sorted(set(times)) != times or max(memory["capacities"]) > 8
                            or memory["observations"] != row["steps"] + 1
                            or memory["final_event_age_steps"] != [row["steps"] - t for t in times]):
                        raise RuntimeError(f"Invalid M3 memory lifecycle in {task} seed {row['seed']}")
                    histories.append(memory)
        best = best_result(points, cfg["eval_epochs"])
        results[model] = {"best_epoch": best["epoch"], "rate": best["rate"],
                          "successes": best["successes"],
                          "epoch_successes": {epoch: point["successes"] for epoch, point in points.items()}}
        failures[model] = dict(stages)
    case = {"swap_blocks": "empty_basket", "cover_blocks": "uncover_order",
            "battery_try": "target_combination"}[task]
    configurations = Counter(str(state[case]) for state in reference.values())
    if len(configurations) < 2:
        raise RuntimeError(f"Fixed seeds for {task} lack placement diversity")
    return {
        "task": task, "episodes": 500, "models": results, "failure_stages": failures,
        "initial_layout_counts": dict(configurations),
        "memory": {"initial_retained": sum(h["initial_retained"] for h in histories),
                   "episodes": len(histories),
                   "writes": sum(h["write_count"] for h in histories),
                   "replacements": sum(h["replacement_count"] for h in histories)},
    }


def update_plan(root, phase, text):
    plan = root / "task_plan.md"
    lines = plan.read_text().splitlines()
    lines = [f"{line.rsplit(' [', 1)[0]} [{phase[1]}]" if line.startswith(f"{phase[0]}. ") else line
             for line in lines]
    plan.write_text("\n".join(lines) + "\n")
    with (root / "progress.md").open("a", encoding="utf-8") as log:
        log.write(f"- {text}\n")
    record("phase", phase=phase[0], state=phase[1], detail=text)


def stage_configs(args, task, destination):
    config_dir = destination / "configs"
    config_dir.mkdir(exist_ok=True)
    original = getattr(args, "protocol_template", args.root / "audit/put_back_original/experiment/configs/protocol.json")
    cfg = read_json(original)
    cfg.pop("event_thresholds", None)
    limit = yaml.safe_load((args.rmbench / "env_cfg/task_config/_eval_step_limit.yml").read_text())[task]
    cfg.update(task=task, max_episode_steps=limit,
               resource_policy={"max_gpus": len(args.gpus), "routine_check_seconds": 3600})
    protocol = config_dir / "protocol.json"
    if not protocol.exists():
        atomic_json(protocol, cfg)
    else:
        saved = read_json(protocol)
        if saved["task"] != task or saved["max_episode_steps"] != limit:
            raise RuntimeError("Existing protocol belongs to a different task")
    versions = {}
    for package in ("torch", "torchvision", "mamba-ssm", "causal-conv1d", "diffusers", "sapien",
                    "warp-lang", "curobo", "h5py", "opencv-python", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    commit = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    benchmark = subprocess.check_output(["git", "-C", str(args.rmbench), "rev-parse", "HEAD"],
                                        text=True).strip()
    source_path = config_dir / "source.json"
    previous_source = read_json(source_path) or {}
    atomic_json(source_path, {
        "experiment_commit": previous_source.get("experiment_commit", commit),
        "evaluation_commit": commit, "rmbench": benchmark,
        "source_base": "4577eda", "asset_model": ASSETS[task],
        "environment_patches": ["rmbench_render_device.patch", "rmbench_qpos_workers.patch"],
        "rollout_planning": "CPU TOPP and original gripper interpolation for joint-space actions",
        "expert_archive": f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/{task}/demo_clean.zip",
        "package_versions": versions, "python": sys.version.split()[0],
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    })


def cpu_job(command, log, env=None):
    with log.open("ab") as output:
        subprocess.run([str(arg) for arg in command], check=True, env=env, stdout=output,
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def prepare_data(args, task, destination):
    archive = destination / f"archive/data/{task}/demo_clean.zip"
    prepared = destination / "prepared"
    logs = destination / "logs"
    assets = args.rmbench / "assets"
    env = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "HF_HUB_DISABLE_XET": "1",
           "HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "30",
           "PATH": str(Path(sys.executable).parent) + ":" + os.environ["PATH"]}
    if not archive.exists():
        cpu_job(["bash", args.rmbench / "scripts/_download_data.sh", task,
                 "--cache", destination / "archive"], logs / "download.log", env)
    missing = [asset for asset in ASSETS[task] if not any((assets / "objects" / asset).rglob("*.json"))]
    if missing:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=DATASET, repo_type="dataset", endpoint=env["HF_ENDPOINT"],
                          local_dir=assets, allow_patterns=[f"objects/{asset}/**" for asset in missing])
    if not (prepared / "manifest.json").exists():
        cpu_job([sys.executable, "-u", CODE / "data.py", archive, prepared, args.rmbench, "--task", task],
                logs / "prepare_data.log")
    manifest = read_json(prepared / "manifest.json")
    if manifest["task"] != task or len(manifest["episodes"]) != 50:
        raise RuntimeError("Prepared data is incomplete or belongs to another task")
    return prepared


async def prepare_evaluation(campaign, gpus):
    reset_file = campaign.args.root / "evaluation" / "reset_seeds.json"
    manifest = read_json(reset_file)
    if manifest and manifest["complete"]:
        return
    if len(gpus) == 1:
        await campaign.prepare(gpus[0])
        return

    start = campaign.config["eval_seed_start"]
    limit = start + 10 * campaign.config["eval_episodes"]
    while start < limit:
        shards = []
        for gpu in gpus:
            first, end = start, min(start + 16, limit)
            if first >= end:
                break
            name = f"prepare_seeds_{first}_{end}"
            child = await campaign.start(name, gpu, "evaluate.py", *campaign.evaluator_args(gpu),
                                         "prepare-seeds", "--candidate-start", first, "--candidate-stop", end)
            shards.append(campaign.finish(name, child))
            start = end
        results = await asyncio.gather(*shards, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result
        child = await campaign.start("merge_seeds", gpus[0], "evaluate.py",
                                     *campaign.evaluator_args(gpus[0]), "merge-seeds")
        await campaign.finish("merge_seeds", child)
        if read_json(reset_file)["complete"]:
            return
    raise RuntimeError("Insufficient official expert-valid evaluation seeds")


def validate_trainer(pid, model, gpu, campaign):
    process = Path(f"/proc/{pid}")
    argv = (process / "cmdline").read_bytes().split(b"\0")
    env = (process / "environ").read_bytes().split(b"\0")
    expected = [str(CODE / "train.py").encode(), b"--root", str(campaign.args.root).encode(),
                b"--data", str(campaign.args.data).encode(), b"--model", model.encode()]
    if (process.stat().st_uid != os.getuid() or argv[2:-1] != expected
            or f"CUDA_VISIBLE_DEVICES={gpu}".encode() not in env):
        raise RuntimeError(f"PID {pid} does not match the requested training job")


def describe_evaluator(pid, campaign, gpu):
    process = Path(f"/proc/{pid}")
    argv = (process / "cmdline").read_bytes().split(b"\0")
    env = (process / "environ").read_bytes().split(b"\0")
    expected = [str(CODE / "evaluate.py").encode(), b"--experiment", str(campaign.args.root).encode()]
    suffix = argv[-8:-1]
    if (process.stat().st_uid != os.getuid() or argv[2:5] != expected
            or len(suffix) != 7 or suffix[:2] != [b"rollout", b"--model"]
            or suffix[3] != b"--epoch" or suffix[5:] != [b"--batch-size", b"2"]
            or f"CUDA_VISIBLE_DEVICES={gpu}".encode() not in env):
        raise RuntimeError(f"PID {pid} is not an evaluator for {gpu}")
    model, epoch = suffix[2].decode(), int(suffix[4])
    if model not in MODEL_NAMES or epoch not in campaign.config["eval_epochs"]:
        raise RuntimeError(f"PID {pid} is running an unexpected evaluation")
    return pid, model, epoch


async def wait_for_process(pid):
    process_stat = Path(f"/proc/{pid}/stat")
    start_time = process_stat.read_text().rsplit(")", 1)[1].split()[19]
    while True:
        try:
            fields = process_stat.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            break
        if fields[0] in ("Z", "X") or fields[19] != start_time:
            break
        await asyncio.sleep(12)


async def wait_for_trainer(pid, model, gpu, campaign):
    if (campaign.logs / f"{model}_complete.json").is_file():
        return
    validate_trainer(pid, model, gpu, campaign)
    record("attached_training", model=model, gpu=gpu, pid=pid)
    await wait_for_process(pid)
    if not (campaign.logs / f"{model}_complete.json").is_file():
        record("training_recovery", model=model, gpu=gpu, previous_pid=pid)
        campaign.check_idle(gpu)
        await campaign.train(model, gpu)


async def train_and_evaluate(campaign, gpus, adopted, active_evaluations):
    pending = [(model, epoch) for epoch in campaign.config["eval_epochs"] for model in MODEL_NAMES]
    training = [model for model in MODEL_NAMES if not (campaign.logs / f"{model}_complete.json").exists()]
    training_failed = asyncio.Event()
    for _, model, epoch in active_evaluations.values():
        pending.remove((model, epoch))

    async def worker(gpu, models):
        for model in models:
            try:
                if model in adopted:
                    await wait_for_trainer(adopted[model], model, gpu, campaign)
                else:
                    campaign.check_idle(gpu)
                    await campaign.train(model, gpu)
            except Exception:
                training_failed.set()
                raise
        if gpu in active_evaluations:
            pid, model, epoch = active_evaluations[gpu]
            record("attached_evaluation", model=model, epoch=epoch, gpu=gpu, pid=pid)
            await wait_for_process(pid)
            done = read_json(campaign.args.root / f"evaluation/{model}/epoch_{epoch:03d}/summary.json")
            if not done or not done["complete"]:
                campaign.check_idle(gpu)
                await campaign.evaluate(gpu, model, epoch)
            else:
                report(campaign.args.root)
        while pending:
            if training_failed.is_set():
                return
            job = next((
                (model, epoch) for model, epoch in pending
                if (campaign.args.root / "checkpoints" / model / f"epoch_{epoch:03d}.pt").is_file()
            ), None)
            if job is None:
                if all((campaign.logs / f"{model}_complete.json").exists() for model in MODEL_NAMES):
                    raise RuntimeError("Training ended with missing evaluation checkpoints")
                await asyncio.sleep(12)
                continue
            model, epoch = job
            result = read_json(campaign.args.root / f"evaluation/{model}/epoch_{epoch:03d}/summary.json")
            if result is not None and result["complete"]:
                pending.remove(job)
                continue
            try:
                campaign.check_idle(gpu)
            except RuntimeError as error:
                record("evaluation_gpu_skipped", gpu=gpu, reason=str(error))
                return
            pending.remove(job)
            await campaign.evaluate(gpu, model, epoch)

    assigned = [[] for _ in gpus]
    if adopted:
        for index, model in enumerate(MODEL_NAMES):
            if model in training:
                assigned[index].append(model)
    else:
        for index, model in enumerate(training):
            assigned[index % len(gpus)].append(model)
    outcomes = await asyncio.gather(*(worker(gpu, models) for gpu, models in zip(gpus, assigned)),
                                    return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            raise outcome
    if pending:
        raise RuntimeError("No available GPU workers for the remaining evaluation checkpoints")


async def run_task(args, task, phase):
    root = args.root / task
    root.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    complete = root / "result.json"
    if complete.exists():
        return read_json(complete)

    update_plan(args.root, (phase, "in_progress"), f"{task} 进入数据、训练和评测阶段。")
    stage_configs(args, task, root)
    data = prepare_data(args, task, root)
    campaign = Campaign(SimpleNamespace(root=root, data=data, rmbench=args.rmbench,
                                        camera_shader="rt", eval_batch_size=2, switch_at=None))
    adopted = dict(zip(MODEL_NAMES, args.adopt_swap_trainers)) if task == "swap_blocks" else {}
    active_evaluations = {}
    if task == "swap_blocks":
        active_evaluations = {
            gpu: describe_evaluator(pid, campaign, gpu)
            for gpu, pid in zip(args.gpus, args.adopt_swap_evaluators)
        }
    gpus = []
    for index, gpu in enumerate(args.gpus):
        try:
            if gpu in active_evaluations:
                campaign.pcis[gpu] = subprocess.check_output(
                    ["nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                    text=True,
                ).strip()
            elif adopted and index < len(MODEL_NAMES):
                model = MODEL_NAMES[index]
                if not (campaign.logs / f"{model}_complete.json").is_file():
                    validate_trainer(adopted[model], model, gpu, campaign)
                pci = subprocess.check_output([
                    "nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id", "--format=csv,noheader",
                ], text=True).strip()
                campaign.pcis[gpu] = pci
            else:
                campaign.check_idle(gpu)
            gpus.append(gpu)
        except RuntimeError as error:
            if adopted and index < len(MODEL_NAMES):
                raise
            record("gpu_skipped", gpu=gpu, reason=str(error))
    if not gpus:
        raise RuntimeError("No selected GPU is available")

    primary = gpus[0]
    if "event_thresholds" not in campaign.config:
        child = await campaign.start("calibrate", primary, "calibrate.py", "--root", root, "--data", data)
        await campaign.finish("calibrate", child)
    campaign.config = read_json(root / "configs/protocol.json")
    await prepare_evaluation(campaign, gpus)
    seeds = read_json(root / "evaluation/reset_seeds.json")["seeds"]
    if not (root / "evaluation/verification/real_closed_loop.json").exists():
        child = await campaign.start("verify", primary, "evaluate.py", *campaign.evaluator_args(primary),
                                     "verify", "--seed", seeds[0])
        await campaign.finish("verify", child)
    await train_and_evaluate(campaign, gpus, adopted, active_evaluations)
    result = verify_result(root, task)
    report(root)
    atomic_json(complete, result)
    update_plan(args.root, (phase, "complete"), f"{task} 已完成两模型训练和 500 回合闭环。")
    return result


def final_report(root, results):
    names = {"swap_blocks": "Swap Blocks", "cover_blocks": "Cover Blocks"}
    lines = [
        "# RMBench 三任务单种子闭环对比",
        "",
        "所有任务使用 demo_clean 和训练种子 0。每个模型独立从头训练 250 轮，在同一任务的固定 50 个环境初始条件上评测五个检查点，取最高成功率。同分取最早节点。",
        "",
        "| 任务 | 记忆类别 | M0 最佳 | M3 最佳 | 差值（百分点） |",
        "|---|---|---:|---:|---:|",
        "| Put Back Block | M(1) | 26%（第 250 轮） | 100%（第 50 轮） | +74 |",
    ]
    for result, category in zip(results, ("M(1)", "M(n)")):
        m0, m3 = (result["models"][model] for model in MODEL_NAMES)
        lines.append(f"| {names[result['task']]} | {category} | {m0['rate']:.0%}（第 {m0['best_epoch']} 轮） | "
                     f"{m3['rate']:.0%}（第 {m3['best_epoch']} 轮） | {100 * (m3['rate'] - m0['rate']):+.0f} |")
    improved = [names[r["task"]] for r in results if r["models"]["M3"]["rate"] > r["models"]["M0"]["rate"]]
    lines += [
        "",
        "M3 在 Put Back Block 取得更高成功率。两个新增任务中，"
        + ("M3 在 " + "、".join(improved) + " 取得更高成功率。" if improved else "M3 未取得更高成功率。"),
        "这些数据比较完整 M3 与原始 M0 的单种子表现，3.3 情景记忆的独立贡献需要后续对照。",
        "",
        "Put Back Block 的关键真实性和公平性检查通过。旧日志未保存完整物理初态，固定种子复核覆盖四类官方起始托盘。"
        "[审计报告](../audit/put_back_block_audit.md)",
        "",
        "Put Back Block 的 M0 失败主要发生在按键后未能回位，五个节点合计阶段 0 有 95 次、阶段 1 有 118 次。M3 的 250 回合均成功。",
    ]
    for result in results:
        task = result["task"]
        memory = result["memory"]
        lines += ["", f"{names[task]} 的 M3 在 {memory['episodes']} 个回合全部保留初始事件，"
                  f"写入 {memory['writes']:,} 次，替换 {memory['replacements']:,} 次。"
                  "核对通过的轨迹均以时间顺序记录、每帧更新一次且容量不超过 8。",
                  f"[节点成绩、轨迹与诊断](../{task}/evaluation/report.md)"]
        for model in MODEL_NAMES:
            stages = result["failure_stages"][model]
            lines.append(f"- {model} 五个节点的失败阶段计数为 {stages}。" if stages else f"- {model} 五个节点均成功。")
    lines += [
        "",
        "Swap Blocks 阶段按观测进展计数。0 为首次搬运前，1 为首块到空托盘，2 为第二块移到首块原位，3 为两块交换后等待按键，4 为官方成功。",
        "Cover Blocks 阶段使用官方状态序列指针。0–3 表示依次盖上三个方块，4–6 表示按颜色顺序完成揭盖。",
        "本次采用自定义 50 回合和五节点取最佳协议。逐回合记录、环境版本及完整配置位于各任务目录。",
    ]
    (root / "summary/three_task_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--adopt-swap-trainers", nargs=2, type=int, default=(),
                        metavar=("M0_PID", "M3_PID"))
    parser.add_argument("--swap-only", action="store_true")
    parser.add_argument("--adopt-swap-evaluators", nargs="+", type=int, default=())
    args = parser.parse_args()
    if len(args.adopt_swap_evaluators) > len(args.gpus):
        parser.error("One GPU is required for each adopted evaluator")
    args.root, args.rmbench = args.root.resolve(), args.rmbench.resolve()
    if not 1 <= len(args.gpus) <= 4 or len(set(args.gpus)) != len(args.gpus):
        parser.error("Select one to four distinct GPU UUIDs")
    if args.adopt_swap_trainers and len(args.gpus) < 2:
        parser.error("Existing trainers require their two original GPU UUIDs first")
    if not read_json(args.root / "audit/put_back_pass.json")["passed"]:
        raise RuntimeError("Put Back Block audit must pass before extending")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    with (args.root / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reproduction = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                        "--rmbench", str(args.rmbench), "--gpus", *args.gpus]
        if args.swap_only:
            reproduction.append("--swap-only")
        command = shlex.join(reproduction)
        (args.root / "summary/reproduce.txt").write_text(
            f"export PYTHONNOUSERSITE=1\nexport PYTHONPATH={shlex.quote(os.environ['PYTHONPATH'])}\n{command}\n"
        )
        if args.swap_only:
            await run_task(args, "swap_blocks", 2)
            return
        results = [await run_task(args, task, phase) for phase, task in enumerate(TASKS, 2)]
        update_plan(args.root, (4, "in_progress"), "开始整理三任务对比。")
        final_report(args.root, results)
        update_plan(args.root, (4, "complete"), "三任务最终报告已生成。")


if __name__ == "__main__":
    asyncio.run(main())
