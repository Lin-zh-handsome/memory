"""Prepare task-specific data and fill the active RMBench experiment queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


CODE = Path(__file__).resolve().parent
TASK_PATHS = {
    "put_back_block": "rmbench_put_back_block_m0_m3_seed0",
    "swap_blocks": "rmbench_memory_validation/swap_blocks",
    "battery_try": "rmbench_mn_validation/battery_try",
    "observe_and_pickup": None,
    "rearrange_blocks": None,
    "swap_T": None,
    "blocks_ranking_try": None,
    "press_button": None,
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def command(script, *args):
    subprocess.run([sys.executable, "-u", str(CODE / script), *map(str, args)], check=True)


def plan_task(base, root, task, limits, template, queue, weights, rmbench):
    experiments = base / "experiments"
    old_prefix = TASK_PATHS[task]
    old = experiments / old_prefix if old_prefix else None
    if task == "put_back_block":
        data = base / "data/prepared"
        archive = base / "data/put_back_block_demo_clean.zip"
    elif old:
        data = old / "prepared"
        archive = old / f"archive/data/{task}/demo_clean.zip"
    else:
        data = root / f"datasets/{task}/prepared"
        archive = root / f"datasets/{task}.zip"
        if not archive.is_file():
            download_result = root / "logs/download_status.json"
            if download_result.is_file():
                status = json.loads(download_result.read_text())
                pid = status.get("pid")
                identity = Path(f"/proc/{pid}/stat")
                try:
                    fields = identity.read_text().rsplit(")", 1)[1].split()
                    active = fields[0] != "Z" and fields[19] == status["start"]
                except (FileNotFoundError, KeyError):
                    active = False
                if status["finished"] or not active:
                    raise FileNotFoundError(f"Download ended without {archive}; see downloads.log")
            return False
        if not (data / "manifest.json").is_file():
            command("data.py", archive, data, rmbench, "--task", task)

    task_root = root / task
    seed0_config = task_root / "seed0/configs/protocol.json"
    if seed0_config.exists():
        cfg = json.loads(seed0_config.read_text())
    else:
        cfg = dict(template)
        cfg["task"] = task
        cfg["max_episode_steps"] = limits[task]
        cfg["shared_reset_seeds"] = str(task_root / "reset_seeds.json")
        cfg["models"] = ["M0", "mamba_only", "memory_only", "M5"]
        if old:
            prev = json.loads((old / "configs/protocol.json").read_text())
            cfg["event_thresholds"] = prev["event_thresholds"]
            cfg["eval_seed_start"] = prev["eval_seed_start"]
            manifest = json.loads((old / "evaluation/reset_seeds.json").read_text())
            if not manifest["complete"] or len(manifest["seeds"]) != 50:
                raise ValueError(f"Incomplete historical reset list for {task}")
            write(task_root / "reset_seeds.json", manifest)
        else:
            cfg.pop("event_thresholds")
        write(seed0_config, cfg)
    if "event_thresholds" not in cfg:
        command("calibrate.py", "--root", seed0_config.parents[1], "--data", data, "--device", "cpu")
        cfg = json.loads(seed0_config.read_text())

    known = {(item["root"], item["model"]) for item in queue["jobs"]}
    targets = task_root / "annotations"
    for seed in (0, 1, 2):
        job_root = task_root / f"seed{seed}"
        protocol = job_root / "configs/protocol.json"
        if not protocol.exists():
            write(protocol, {**cfg, "seed": seed})
        for model in ("M0", "mamba_only", "memory_only", "M5"):
            if model == "M0" and seed == 0 and old:
                continue
            if model == "M5" and not (targets / "complete.json").is_file():
                continue
            job_id = (str(job_root), model)
            if job_id in known:
                continue
            queue["jobs"].append(dict(root=str(job_root), data=str(data), model=model,
                                      **({"targets": str(targets)} if model == "M5" else {})))
    write(root / "queue.json", queue)
    if not (targets / "complete.json").is_file():
        command("ablation_targets.py", "--archive", archive, "--data", data,
                "--weights", weights, "--output", targets, "--rmbench", rmbench)
        return plan_task(base, root, task, limits, template, queue, weights, rmbench)
    print(json.dumps({"task_ready": task, "queued_jobs": len(queue["jobs"])}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    root = base / "experiments/rmbench/m5_ablation_seeds"
    reference = root.parent / "m5_goal_spatial_r3m"
    template = json.loads((reference / "configs/protocol.json").read_text())
    rmbench = base / "source/RMBench"
    limits = yaml.safe_load((rmbench / "env_cfg/task_config/_eval_step_limit.yml").read_text())
    queue = json.loads((root / "queue.json").read_text())
    pending = list(TASK_PATHS)
    failures = {}
    queue["preparation_finished"] = False
    queue["preparation_errors"] = {}
    while pending:
        for task in pending[:]:
            try:
                if not plan_task(base, root, task, limits, template, queue,
                                 reference / "teacher/pytorch_model.bin", rmbench):
                    continue
            except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
                print(json.dumps({"task_preparation_failed": task, "error": str(error)}), flush=True)
                failures[task] = str(error)
            pending.remove(task)
        if pending:
            time.sleep(60)
    queue["preparation_finished"] = True
    queue["preparation_errors"] = failures
    queue["fully_prepared"] = not failures and len(queue["jobs"]) == 103
    write(root / "queue.json", queue)
    if not queue["fully_prepared"]:
        raise SystemExit(f"Unprepared tasks: {sorted(failures)}. Queued: {len(queue['jobs'])}")


if __name__ == "__main__":
    main()
