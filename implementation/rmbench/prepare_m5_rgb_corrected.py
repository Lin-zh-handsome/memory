"""Prepare corrected RGB training for unfinished M5 seed-0 RMBench tasks."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from report_ablation import EPOCHS, TASKS, REUSED


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluated(root):
    for epoch in EPOCHS:
        path = root / f"evaluation/M5/epoch_{epoch:03d}/summary.json"
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        if not (data.get("complete") and data.get("episodes") == 50
                and data.get("model") == "M5" and data.get("epoch") == epoch):
            return False
    return True


def prepare(base, output):
    original = base / "experiments/rmbench/m5_ablation_seeds"
    jobs = json.loads((original / "queue.json").read_text())["jobs"]
    previous = {Path(job["root"]).parent.name: job for job in jobs
                if job["model"] == "M5" and Path(job["root"]).name == "seed0"}
    source_file = output / "result_sources.json"
    if source_file.exists():
        sources = json.loads(source_file.read_text())
    else:
        sources = {"created": datetime.now().astimezone().isoformat(),
                   "source_campaign": str(original), "tasks": []}
        for task in TASKS:
            reused = REUSED.get((task, "M5"))
            old = base / "experiments" / reused if reused else Path(previous[task]["root"])
            complete = evaluated(old)
            sources["tasks"].append({
                "task": task, "root": str(old if complete else output / task / "seed0"),
                "previous_root": str(old), "rgb_normalization": "range" if complete else "identity",
                "reused_complete": complete})
        write(source_file, sources)
    queue = {"jobs": [], "fully_prepared": False, "preparation_finished": False}
    write(output / "queue.json", queue)
    (output / "logs").mkdir(exist_ok=True)
    for item in sources["tasks"]:
        if item["reused_complete"]:
            continue
        task = item["task"]
        source = previous[task]
        old = Path(item["previous_root"])
        root = Path(item["root"])
        protocol_path = root / "configs/protocol.json"
        (root / "logs").mkdir(parents=True, exist_ok=True)
        if protocol_path.exists():
            config = json.loads(protocol_path.read_text())
            assert config["vision"]["rgb_normalization"] == "identity"
            assert config["task"] == task and config["seed"] == 0
        else:
            config = json.loads((old / "configs/protocol.json").read_text())
            config["vision"]["rgb_normalization"] = "identity"
            config["preprocessing_revision"] = "rgb_identity_v1"
            config["models"] = ["M5"]
            config.pop("event_thresholds", None)
            original_resets = Path(config.get("shared_reset_seeds", old / "evaluation/reset_seeds.json"))
            seeds = json.loads(original_resets.read_text())
            if (seeds["task"] != task or len(seeds["seeds"]) > 50
                    or len(set(seeds["seeds"])) != len(seeds["seeds"])
                    or seeds["complete"] and len(seeds["seeds"]) != 50):
                raise ValueError(f"Unexpected original reset manifest for {task}")
            # The official evaluator completes an expert-qualified prefix before scoring.
            new_resets = output / task / "reset_seeds.json"
            write(new_resets, seeds)
            config["shared_reset_seeds"] = str(new_resets)
            write(protocol_path, config)
        if "event_thresholds" not in config:
            log = root / "logs/calibration.log"
            with log.open("ab") as handle:
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("calibrate.py")),
                                "--root", str(root), "--data", source["data"], "--device", "cpu"],
                               check=True, stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
        queue["jobs"].append({**source, "root": str(root)})
        write(output / "queue.json", queue)
        print(json.dumps({"task_ready": task, "ready_jobs": len(queue["jobs"])}), flush=True)
    queue.update(fully_prepared=True, preparation_finished=True)
    write(output / "queue.json", queue)
    print(json.dumps({"preparation_complete": True, "new_training_jobs": len(queue["jobs"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(args.base.resolve(), args.output.resolve())
        except Exception as error:
            path = args.output / "queue.json"
            if path.exists():
                queue = json.loads(path.read_text())
                queue.update(fully_prepared=False, preparation_finished=True,
                             preparation_errors={"prepare": str(error)})
                write(path, queue)
            raise


if __name__ == "__main__":
    main()

