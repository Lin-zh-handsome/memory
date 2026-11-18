"""Run M5 Cover Blocks with resumable training, fixed rollouts and a 40% seed gate."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from report_m5 import EPOCHS, best_epoch, read_results, report, stage_metrics
from run_m4 import can_start_evaluation, cards, choose_card, compute_processes, record, run_job, stop_own_job


def write_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def baseline_paths():
    rmbench = Path(__file__).resolve().parents[2] / "experiments/rmbench"
    return {0: rmbench / "mn_validation/m4_cover_blocks", 1: rmbench / "m4_cover_blocks_seed1"}


def prepare_seed(root, reference, seed):
    baseline = json.loads((reference / "configs/protocol.json").read_text())
    if (baseline["seed"] != seed or baseline["task"] != "cover_blocks"
            or baseline["task_config"] != "demo_clean" or baseline["epochs"] != 250
            or tuple(baseline["eval_epochs"]) != EPOCHS or baseline["eval_episodes"] != 50):
        raise ValueError(f"Unexpected M4 baseline protocol at {reference}")
    baseline.pop("past_steps")
    baseline.pop("past_loss_weight")
    baseline["models"] = ["M5"]
    baseline["goal_aux"] = {"weight": 0.02, "warmup_epochs": 10}
    baseline["resource_policy"] = {"max_gpus": 2, "routine_check_seconds": 3600}
    protocol = root / "configs/protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != baseline:
        raise ValueError(f"Refusing to replace changed M5 protocol: {protocol}")
    write_json(protocol, baseline)
    seeds = json.loads((reference / "evaluation/reset_seeds.json").read_text())
    if not seeds["complete"] or seeds["seeds"] != list(range(100000, 100050)):
        raise ValueError(f"Unexpected M4 reset seed list at {reference}")
    write_json(root / "evaluation/reset_seeds.json", seeds)


def eval_finished(root, epoch):
    path = root / f"evaluation/M5/epoch_{epoch:03d}/summary.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return (data["complete"] and data["model"] == "M5"
            and data["epoch"] == epoch and data["episodes"] == 50)


def available_evaluation_cards(primary, running, trainer, allow_second):
    taken = {card.uuid for card, _ in running.values()}
    busy = compute_processes()
    other = next((card for card in cards() if card.uuid != primary.uuid
                  and card.uuid not in taken and card.allocated_mib <= 128 and not busy.get(card.uuid)), None)
    # Use the training card first when it is free; avoid racing evaluation and training if
    # another card is free. While a second evaluation runs, do not acquire a third GPU.
    if trainer is None:
        if primary.uuid not in taken and can_start_evaluation(primary, None):
            yield primary
        if allow_second and not any(uuid != primary.uuid for uuid in taken) and other is not None:
            yield other
    elif allow_second and not any(uuid != primary.uuid for uuid in taken) and other is not None:
        yield other
    elif primary.uuid not in taken and not running and can_start_evaluation(primary, trainer):
        yield primary


def run_seed(args, root, seed, lock):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    pending = [epoch for epoch in EPOCHS if not eval_finished(root, epoch)]
    completion = logs / "M5_complete.json"
    if completion.exists() and json.loads(completion.read_text())["epoch"] != 250:
        raise ValueError(f"Unexpected completion marker at {completion}")
    if completion.exists() and not pending:
        return
    primary = choose_card(SimpleNamespace(gpu="auto", reserved_gpu=None, wait_for_pid=None, render_pci=None))
    trainer = None
    evaluating = {}
    try:
        if not completion.exists():
            trainer = run_job("train.py", ("--root", root, "--data", args.data, "--model", "M5",
                              "--goal-targets", args.root / "annotations"), logs / "M5_train.log", primary, lock)
        while trainer is not None or pending or evaluating:
            if trainer is not None and (code := trainer.poll()) is not None:
                trainer = None
                if code or not completion.exists():
                    raise RuntimeError(f"M5 seed {seed} training exited with code {code}")
                record("training_complete", seed=seed)
            for epoch, (_, job) in list(evaluating.items()):
                if (code := job.poll()) is not None:
                    del evaluating[epoch]
                    if code or not eval_finished(root, epoch):
                        raise RuntimeError(f"M5 seed {seed} epoch {epoch} evaluation exited with code {code}")
                    record("evaluation_complete", seed=seed, epoch=epoch)
            ready = [epoch for epoch in pending if (root / f"checkpoints/M5/epoch_{epoch:03d}.pt").is_file()]
            if trainer is None and len(ready) != len(pending):
                raise FileNotFoundError("A pending evaluation checkpoint was not produced")
            if ready:
                devices = available_evaluation_cards(primary, evaluating, trainer, args.two_gpus)
                for epoch, card in zip(ready, devices):
                    job = run_job(
                        "evaluate.py",
                        ("--experiment", root, "--rmbench", args.rmbench, "--data", args.data,
                         "--render-pci", card.pci, "--camera-shader", "rt", "rollout",
                         "--model", "M5", "--epoch", epoch, "--batch-size", 2),
                        logs / f"M5_eval_{epoch:03d}.log", card, lock,
                    )
                    evaluating[epoch] = card, job
                    pending.remove(epoch)
            if trainer is not None or pending or evaluating:
                time.sleep(60)
    finally:
        for _, job in evaluating.values():
            stop_own_job(job, group=True)
        stop_own_job(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("root", "data", "rmbench"):
        parser.add_argument(f"--{item}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--two-gpus", action="store_true", help="Allow one extra physical GPU for checkpoints")
    parser.add_argument("--publish", action="store_true", help="Publish the completed report and compact results")
    args = parser.parse_args()
    if not (args.root / "annotations/goal_features.npy").is_file():
        raise FileNotFoundError("Generate frozen teacher features before starting the campaign")
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "M5_campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)

        references = baseline_paths()
        seed_roots = {0: args.root}
        prepare_seed(args.root, references[0], 0)
        run_seed(args, args.root, 0, lock)
        metrics = {ep: stage_metrics(rows) for ep, rows in read_results(args.root, "M5").items()}
        best = best_epoch(metrics)
        gate = metrics[best]["success"] >= 20
        write_json(args.root / "seed_gate.json", {"best_epoch": best, "successes": metrics[best]["success"],
                   "episodes": 50, "run_seed_1": gate})
        if gate:
            seed_roots[1] = args.root / "seed1"
            prepare_seed(seed_roots[1], references[1], 1)
            run_seed(args, seed_roots[1], 1, lock)
        report(args.root, references, seed_roots)
        record("complete", seeds=list(seed_roots))
        if args.publish:
            subprocess.run([sys.executable, str(Path(__file__).with_name("publish_m5.py")), str(args.root)],
                           check=True)


if __name__ == "__main__":
    main()
