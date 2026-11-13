"""Train M4 and evaluate each checkpoint on one available physical GPU."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CODE = Path(__file__).resolve().parent
POLL_SECONDS = 60
EPOCHS = (50, 100, 150, 200, 250)


def record(event, **fields):
    print(json.dumps({"time": datetime.now().astimezone().isoformat(), "event": event, **fields}),
          flush=True)


def nvidia_query(kind, columns):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={columns}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()


@dataclass(frozen=True)
class Card:
    index: int
    uuid: str
    pci: str
    allocated_mib: int
    free_mib: int


def cards():
    devices = []
    for line in nvidia_query("gpu", "index,uuid,pci.bus_id,memory.used,memory.free"):
        index, uuid, pci, used, free = map(str.strip, line.split(","))
        devices.append(Card(int(index), uuid, pci, int(used), int(free)))
    return devices


def compute_processes():
    busy = {}
    for line in nvidia_query("compute-apps", "gpu_uuid,pid"):
        uuid, pid = map(str.strip, line.split(","))
        busy.setdefault(uuid, set()).add(int(pid))
    return busy


def process_token(pid):
    if pid is None:
        return None
    try:
        # After the final parenthesis, field 3 is first and field 22 is starttime.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[19] if fields[0] != "Z" else None


def pci_address(value):
    return ":".join(value.strip().lower().split(":")[-2:])


def choose_card(args):
    dependency = process_token(args.wait_for_pid)
    announced = False
    while True:
        devices = cards()
        valid = {device.uuid for device in devices}
        if args.gpu != "auto" and args.gpu not in valid:
            raise ValueError(f"GPU UUID {args.gpu} was not found")
        if args.reserved_gpu and args.reserved_gpu not in valid:
            raise ValueError(f"Reserved GPU UUID {args.reserved_gpu} was not found")
        active_dependency = dependency is not None and process_token(args.wait_for_pid) == dependency
        busy = compute_processes()
        for device in devices:
            if args.gpu != "auto" and device.uuid != args.gpu:
                continue
            if args.render_pci and pci_address(args.render_pci) != pci_address(device.pci):
                raise ValueError(f"GPU UUID {device.uuid} uses PCI {device.pci}, not {args.render_pci}")
            if active_dependency and device.uuid == args.reserved_gpu:
                continue
            if not busy.get(device.uuid) and device.allocated_mib <= 128:
                record("gpu_selected", gpu_index=device.index, gpu_uuid=device.uuid, pci=device.pci)
                return device
        if not announced:
            record("waiting_for_idle_gpu", gpu=args.gpu, reserved_gpu=args.reserved_gpu,
                   wait_for_pid=args.wait_for_pid if active_dependency else None)
            announced = True
        time.sleep(POLL_SECONDS)


def can_start_evaluation(card, trainer):
    allowed_group = trainer.pid if trainer is not None and trainer.poll() is None else None
    for pid in compute_processes().get(card.uuid, ()):
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group != allowed_group:
            return False
    # Same available-memory floor as the current one-GPU RMBench campaign.
    return next(device.free_mib for device in cards() if device.uuid == card.uuid) >= 24000


def train_finished(root, config):
    marker = root / "logs/M4_complete.json"
    if not marker.is_file():
        return False
    if json.loads(marker.read_text())["epoch"] != config["epochs"]:
        raise ValueError(f"Training completion marker has a different epoch: {marker}")
    return True


def eval_finished(root, config, epoch):
    summary_path = root / f"evaluation/M4/epoch_{epoch:03d}/summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    return (summary["complete"] and summary["model"] == "M4" and summary["epoch"] == epoch
            and summary["episodes"] == config["eval_episodes"])


def run_job(script, args, log, card, lock):
    command = [sys.executable, "-u", str(CODE / script), *map(str, args)]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": card.uuid}
    with log.open("ab") as output:
        job = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                               env=env, start_new_session=True, pass_fds=(lock.fileno(),))
    record("started", job=log.stem, pid=job.pid, gpu_index=card.index, log=str(log))
    return job


def stop_own_job(job, group=False):
    if job is None or job.poll() is not None:
        return
    try:
        if group:
            os.killpg(job.pid, signal.SIGTERM)
        else:
            job.terminate()
    except ProcessLookupError:
        pass
    job.wait()


def run(args, lock, config):
    root = args.root
    logs = root / "logs"
    checkpoints = root / "checkpoints/M4"
    train_done = train_finished(root, config)
    pending = [epoch for epoch in EPOCHS if not eval_finished(root, config, epoch)]
    if train_done and not pending:
        record("complete", model="M4", checkpoints=list(EPOCHS))
        return
    if train_done and any(not (checkpoints / f"epoch_{epoch:03d}.pt").is_file() for epoch in pending):
        raise RuntimeError("A checkpoint is missing for a pending evaluation")
    if not train_done and not (checkpoints / "latest.pt").is_file():
        if (logs / "M4_epochs.csv").exists() or any(checkpoints.glob("epoch_*.pt")):
            raise RuntimeError("Training artifacts exist without a valid resume entry point")

    card = choose_card(args)
    trainer = evaluator = None
    evaluating = None
    eval_error = None
    try:
        if not train_done:
            trainer = run_job("train.py", ("--root", root, "--data", args.data, "--model", "M4"),
                              logs / "M4_train.log", card, lock)
        while trainer is not None or pending or evaluator is not None:
            if trainer is not None and (code := trainer.poll()) is not None:
                record("exited", job="M4_train", code=code)
                trainer = None
                if code or not train_finished(root, config):
                    raise RuntimeError(f"M4 training did not complete (exit {code}); see M4_train.log")
                train_done = True

            if evaluator is not None and (code := evaluator.poll()) is not None:
                record("exited", job=f"M4_eval_{evaluating:03d}", code=code)
                evaluator = None
                if code or not eval_finished(root, config, evaluating):
                    eval_error = f"M4 epoch {evaluating} evaluation incomplete; see M4_eval_{evaluating:03d}.log"
                    record("evaluation_failed", detail=eval_error)
                    pending.clear()

            if evaluator is None and pending:
                epoch = pending[0]
                if (checkpoints / f"epoch_{epoch:03d}.pt").is_file():
                    if can_start_evaluation(card, trainer):
                        evaluator = run_job(
                            "evaluate.py",
                            ("--experiment", root, "--rmbench", args.rmbench, "--data", args.data,
                             "--render-pci", card.pci, "--camera-shader", "rt", "rollout",
                             "--model", "M4", "--epoch", epoch, "--batch-size", 2),
                            logs / f"M4_eval_{epoch:03d}.log", card, lock,
                        )
                        evaluating = pending.pop(0)
                elif train_done:
                    raise RuntimeError(f"M4 checkpoint {epoch} is missing")

            if trainer is not None or pending or evaluator is not None:
                time.sleep(POLL_SECONDS)

        if eval_error:
            raise RuntimeError(eval_error)
        record("complete", model="M4", checkpoints=list(EPOCHS))
    finally:
        stop_own_job(evaluator, group=True)
        stop_own_job(trainer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--gpu", default="auto", help="auto or a physical GPU UUID")
    parser.add_argument("--render-pci", help="optional PCI check for a fixed GPU")
    parser.add_argument("--reserved-gpu", help="GPU UUID reserved while --wait-for-pid is active")
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if bool(args.reserved_gpu) != (args.wait_for_pid is not None):
        parser.error("--reserved-gpu and --wait-for-pid must be supplied together")
    if args.wait_for_pid is not None and args.wait_for_pid <= 0:
        parser.error("--wait-for-pid must be positive")
    if args.gpu == "auto" and args.render_pci:
        parser.error("--render-pci is only valid with a fixed GPU UUID")
    args.root, args.data, args.rmbench = (p.resolve() for p in (args.root, args.data, args.rmbench))
    config = json.loads((args.root / "configs/protocol.json").read_text())
    if config["seed"] != 0 or config["epochs"] != 250 or tuple(config["eval_epochs"]) != EPOCHS:
        raise ValueError("Expected the matched seed-0, 250-epoch protocol")
    seeds = json.loads((args.root / "evaluation/reset_seeds.json").read_text())
    if (not seeds["complete"] or config["eval_episodes"] != 50 or len(set(seeds["seeds"])) != 50
            or len(seeds["seeds"]) != 50 or seeds["task"] != config["task"]
            or seeds["task_config"] != config["task_config"]):
        raise ValueError("Expected the existing complete task-specific reset seed manifest")
    if not (args.data / "manifest.json").is_file() or not (args.rmbench / "env_cfg").is_dir():
        raise FileNotFoundError("Prepared demonstrations or RMBench checkout are missing")

    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "M4_campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"M4 campaign already active under {args.root}") from error

        def stop(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        record("queued", model="M4", pid=os.getpid(), root=str(args.root))
        run(args, lock, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        record("stopped", model="M4")
        raise SystemExit(130)
