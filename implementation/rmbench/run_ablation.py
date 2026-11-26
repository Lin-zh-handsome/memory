"""Run the RMBench ablation queue on at most two explicitly selected GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from ablation_eval_shards import RANGES, merge, shard_dir
from run_m4 import cards, compute_processes, record, run_job, stop_own_job


def complete(path, **expected):
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return all(data.get(key) == value for key, value in expected.items())


def train_marker(spec):
    return spec.root / f"logs/{spec.model}_complete.json"


def eval_marker(spec, epoch):
    return spec.root / f"evaluation/{spec.model}/epoch_{epoch:03d}/summary.json"


def evaluation_done(spec, epoch, span):
    if span is None:
        return complete(eval_marker(spec, epoch), complete=True, episodes=50,
                        model=spec.model, epoch=epoch)
    first, stop = span
    path = shard_dir(eval_marker(spec, epoch).parent, span)
    return complete(path / "summary.json", complete=True, episodes=stop - first,
                    model=spec.model, epoch=epoch)


def jobs(specs, split):
    for spec in specs:
        if not complete(train_marker(spec), epoch=250):
            yield ("train", spec, 0, None)
    # Fill the first range across checkpoints before launching their second ranges.
    for span in RANGES if split else (None,):
        for spec in specs:
            for epoch in (50, 100, 150, 200, 250):
                marker = eval_marker(spec, epoch)
                if evaluation_done(spec, epoch, None):
                    continue
                checkpoint = spec.root / f"checkpoints/{spec.model}/epoch_{epoch:03d}.pt"
                if not checkpoint.is_file():
                    if complete(train_marker(spec), epoch=250):
                        raise FileNotFoundError(
                            f"Training finished but pending evaluation checkpoint is missing: {checkpoint}"
                        )
                    continue
                if split:
                    prefix = merge(marker.parent, spec, epoch)
                    if prefix >= span[1] or evaluation_done(spec, epoch, span):
                        continue
                yield ("eval", spec, epoch, span)


def choose_job(available, kind):
    for candidate in available:
        job_type, spec, epoch, _ = candidate
        if job_type == kind and (kind == "train"
                                or (spec.root / f"checkpoints/{spec.model}/epoch_{epoch:03d}.pt").is_file()):
            return candidate


def card_is_available(card, running, busy):
    groups = {child.pid for slot, (_, child) in running.items() if slot[0] == card.uuid}
    if not groups:
        return not busy and card.allocated_mib <= 128
    for pid in busy:
        try:
            if os.getpgid(pid) not in groups:
                return False
        except ProcessLookupError:
            continue
    return True


def launch(candidate, card, lock, rmbench):
    kind, spec, epoch, span = candidate
    logs = spec.root / "logs"
    logs.mkdir(exist_ok=True)
    if kind == "train":
        args = ["--root", spec.root, "--data", spec.data, "--model", spec.model]
        if spec.model == "M5":
            args.extend(("--goal-targets", spec.targets))
        return run_job("train.py", args, logs / f"{spec.model}_train.log", card, lock)
    args = ("--experiment", spec.root, "--rmbench", rmbench, "--data", spec.data,
            "--render-pci", card.pci, "--camera-shader", "rt", "rollout",
            "--model", spec.model, "--epoch", epoch, "--batch-size", 2)
    suffix = ""
    if span:
        args = (*args, "--episode-range", *span)
        suffix = f"_{span[0]:02d}_{span[1]:02d}"
    return run_job("evaluate.py", args, logs / f"{spec.model}_eval_{epoch:03d}{suffix}.log", card, lock)


def priority(spec):
    seed = int(spec.root.name.removeprefix("seed"))
    # Within each seed, run baseline/full comparisons before inner ablations.
    resume = (spec.root / f"checkpoints/{spec.model}/latest.pt").is_file()
    return not resume, seed, spec.model not in ("M0", "M5")


def record_status(queue_path, state, **details):
    target = queue_path.parent / "logs/campaign_status.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"status": state, **details}, indent=2) + "\n")
    tmp.replace(target)


def run(args, lock):
    running = {}
    failures = {}
    record_status(args.queue, "running" if args.train_slots else "training_paused")
    try:
        while True:
            queue = json.loads(args.queue.read_text())
            specs = [SimpleNamespace(**{**job, "root": Path(job["root"])}) for job in queue["jobs"]]
            specs.sort(key=priority)
            devices = {card.uuid: card for card in cards() if card.uuid in args.gpu}
            busy = compute_processes()
            for slot, (candidate, child) in list(running.items()):
                status = child.poll()
                if status is None:
                    continue
                del running[slot]
                kind, spec, epoch, span = candidate
                finished = (complete(train_marker(spec), epoch=250) if kind == "train" else
                            evaluation_done(spec, epoch, span))
                key = (kind, spec.root, spec.model, epoch, span)
                if status or not finished:
                    failures[key] = failures.get(key, 0) + 1
                record("job_exit", kind=kind, experiment=str(spec.root), model=spec.model,
                       epoch=epoch, episode_range=span, code=status, complete=finished, attempts=failures.get(key, 0))
            first_model = queue.get("finish_model_first")
            first_specs = [spec for spec in specs if spec.model == first_model]
            pending = list(jobs(first_specs, args.split_evaluations))
            if not pending and not any(spec.model == first_model for (_, spec, _, _), _ in running.values()):
                pending = list(jobs(specs, args.split_evaluations))
            if not pending and not running and queue["fully_prepared"]:
                record_status(args.queue, "completed")
                record("campaign_complete")
                return
            active = {(t, s.root, s.model, e, span) for (t, s, e, span), _ in running.values()}
            available = [job for job in pending
                         if (key := (job[0], job[1].root, job[1].model, job[2], job[3])) not in active
                         and failures.get(key, 0) < 2 and (args.train_slots or job[0] != "train")]
            available.sort(key=lambda job: job[0] == "eval" and
                           not eval_marker(job[1], job[2]).is_file())
            preparation_finished = queue.get("preparation_finished", False) or queue["fully_prepared"]
            if (args.train_slots and preparation_finished and not running and not choose_job(available, "train")
                    and not choose_job(available, "eval")):
                exhausted = [{"kind": kind, "root": str(root), "model": model, "epoch": epoch,
                              "episode_range": span, "attempts": attempts} for (kind, root, model, epoch, span), attempts in failures.items()
                             if attempts >= 2]
                details = {"exhausted": exhausted, "pending": len(pending),
                           "preparation_errors": queue.get("preparation_errors", {})}
                record_status(args.queue, "blocked", **details)
                raise RuntimeError(f"No runnable jobs remain. {details}")
            for uuid in args.gpu:
                card = devices[uuid]
                if not card_is_available(card, running, busy.get(uuid, ())):
                    continue
                slots = [("train", i) for i in range(args.train_slots)]
                slots.extend(("eval", i) for i in range(args.eval_slots))
                remaining_mib = card.free_mib
                for kind, index in slots:
                    slot = (uuid, kind, index)
                    if slot in running:
                        continue
                    if remaining_mib < 24000:
                        break
                    candidate = choose_job(available, kind)
                    fill_idle = args.fill_idle_slots or (
                        kind == "eval" and args.fill_eval_slot_with_training
                    )
                    if candidate is None and fill_idle:
                        alternative = "eval" if kind == "train" else "train"
                        if alternative == "eval" or args.train_slots:
                            candidate = choose_job(available, alternative)
                    if candidate is None:
                        continue
                    running[slot] = candidate, launch(candidate, card, lock, args.rmbench)
                    available.remove(candidate)
                    # Account for startup allocations before nvidia-smi reflects them.
                    remaining_mib -= 12000
            time.sleep(60)
    except FileNotFoundError as error:
        record_status(args.queue, "blocked", reason="missing_artifact", error=str(error))
        raise
    finally:
        for (kind, _, _, _), child in running.values():
            stop_own_job(child, group=(kind == "eval"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--gpu", nargs="+", required=True)
    parser.add_argument("--train-slots", type=int, default=1)
    parser.add_argument("--eval-slots", type=int, default=1)
    parser.add_argument("--split-evaluations", action="store_true")
    parser.add_argument("--fill-eval-slot-with-training", action="store_true")
    parser.add_argument("--fill-idle-slots", action="store_true")
    args = parser.parse_args()
    if args.train_slots < 0 or args.eval_slots < 1:
        parser.error("Training slots must be nonnegative and evaluation slots positive")
    if not 1 <= len(set(args.gpu)) == len(args.gpu) <= 2:
        parser.error("Specify one or two distinct GPU UUIDs")

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    with args.queue.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, lock)


if __name__ == "__main__":
    main()
