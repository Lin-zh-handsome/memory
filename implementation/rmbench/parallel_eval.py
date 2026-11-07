"""Run independent formal evaluations on idle GPUs before the resource cutoff."""

import argparse
import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import signal

from campaign import Campaign, record


def complete_file(path, predicate):
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return predicate(data)


class ParallelEvaluation(Campaign):
    def __init__(self, args):
        super().__init__(args)
        self.pause_time = (args.switch_at - timedelta(seconds=args.drain_seconds)) if args.switch_at else None

    def can_start(self):
        return self.pause_time is None or datetime.now().timestamp() < self.pause_time.timestamp()

    async def wait_for_dependencies(self):
        markers = [self.logs / f"{model}_complete.json" for model in self.config["models"]]
        resets = self.args.root / "evaluation" / "reset_seeds.json"
        while self.can_start():
            if (
                (self.args.during_training or all(complete_file(path, lambda data: data["epoch"] == self.config["epochs"]) for path in markers))
                and complete_file(resets, lambda data: data["complete"])
            ):
                return True
            await asyncio.sleep(15)
        return False

    async def run_one(self, gpu, model, epoch):
        pci = self.pcis[gpu].split(":")[-2]
        name = f"{model}_eval_{epoch:03d}_pci_{pci}"
        timing = ["--stop-at", self.pause_time.isoformat()] if self.pause_time else []
        child = await self.start(
            name, gpu, "evaluate.py", *self.evaluator_args(gpu), "rollout",
            "--model", model, "--epoch", epoch, "--batch-size", self.args.eval_batch_size,
            *timing,
        )
        if self.cutover is None:
            return await self.finish(name, child)
        try:
            code = await asyncio.wait_for(
                child.wait(), timeout=max(0, self.cutover - 1 - datetime.now().timestamp()),
            )
        except asyncio.TimeoutError:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = await child.wait()
        record("evaluation_worker_exit", name=name, gpu=gpu, code=code)
        if code not in (0, 75) and datetime.now().timestamp() < self.pause_time.timestamp():
            raise RuntimeError(f"{name} exited with code {code}")

    async def worker(self, gpu, jobs):
        while jobs and self.can_start():
            if self.args.during_training:
                ready = next((
                    index for index, (model, epoch) in enumerate(jobs)
                    if (self.args.root / "checkpoints" / model / f"epoch_{epoch:03d}.pt").is_file()
                ), None)
                if ready is None:
                    await asyncio.sleep(15)
                    continue
                model, epoch = jobs.pop(ready)
            else:
                model, epoch = jobs.pop()
            summary = self.args.root / "evaluation" / model / f"epoch_{epoch:03d}" / "summary.json"
            if complete_file(summary, lambda data: data["complete"]):
                continue
            try:
                self.check_idle(gpu)
            except RuntimeError as error:
                record("evaluation_gpu_skipped", gpu=gpu, reason=str(error))
                return
            await self.run_one(gpu, model, epoch)

    async def run(self):
        if not await self.wait_for_dependencies():
            record("no_parallel_evaluation_window")
            return
        # The main campaign visits in ascending order. Start from the opposite
        # end so workers tackle distinct checkpoints while both make progress.
        epochs = self.config["eval_epochs"]
        if not self.args.during_training:
            epochs = epochs[1:]
        jobs = [
            (model, epoch)
            for epoch in epochs
            for model in self.config["models"]
        ]
        workers = []
        for gpu in self.args.gpus:
            try:
                self.check_idle(gpu)
            except RuntimeError as error:
                record("evaluation_gpu_skipped", gpu=gpu, reason=str(error))
                continue
            workers.append(self.worker(gpu, jobs))
        results = await asyncio.gather(*workers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                record("parallel_evaluation_worker_failed", reason=str(result))
        record("parallel_evaluation_finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--switch-at", type=datetime.fromisoformat)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--during-training", action="store_true",
                        help="Evaluate each fully saved checkpoint as soon as its seeds are ready")
    parser.add_argument("--camera-shader", default="rt")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--drain-seconds", type=int, default=150)
    args = parser.parse_args()
    if ((args.switch_at and args.switch_at.utcoffset() is None)
            or len(args.gpus) > 5 or len(args.gpus) != len(set(args.gpus))):
        parser.error("Use at most five distinct auxiliary GPUs and include a timezone if specifying a cutoff")
    for name in ("root", "data", "rmbench"):
        setattr(args, name, getattr(args, name).resolve())
    asyncio.run(ParallelEvaluation(args).run())


if __name__ == "__main__":
    main()
