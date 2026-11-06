"""Run the fixed RMBench campaign independently of the interactive session."""

import argparse
import asyncio
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


CODE = Path(__file__).resolve().parent


def record(event, **fields):
    print(json.dumps({"time": datetime.now().astimezone().isoformat(), "event": event, **fields}), flush=True)


class Campaign:
    def __init__(self, args):
        self.args = args
        self.logs = args.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((args.root / "configs" / "protocol.json").read_text())
        self.cutover = args.switch_at.timestamp() if args.switch_at else None
        self.pcis = {}

    def check_idle(self, gpu):
        query = subprocess.check_output([
            "nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True).strip()
        pci, memory = query.split(",")
        self.pcis[gpu] = pci.strip()
        if int(memory) > 128:
            raise RuntimeError(f"{gpu} is no longer idle ({memory.strip()} MiB allocated)")

    async def start(self, name, gpu, script, *arguments):
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUDA_VISIBLE_DEVICES": gpu}
        argv = [sys.executable, "-u", str(CODE / script), *map(str, arguments)]
        with (self.logs / f"{name}.log").open("ab") as log:
            child = await asyncio.create_subprocess_exec(
                *argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True,
            )
        record("started", name=name, gpu=gpu, pid=child.pid)
        return child

    async def finish(self, name, child):
        code = await child.wait()
        record("exited", name=name, pid=child.pid, code=code)
        if code:
            raise RuntimeError(f"{name} exited with {code}. See {self.logs / (name + '.log')}")

    async def sleep_until(self, timestamp):
        await asyncio.sleep(max(0, timestamp - datetime.now().timestamp()))

    async def train(self, model, gpu, migrate=False):
        command = ("--root", self.args.root, "--data", self.args.data, "--model", model)
        child = await self.start(f"{model}_train", gpu, "train.py", *command)
        if not migrate:
            return await self.finish(f"{model}_train", child)

        # Initiate an atomic checkpoint shortly before 05:00 so the auxiliary
        # GPU is released by the deadline even during a large checkpoint write.
        remaining = max(0, self.cutover - self.args.migration_grace - datetime.now().timestamp())
        forced = False
        try:
            code = await asyncio.wait_for(child.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            if child.returncode is None:
                record("migration_checkpoint_requested", model=model, gpu=gpu, pid=child.pid)
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
            try:
                code = await asyncio.wait_for(
                    child.wait(), timeout=max(0, self.cutover - datetime.now().timestamp())
                )
            except asyncio.TimeoutError:
                if child.returncode is None:
                    record("migration_grace_expired", model=model, pid=child.pid)
                    forced = True
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass
                code = await child.wait()
        record("aux_training_exited", model=model, gpu=gpu, pid=child.pid, code=code)
        if code == 0:
            return
        if code != 75 and not (forced and code == -9):
            raise RuntimeError(f"{model} failed before migration. See {self.logs / (model + '_train.log')}")
        await self.sleep_until(self.cutover)
        record("resuming_on_primary", model=model, gpu=self.args.primary)
        child = await self.start(f"{model}_train", self.args.primary, "train.py", *command)
        await self.finish(f"{model}_train", child)

    def evaluator_args(self, gpu):
        return (
            "--experiment", self.args.root, "--rmbench", self.args.rmbench, "--data", self.args.data,
            "--render-pci", self.pcis[gpu], "--camera-shader", self.args.camera_shader,
        )

    async def prepare(self, gpu):
        reset_list = self.args.root / "evaluation/reset_seeds.json"
        if reset_list.exists() and json.loads(reset_list.read_text())["complete"]:
            return
        name = "prepare_seeds"
        child = await self.start(name, gpu, "evaluate.py", *self.evaluator_args(gpu), "prepare-seeds")
        await self.finish(name, child)

    async def seed_shard(self, gpu, start, stop):
        name = f"prepare_seeds_{start}_{stop}"
        child = await self.start(
            name, gpu, "evaluate.py", *self.evaluator_args(gpu), "prepare-seeds",
            "--candidate-start", start, "--candidate-stop", stop,
        )
        try:
            await asyncio.wait_for(
                self.finish(name, child),
                timeout=max(0, self.cutover - 5 - datetime.now().timestamp()),
            )
        except asyncio.TimeoutError:
            record("seed_shard_cutover", name=name, gpu=gpu, pid=child.pid)
            for stop_signal in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, stop_signal)
                except ProcessLookupError:
                    break
                if stop_signal == signal.SIGTERM:
                    await asyncio.sleep(1)
            await child.wait()

    async def prepare_on_idle_cards(self, gpus):
        reset_list = self.args.root / "evaluation/reset_seeds.json"
        if reset_list.exists() and json.loads(reset_list.read_text())["complete"]:
            return
        start = self.config["eval_seed_start"]
        shard_size = 16
        shards = []
        for gpu in gpus:
            try:
                self.check_idle(gpu)
            except RuntimeError as error:
                record("seed_gpu_skipped", gpu=gpu, reason=str(error))
                continue
            shards.append(self.seed_shard(gpu, start, start + shard_size))
            start += shard_size
        if not shards:
            return
        results = await asyncio.gather(*shards, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                record("seed_shard_failed", reason=str(result))
        # The merger accepts only the complete, contiguous expert-checked prefix.
        # Missing candidates can then resume on the primary card.
        child = await self.start(
            "merge_seeds", self.args.primary, "evaluate.py", *self.evaluator_args(self.args.primary),
            "merge-seeds",
        )
        await self.finish("merge_seeds", child)

    async def evaluate(self, gpu, model, epoch):
        name = f"{model}_eval_{epoch:03d}"
        child = await self.start(
            name, gpu, "evaluate.py", *self.evaluator_args(gpu), "rollout",
            "--model", model, "--epoch", epoch, "--batch-size", self.args.eval_batch_size,
        )
        await self.finish(name, child)
        subprocess.run([sys.executable, str(CODE / "summarize.py"), "--root", str(self.args.root)], check=True)

    async def run(self):
        primary = self.args.primary
        auxiliary = self.args.auxiliary
        separate = auxiliary is not None and (
            datetime.now().timestamp() < self.cutover - self.args.migration_grace
        )
        self.check_idle(primary)
        if separate:
            self.check_idle(auxiliary)
        jobs = [
            self.train("M0", auxiliary if separate else primary, migrate=separate),
            self.train("M3", primary),
        ]
        if self.args.seed_gpus and datetime.now().timestamp() < self.cutover - 60:
            jobs.append(self.prepare_on_idle_cards(self.args.seed_gpus))
        results = await asyncio.gather(*jobs, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result
        await self.prepare(primary)
        for epoch in self.config["eval_epochs"]:
            for model in self.config["models"]:
                await self.evaluate(primary, model, epoch)
        record("campaign_complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--auxiliary")
    parser.add_argument("--seed-gpus", nargs="*", default=[])
    parser.add_argument("--switch-at", type=datetime.fromisoformat)
    parser.add_argument("--migration-grace", type=int, default=90)
    parser.add_argument("--camera-shader", default="rt")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    args = parser.parse_args()
    gpus = [args.primary, *([args.auxiliary] if args.auxiliary else []), *args.seed_gpus]
    if len(gpus) != len(set(gpus)) or len(gpus) > 6:
        parser.error("Provide at most six distinct physical GPUs")
    if len(gpus) > 1 and (args.switch_at is None or args.switch_at.utcoffset() is None):
        parser.error("Auxiliary GPUs require a time zone in --switch-at")
    for name in ("root", "data", "rmbench"):
        setattr(args, name, getattr(args, name).resolve())
    with (args.root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(Campaign(args).run())


if __name__ == "__main__":
    main()
