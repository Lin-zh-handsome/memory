"""Train continuous Mamba M2 and evaluate on the original Put Back Block reset set."""

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from campaign import Campaign
from summarize import best_result, checkpoint_result, read_json


async def run(args):
    root = args.root
    campaign = Campaign(SimpleNamespace(root=root, data=args.data, rmbench=args.rmbench,
                                        camera_shader="rt", eval_batch_size=2, switch_at=None))
    reference = Path(read_json(root / "configs/source.json")["reference_experiment"])
    seeds = read_json(root / "evaluation/reset_seeds.json")
    if seeds != read_json(reference / "evaluation/reset_seeds.json"):
        raise RuntimeError("The reset manifest differs from the original M0/M3 experiment")
    for gpu in args.gpus:
        campaign.check_idle(gpu)
    pending = campaign.config["eval_epochs"].copy()
    training_done = asyncio.Event()
    expected_updates = 75000

    async def evaluate(gpu):
        while pending:
            ready = next((epoch for epoch in pending
                          if (root / f"checkpoints/M2/epoch_{epoch:03d}.pt").is_file()), None)
            if ready is None:
                if training_done.is_set():
                    raise RuntimeError("Training ended with missing checkpoints")
                await asyncio.sleep(15)
                continue
            pending.remove(ready)
            result = checkpoint_result(root / "evaluation", "M2", ready, seeds["seeds"], 50)
            if result is None:
                campaign.check_idle(gpu)
                name = f"M2_eval_{ready:03d}"
                child = await campaign.start(name, gpu, "evaluate.py", *campaign.evaluator_args(gpu),
                                             "rollout", "--model", "M2", "--epoch", ready,
                                             "--batch-size", 2)
                await campaign.finish(name, child)

    async def train():
        try:
            done = read_json(root / "logs/M2_complete.json")
            if done is None:
                await campaign.train("M2", args.gpus[0])
                done = read_json(root / "logs/M2_complete.json")
            if done != {"epoch": 250, "updates": expected_updates}:
                raise RuntimeError(f"Unexpected M2 training budget: {done}")
        finally:
            training_done.set()
        await evaluate(args.gpus[0])

    outcomes = await asyncio.gather(train(), evaluate(args.gpus[1]), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            raise outcome
    epochs = campaign.config["eval_epochs"]
    scores = {model: {} for model in ("M0", "M2", "M3")}
    for model in scores:
        for epoch in epochs:
            directory = root if model == "M2" else reference
            result = checkpoint_result(directory / "evaluation", model, epoch, seeds["seeds"], 50)
            if result is None:
                raise RuntimeError(f"Missing completed result for {model} epoch {epoch}")
            scores[model][epoch] = result
    best = {model: best_result(results, epochs) for model, results in scores.items()}
    lines = ["# RMBench 方块归位单种子记忆消融", "",
             "训练种子为 0，每个检查点使用相同的 50 个固定环境初始条件。", "",
             "| 训练轮次 | M0 | M2 | M3 |", "|---:|---:|---:|---:|"]
    for epoch in epochs:
        cells = [f"{s[epoch]['rate']:.0%}（{s[epoch]['successes']}/50）" for s in scores.values()]
        lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
    cells = [f"{x['rate']:.0%}（第 {x['epoch']} 轮）" for x in best.values()]
    lines.append("| 最佳 | " + " | ".join(cells) + " |")
    lines += ["", "M2 使用连续 Mamba（状态空间序列模型）与过去 8 步动作重构。"
              "M3 在相同历史模块上增加选择性情景记忆。"]
    (root / "evaluation/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "result.json").write_text(json.dumps({
        "model": "M2", "episodes": 250, "best_epoch": best["M2"]["epoch"],
        "epoch_successes": {epoch: row["successes"] for epoch, row in scores["M2"].items()},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "data", "rmbench"):
        parser.add_argument(f"--{option}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpus", nargs=2, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        parser.error("Use distinct training and evaluation GPUs")
    with (args.root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
