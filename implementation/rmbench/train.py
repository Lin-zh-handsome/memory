"""Train the matched Stanford CNN-DP and continuous-memory RMBench policies."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import signal
import time

from diffusers.optimization import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
import numpy as np
import torch

from data import EpisodeDataset, observation_meta
from models import build_policy


def atomic_save(payload, path):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def configure_precision():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def cuda_observations(observations):
    return {
        key: (value.to("cuda", dtype=torch.float32).div_(255)
              if value.dtype == torch.uint8 else value.to("cuda"))
        for key, value in observations.items()
    }


def move_batch(batch, variant):
    obs = cuda_observations(batch["obs"])
    action = batch["action"].to("cuda")
    if variant == "M0":
        previous = cuda_observations(batch["previous_obs"])
        n = len(action)
        obs = {
            key: torch.stack((torch.cat((previous[key], value[:n - 1])), value[:n]), dim=1)
            for key, value in obs.items()
        }
        return {"obs": obs, "action": action}
    return {
        "obs": obs,
        "action": action,
        "anchors": batch["anchors"].to("cuda"),
        "past_action": batch["past_action"].to("cuda"),
        "past_mask": batch["past_mask"].to("cuda"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["M0", "M2", "M3", "M4"])
    args = parser.parse_args()
    root, variant = args.root, args.model
    uses_memory = variant != "M0"
    cfg = json.loads((root / "configs/protocol.json").read_text())
    if variant in ("M3", "M4") and "event_thresholds" not in cfg:
        raise RuntimeError("Set the event thresholds calibrated on training demonstrations first")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(3)
    configure_precision()

    dataset = EpisodeDataset(args.data)
    batch_size, horizon = cfg["batch_size"], cfg["horizon"]
    n_action_steps = cfg["n_action_steps"]
    steps_per_epoch = dataset.num_batches(batch_size, horizon, n_action_steps)
    samples_per_epoch = dataset.num_samples(horizon, n_action_steps)
    meta = observation_meta(cfg)
    policy = build_policy(variant, meta, cfg)
    policy.set_normalizer(dataset.normalizer())
    policy.cuda()
    ema_policy = copy.deepcopy(policy)
    ema_policy.requires_grad_(False)
    ema_opts = {k: v for k, v in cfg["ema"].items() if k != "enabled"}
    ema = EMAModel(ema_policy, **ema_opts)
    optimizer_opts = {k: v for k, v in cfg["optimizer"].items() if k != "name"}
    optimizer = torch.optim.AdamW(policy.parameters(), **optimizer_opts)
    scheduler = get_scheduler(cfg["lr_scheduler"]["name"], optimizer,
                              num_warmup_steps=cfg["lr_scheduler"]["warmup_steps"],
                              num_training_steps=steps_per_epoch * cfg["epochs"])

    ckdir = root / "checkpoints" / variant
    ckdir.mkdir(parents=True, exist_ok=True)
    latest = ckdir / "latest.pt"
    header = {"model_variant": variant, "seed": seed, "config": cfg, "cfg": cfg,
              "shape_meta": meta}
    (root / "logs").mkdir(exist_ok=True)
    log_path = root / "logs" / f"{variant}_epochs.csv"
    trainable = {name: param.numel() for name, param in policy.named_parameters() if param.requires_grad}
    parameter_counts = {
        "trainable_parameters": sum(trainable.values()),
        "inference_parameters": sum(size for name, size in trainable.items()
                                    if not name.startswith("past_head.")),
    }
    model_config = {**cfg, "model": variant, "shape_meta": meta, **parameter_counts,
                    "updates_per_epoch": steps_per_epoch, "samples_per_epoch": samples_per_epoch}
    (root / "configs" / f"{variant}.json").write_text(json.dumps(model_config, indent=2) + "\n")
    first_epoch, consumed_batches, global_step = 1, 0, 0
    totals = np.zeros(3)
    count = 0
    carry = None

    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["model_variant"] != variant:
            raise RuntimeError("Cannot resume with a different protocol or model")
        policy.load_state_dict(saved["model"])
        ema_policy.load_state_dict(saved["ema"])
        policy.normalizer.cuda()
        ema_policy.normalizer.cuda()
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        global_step = saved["global_step"]
        ema.optimization_step = global_step
        first_epoch = saved["next_epoch"]
        consumed_batches = saved["consumed_batches"]
        totals = saved["totals"]
        count = saved["count"]
        if saved.get("carry") is not None:
            carry = saved["carry"].to("cuda")
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        print(json.dumps({"resume_epoch": first_epoch, "batch": consumed_batches,
                          "global_step": global_step}), flush=True)
        del saved

    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_resume(epoch, batch_index, epoch_totals, n_samples, carry=None):
        save_start = time.monotonic()
        atomic_save({
            **header, "next_epoch": epoch, "consumed_batches": batch_index,
            "global_step": global_step, "totals": epoch_totals, "count": n_samples,
            "carry": None if carry is None else carry.to("cpu"),
            "model": policy.state_dict(), "ema": ema_policy.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        }, latest)
        print(json.dumps({"resume_saved": True, "next_epoch": epoch, "batch": batch_index,
                          "global_step": global_step, "checkpoint_seconds": time.monotonic() - save_start}),
              flush=True)

    print(json.dumps({"model": variant, "seed": seed, "updates_per_epoch": steps_per_epoch,
                      "samples_per_epoch": samples_per_epoch,
                      **parameter_counts}), flush=True)
    for epoch in range(first_epoch, cfg["epochs"] + 1):
        start = time.monotonic()
        policy.train()
        torch.cuda.reset_peak_memory_stats()
        for index, cpu_batch in enumerate(
            dataset.batches(batch_size, horizon, n_action_steps, epoch, seed), start=1
        ):
            if index <= consumed_batches:
                continue
            batch = move_batch(cpu_batch, variant)
            optimizer.zero_grad(set_to_none=True)
            if uses_memory:
                if cpu_batch["segment_start"] == 0:
                    carry = None
                elif carry is None:
                    raise RuntimeError("Cannot continue an episode without its TBPTT state")
                loss, carry = policy.compute_loss(batch, carry)
            else:
                loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} non-finite loss at epoch {epoch} batch {index}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.step(policy)
            global_step += 1
            if uses_memory:
                carry = None if cpu_batch["end_of_episode"] else carry.detach()
            parts = policy.loss_parts if uses_memory else (loss.detach(), 0.0)
            n = batch["action"].shape[0]
            totals += n * np.array([loss.item(), float(parts[0]), float(parts[1])])
            count += n
            if global_step <= 3 or global_step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "loss": loss.item(),
                                  "ddpm_loss": float(parts[0]), "past_loss": float(parts[1]),
                                  "lr": scheduler.get_last_lr()[0]}), flush=True)
            if stop and index < steps_per_epoch:
                save_resume(epoch, index, totals, count, carry)
                print(json.dumps({"paused_epoch": epoch, "batch": index, "global_step": global_step}),
                      flush=True)
                return 75

        if global_step != epoch * steps_per_epoch or count != samples_per_epoch or carry is not None:
            raise RuntimeError("Sampler updates, samples, or episode boundary disagree with the protocol")
        row = {"epoch": epoch, "global_step": global_step, "samples": count, "loss": totals[0] / count,
               "ddpm_loss": totals[1] / count, "past_loss": totals[2] / count,
               "seconds_this_session": time.monotonic() - start,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
        rows = []
        if log_path.exists():
            with log_path.open(newline="") as handle:
                rows = [r for r in csv.DictReader(handle) if int(r["epoch"]) < epoch]
        tmp = log_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(row))
            writer.writeheader()
            writer.writerows([*rows, row])
        os.replace(tmp, log_path)
        if epoch in cfg["eval_epochs"]:
            atomic_save({**header, "epoch": epoch, "ema": ema_policy.state_dict()},
                        ckdir / f"epoch_{epoch:03d}.pt")
        totals, count, consumed_batches = np.zeros(3), 0, 0
        checkpointed = (stop or epoch == 1 or epoch in cfg["eval_epochs"]
                        or epoch % cfg["resume_every_epochs"] == 0)
        if checkpointed:
            save_resume(epoch + 1, 0, totals, count)
        print(json.dumps({"epoch_complete": row}), flush=True)
        if stop and epoch < cfg["epochs"]:
            if not checkpointed:
                save_resume(epoch + 1, 0, totals, count)
            print(json.dumps({"paused_epoch": epoch + 1, "batch": 0, "global_step": global_step}),
                  flush=True)
            return 75

    (root / "logs" / f"{variant}_complete.json").write_text(
        json.dumps({"epoch": cfg["epochs"], "updates": global_step}) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
