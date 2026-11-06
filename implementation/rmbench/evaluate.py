"""Official RMBench reset qualification and matched closed-loop DP evaluation."""

import argparse
from collections import deque
from datetime import datetime
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import yaml

IMPLEMENTATION = Path(__file__).resolve().parent
WORKSPACE = IMPLEMENTATION.parent
# Keep the upstream namespace package "data" distinct from our adapter.
# Curobo's spawned worker must keep the upstream checkout on its sys.path.
sys.path[:] = [path for path in sys.path if Path(path).resolve() != IMPLEMENTATION]
if str(WORKSPACE) not in sys.path:
    sys.path.append(str(WORKSPACE))
adapter = importlib.import_module(f"{IMPLEMENTATION.name}.data")
STATE_KEY = adapter.STATE_KEY


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pci_address(value):
    return ":".join(value.strip().lower().split(":")[-2:])


def make_simulator(args, protocol):
    os.chdir(args.rmbench)
    sys.path.insert(0, str(args.rmbench))
    if importlib.util.find_spec("curobo") is None:
        source = Path.home() / "project/benchmark/RoboTwin/envs/curobo/src"
        if (source / "curobo").is_dir():
            sys.path.insert(0, str(source))

    import sapien
    from envs.put_back_block import put_back_block
    from envs.utils.create_actor import UnStableError

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU to each evaluation process")
    pci = pci_address(args.render_pci)
    device = sapien.Device(f"pci:{pci}")
    if device.cuda_id != 0 or not device.can_render() or pci_address(device.pci_string) != pci:
        raise RuntimeError(f"SAPIEN device {device} is not the selected CUDA/Vulkan GPU at {pci}")

    configs = Path("env_cfg/task_config")
    task_config = protocol["task_config"]
    config = yaml.safe_load((configs / f"{task_config}.yml").read_text())
    embodiment, = config["embodiment"]
    robot_dir = yaml.safe_load((configs / "_embodiment_config.yml").read_text())[embodiment]["file_path"]
    robot_config = yaml.safe_load((Path(robot_dir) / "config.yml").read_text())
    config.update(
        task_name=protocol["task"], task_config=task_config, eval_mode=True,
        render_freq=0, render_device=device, camera_shader=args.camera_shader,
        left_robot_file=robot_dir, right_robot_file=robot_dir, dual_arm_embodied=True,
        left_embodiment_config=robot_config, right_embodiment_config=robot_config,
    )
    config["camera"]["collect_wrist_camera"] = bool(
        {"left_camera", "right_camera"} & set(protocol["cameras"])
    )
    config["data_type"].update(endpose=False, third_view="third_view" in protocol["cameras"])
    if not config["data_type"]["rgb"] or not config["data_type"]["qpos"]:
        raise RuntimeError("Evaluation requires official RGB and joint-state observations")
    official_limit = yaml.safe_load((configs / "_eval_step_limit.yml").read_text())[protocol["task"]]
    if official_limit != protocol["max_episode_steps"]:
        raise ValueError(f"Official action limit is {official_limit}, expected {protocol['max_episode_steps']}")
    return put_back_block, UnStableError, config


def reset(env, config, seed, episode_id, limit):
    env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **config)
    if env.step_lim != limit:
        raise RuntimeError(f"Episode {seed} uses unexpected action limit {env.step_lim}")
    bound = pci_address(env.scene.render_system.device.pci_string)
    if bound != pci_address(config["render_device"].pci_string):
        raise RuntimeError(f"SAPIEN scene bound to {bound}")
    return env


def seed_protocol(protocol):
    return {
        "task": protocol["task"], "task_config": protocol["task_config"],
        "seed_start": protocol["eval_seed_start"],
    }


def prepare_seeds(args, protocol, factory, unstable, config):
    evaluation = args.experiment / "evaluation"
    start = protocol["eval_seed_start"]
    count = protocol["eval_episodes"]
    sharded = args.candidate_start is not None
    first = args.candidate_start if sharded else start
    end = args.candidate_stop if sharded else start + 10 * count
    if sharded:
        path = evaluation / f"seed_shards/{first}_{end}.json"
    else:
        path = evaluation / "reset_seeds.json"
    expected = seed_protocol(protocol)
    if path.exists():
        progress = json.loads(path.read_text())
        if any(progress[k] != v for k, v in expected.items()):
            raise ValueError("Saved resets belong to another task or protocol")
    else:
        progress = {**expected, "seeds": [], "next_candidate": first, "complete": False}
        if sharded:
            progress.update(candidate_start=first, candidate_stop=end)
    seeds = progress["seeds"]
    env = factory()
    while progress["next_candidate"] < end and (sharded or len(seeds) < count):
        candidate = progress["next_candidate"]
        try:
            reset(env, config, candidate, candidate - start, protocol["max_episode_steps"])
            env.play_once()
            valid = bool(env.plan_success and env.check_success())
        except unstable:
            valid = False
        finally:
            if hasattr(env, "scene"):
                env.close_env(clear_cache=(candidate - first + 1) % config["clear_cache_freq"] == 0)
        if valid:
            seeds.append(candidate)
        complete = candidate + 1 == end if sharded else len(seeds) == count
        progress.update(next_candidate=candidate + 1, complete=complete)
        atomic_json(path, progress)
        print(json.dumps({"candidate": candidate, "expert_valid": valid, "valid_seeds": len(seeds)}), flush=True)
    if not sharded and len(seeds) != count:
        raise RuntimeError("Too few expert-valid official initial conditions")


def merge_seeds(experiment, protocol):
    evaluation = experiment / "evaluation"
    output = evaluation / "reset_seeds.json"
    expected = seed_protocol(protocol)
    start = expected["seed_start"]
    count = protocol["eval_episodes"]
    known = {}
    files = sorted((evaluation / "seed_shards").glob("*.json"))
    if output.exists():
        files.append(output)
    for path in files:
        shard = json.loads(path.read_text())
        if any(shard[k] != v for k, v in expected.items()):
            raise ValueError(f"Seed shard belongs to another protocol: {path}")
        positive = set(shard["seeds"])
        for candidate in range(shard.get("candidate_start", start), shard["next_candidate"]):
            valid = candidate in positive
            if known.setdefault(candidate, valid) != valid:
                raise ValueError(f"Conflicting expert success for reset {candidate}")
    seeds = []
    candidate = start
    while candidate in known and len(seeds) < count:
        if known[candidate]:
            seeds.append(candidate)
        candidate += 1
    merged = {**expected, "seeds": seeds, "next_candidate": candidate, "complete": len(seeds) == count}
    atomic_json(output, merged)
    print(json.dumps(merged), flush=True)


def policy_observation(observation, cameras):
    rgb = {
        key: adapter.preprocess_rgb(observation["third_view_rgb"] if key == "third_view"
                            else observation["observation"][key]["rgb"])
        for key in cameras
    }
    qpos = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise RuntimeError(f"Invalid simulator joint state: {qpos}")
    return {**rgb, STATE_KEY: qpos}


def stack_batch(observations):
    result = {}
    for key in observations[0]:
        value = torch.from_numpy(np.stack([obs[key] for obs in observations]))
        result[key] = (value.to("cuda", dtype=torch.float32).div_(255)
                       if value.dtype == torch.uint8 else value.to("cuda"))
    return result


class Video:
    def __init__(self, path, first):
        self.path = path
        height, width = first.shape[:2]
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", "15", "-i", "pipe:0", "-an",
             "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )
        self.write(first)

    def write(self, image):
        self.process.stdin.write(image.tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError(f"Video encoder failed for {self.path}")


def report_state(episode):
    env = episode["env"]
    position = np.asarray(env.block.get_pose().p)
    error = np.linalg.norm(position[:2] - np.asarray(env.target_pose)[:2])
    return {"step": int(env.take_action_cnt), "stage": int(env.stage_id),
            "button_presses": int(env.press_cnt), "xy_return_error_m": float(error)}


def parameter_counts(policy, variant):
    training = sum(parameter.numel() for parameter in policy.parameters())
    auxiliary = 0 if variant == "M0" else sum(p.numel() for p in policy.past_head.parameters())
    return {"training_parameters": training, "inference_parameters": training - auxiliary}


@torch.inference_mode()
def evaluate_group(args, protocol, policy, config, entries, output, record_video):
    uses_memory = args.model != "M0"
    cameras = protocol["cameras"]
    limit = protocol["max_episode_steps"]
    episodes = {}
    for env_id, env, episode_id, seed in entries:
        env_config = {**config, "data_type": dict(config["data_type"])}
        env_config["data_type"]["third_view"] |= record_video
        reset(env, env_config, seed, episode_id, limit)
        raw = env.get_obs()
        obs = policy_observation(raw, cameras)
        episode = {
            "env": env, "seed": seed, "episode_id": episode_id, "observations": 1,
            "obs": deque([obs, obs], maxlen=2), "video": None, "trace": [],
            "observe_seconds": 0.0, "predict_seconds": 0.0, "replans": 0,
        }
        if record_video:
            episode["video"] = Video(output / f"seed_{seed}.pending.mp4", raw["third_view_rgb"])
        episode["trace"].append(report_state(episode))
        episodes[env_id] = episode

    def advance(ids):
        if uses_memory and ids:
            start = time.perf_counter()
            policy.update_observation(stack_batch([episodes[i]["obs"][-1] for i in ids]), ids)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / len(ids)
            for i in ids:
                episodes[i]["observe_seconds"] += elapsed

    def running(episode):
        env = episode["env"]
        return not env.eval_success and env.take_action_cnt < (args.verify_steps or limit)

    ids = list(episodes)
    if uses_memory:
        policy.reset_memory(ids)
        advance(ids)

    first_seed = entries[0][-1]
    chunk = 0
    try:
        while active := [i for i in ids if running(episodes[i])]:
            torch.manual_seed(first_seed + 1000003 * chunk)
            start = time.perf_counter()
            if uses_memory:
                prediction = policy.predict_stream(active)
            else:
                batched = {}
                for frame in range(2):
                    tensors = stack_batch([episodes[i]["obs"][frame] for i in active])
                    for key, value in tensors.items():
                        batched.setdefault(key, []).append(value)
                prediction = policy.predict_action({
                    key: torch.stack(frames, dim=1) for key, frames in batched.items()
                })
            actions = prediction["action"].float().cpu().numpy()
            expected = (len(active), protocol["n_action_steps"], 14)
            if actions.shape != expected or not np.isfinite(actions).all():
                raise RuntimeError(f"Expected finite joint actions {expected}, got {actions.shape}")
            elapsed = (time.perf_counter() - start) / len(active)
            for env_id in active:
                episodes[env_id]["replans"] += 1
                episodes[env_id]["predict_seconds"] += elapsed
            for action_index in range(actions.shape[1]):
                updated = []
                for row, env_id in enumerate(active):
                    episode = episodes[env_id]
                    if not running(episode):
                        continue
                    env = episode["env"]
                    env.take_action(actions[row, action_index], action_type="qpos")
                    # take_action already captures the terminal success observation.
                    raw = env.now_obs if env.eval_success else env.get_obs()
                    episode["obs"].append(policy_observation(raw, cameras))
                    episode["observations"] += 1
                    updated.append(env_id)
                    if episode["video"] is not None:
                        episode["video"].write(raw["third_view_rgb"])
                    stage = report_state(episode)
                    if stage["stage"] != episode["trace"][-1]["stage"]:
                        episode["trace"].append(stage)
                advance(updated)
            chunk += 1
    finally:
        for episode in episodes.values():
            if episode["video"] is not None:
                episode["video"].close()
            episode["env"].close_env()

    diagnostics = policy.memory_diagnostics(ids) if uses_memory else [None] * len(ids)
    if uses_memory:
        policy.reset_memory(ids)
    for i, diagnostic in zip(ids, diagnostics):
        episode = episodes[i]
        env = episode["env"]
        if episode["observations"] != env.take_action_cnt + 1:
            raise RuntimeError("Actual observations and environment actions lost alignment")
        if diagnostic is not None and diagnostic["observations"] != episode["observations"]:
            raise RuntimeError("Mamba did not consume exactly one update per actual observation")
        yield {
            "episode_id": episode["episode_id"], "seed": episode["seed"],
            "success": bool(env.eval_success), "steps": int(env.take_action_cnt),
            "observations": episode["observations"], "replans": episode["replans"],
            "observe_seconds": episode["observe_seconds"], "predict_seconds": episode["predict_seconds"],
            "stage_transitions": episode["trace"], "final": report_state(episode),
            "render_pci": pci_address(config["render_device"].pci_string),
            "memory": diagnostic, "video": None,
        }


def get_policy(args, protocol):
    build_policy = importlib.import_module(f"{IMPLEMENTATION.name}.models").build_policy

    if args.command == "verify":
        torch.manual_seed(protocol["seed"])
        policy = build_policy(args.model, adapter.observation_meta(protocol), protocol)
        policy.set_normalizer(adapter.EpisodeDataset(args.data).normalizer())
    else:
        checkpoint = args.experiment / f"checkpoints/{args.model}/epoch_{args.epoch:03d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload["model_variant"] != args.model or payload["epoch"] != args.epoch
                or payload["config"] != protocol):
            raise ValueError(f"Checkpoint protocol mismatch: {checkpoint}")
        policy = build_policy(args.model, payload["shape_meta"], payload["config"])
        policy.load_state_dict(payload["ema"])
    return policy.cuda().eval()


def summarize(path, args, protocol, policy, rows):
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    complete = count == protocol["eval_episodes"]
    payload = {
        "model": args.model, "epoch": args.epoch, "episodes": count, "complete": complete,
        "successes": successes, "success_rate": successes / count if complete else None,
        "steps": sum(row["steps"] for row in rows), "replans": sum(row["replans"] for row in rows),
        "observe_seconds": sum(row["observe_seconds"] for row in rows),
        "predict_seconds": sum(row["predict_seconds"] for row in rows),
        "render_pci": pci_address(args.render_pci), "camera_shader": args.camera_shader,
        "render_pcis": sorted({row.get("render_pci", pci_address(args.render_pci)) for row in rows}),
        **parameter_counts(policy, args.model),
    }
    atomic_json(path, payload)
    return payload


def reached_cutoff(args):
    return args.stop_at is not None and time.time() >= args.stop_at.timestamp()


def stopped_at_cutoff(args, output, episodes=None):
    if episodes is None:
        log = output / "episodes.jsonl"
        episodes = sum(bool(line) for line in log.read_text().splitlines()) if log.exists() else 0
    print(json.dumps({"event": "evaluation_cutoff", "model": args.model, "epoch": args.epoch,
                      "episodes": episodes, "complete": False}), flush=True)
    return 75


def evaluate_checkpoint(args, protocol):
    output = args.experiment / f"evaluation/{args.model}/epoch_{args.epoch:03d}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".rollout.lock").open("a") as lock:
        if args.stop_at is None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            while not reached_cutoff(args):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(1, max(0, args.stop_at.timestamp() - time.time())))
            else:
                return stopped_at_cutoff(args, output)

        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if (summary["complete"] and summary["episodes"] == protocol["eval_episodes"]
                    and summary["model"] == args.model and summary["epoch"] == args.epoch):
                print(json.dumps({**summary, "already_complete": True}), flush=True)
                return 0
        if reached_cutoff(args):
            return stopped_at_cutoff(args, output)
        factory, _, config = make_simulator(args, protocol)
        policy = get_policy(args, protocol)
        return rollout(args, protocol, policy, factory, config, output)


def rollout(args, protocol, policy, factory, config, output):
    reset_path = args.experiment / "evaluation/reset_seeds.json"
    manifest = json.loads(reset_path.read_text())
    seeds = manifest["seeds"]
    if (not manifest["complete"] or len(seeds) != protocol["eval_episodes"]
            or len(set(seeds)) != len(seeds) or manifest["task"] != protocol["task"]
            or manifest["task_config"] != protocol["task_config"]):
        raise ValueError("Expected one complete official reset list for all checkpoints")
    run_config_path = output / "evaluation_protocol.json"
    run_config = {
        "batch_size": args.batch_size, "camera_shader": args.camera_shader,
        "seed_order": seeds, "rng": "first seed in batch + 1000003 * action chunk index",
    }
    if run_config_path.exists() and json.loads(run_config_path.read_text()) != run_config:
        raise ValueError("Cannot resume a checkpoint with a different evaluation protocol")
    atomic_json(run_config_path, run_config)
    log = output / "episodes.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line] if log.exists() else []
    if [row["seed"] for row in rows] != seeds[:len(rows)]:
        raise ValueError("Saved rollouts do not match the fixed reset sequence")
    kinds = {row["success"] for row in rows if row["video"] and (output / row["video"]).is_file()}
    batch_size = args.batch_size
    if len(rows) != len(seeds) and len(rows) % batch_size:
        raise ValueError("Saved rollouts end partway through a deterministic batch")
    environments = [factory() for _ in range(min(batch_size, len(seeds) - len(rows)))]
    for start in range(len(rows), len(seeds), batch_size):
        if reached_cutoff(args):
            summarize(output / "summary.json", args, protocol, policy, rows)
            return stopped_at_cutoff(args, output, len(rows))
        entries = [(slot, env, start + slot, seed)
                   for slot, (env, seed) in enumerate(zip(environments, seeds[start:start + batch_size]))]
        group = list(evaluate_group(args, protocol, policy, config, entries, output, len(kinds) < 2))
        for row in group:
            pending = output / ("seed_%s.pending.mp4" % row["seed"])
            if pending.exists():
                if row["success"] not in kinds:
                    kind = "success" if row["success"] else "failure"
                    video = output / ("seed_%s_%s.mp4" % (row["seed"], kind))
                    pending.replace(video)
                    row["video"] = video.name
                    kinds.add(row["success"])
                else:
                    pending.unlink()
        rows.extend(group)
        temporary = log.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                             encoding="utf-8")
        temporary.replace(log)
        summarize(output / "summary.json", args, protocol, policy, rows)
        for row in group:
            print(json.dumps({k: row[k] for k in ("episode_id", "seed", "success", "steps")}), flush=True)
        if (start // batch_size + 1) % config["clear_cache_freq"] == 0:
            environments[0].close_env(clear_cache=True)
    print(json.dumps(summarize(output / "summary.json", args, protocol, policy, rows)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--rmbench", type=Path, default=WORKSPACE / "source/RMBench")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data/prepared")
    parser.add_argument("--render-pci")
    parser.add_argument("--camera-shader", choices=["rt", "default"], default="rt")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-seeds")
    prepare.add_argument("--candidate-start", type=int)
    prepare.add_argument("--candidate-stop", type=int)
    sub.add_parser("merge-seeds")
    verify = sub.add_parser("verify")
    verify.add_argument("--seed", type=int)
    verify.add_argument("--steps", type=int, default=17)
    formal = sub.add_parser("rollout")
    formal.add_argument("--model", choices=["M0", "M3"], required=True)
    formal.add_argument("--epoch", type=int, required=True)
    formal.add_argument("--batch-size", type=int, default=2)
    formal.add_argument("--stop-at", type=datetime.fromisoformat,
                        help="Stop before another batch after this ISO 8601 time with UTC offset")
    args = parser.parse_args()
    args.experiment, args.rmbench, args.data = (p.resolve() for p in (args.experiment, args.rmbench, args.data))
    protocol = json.loads((args.experiment / "configs/protocol.json").read_text())
    if args.command == "merge-seeds":
        merge_seeds(args.experiment, protocol)
        return
    if args.render_pci is None:
        parser.error("--render-pci is required for simulator commands")
    if args.command == "prepare-seeds" and (args.candidate_start is not None or args.candidate_stop is not None):
        if (args.candidate_start is None or args.candidate_stop is None
                or not protocol["eval_seed_start"] <= args.candidate_start < args.candidate_stop):
            parser.error("Expected a valid --candidate-start/--candidate-stop range")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "rollout":
        if args.batch_size < 1 or args.epoch not in protocol["eval_epochs"]:
            parser.error("Expected a protocol checkpoint and a positive evaluation batch size")
        if args.stop_at is not None and args.stop_at.utcoffset() is None:
            parser.error("--stop-at requires a time zone offset")
        args.verify_steps = None
        return evaluate_checkpoint(args, protocol)
    factory, unstable, config = make_simulator(args, protocol)
    if args.command == "prepare-seeds":
        prepare_seeds(args, protocol, factory, unstable, config)
        return
    args.verify_steps = args.steps
    args.model = "M3"
    if args.steps <= protocol["n_action_steps"]:
        parser.error("Verification must cross an action window")
    seed = protocol["eval_seed_start"] if args.seed is None else args.seed
    policy = get_policy(args, protocol)
    output = args.experiment / "evaluation/verification"
    output.mkdir(parents=True, exist_ok=True)
    result, = evaluate_group(args, protocol, policy, config, [(0, factory(), 0, seed)], output, False)
    if result["replans"] < 2 or result["steps"] != args.steps:
        raise RuntimeError(f"Verification did not cover the requested action windows: {result}")
    atomic_json(output / "real_closed_loop.json", {
        "render_pci": pci_address(args.render_pci), "camera_shader": args.camera_shader, **result
    })
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

