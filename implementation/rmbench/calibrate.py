"""Fix separate M3 novelty thresholds using only official training demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import EpisodeDataset
from models import build_policy
from train import configure_precision, cuda_observations, observation_meta


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    protocol = args.root / "configs/protocol.json"
    cfg = json.loads(protocol.read_text())
    if "event_thresholds" in cfg:
        print(json.dumps({"event_thresholds": cfg["event_thresholds"], "already_fixed": True}))
        return

    calibration = cfg["event_threshold_calibration"]
    dataset = EpisodeDataset(args.data)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.set_num_threads(3)
    configure_precision()
    policy = build_policy("M3", observation_meta(cfg), cfg)
    policy.set_normalizer(dataset.normalizer())
    policy.cuda().eval()
    count = min(calibration["train_episodes"], len(dataset.episodes))
    indices = np.linspace(0, len(dataset.episodes) - 1, count, dtype=int)
    episode_ids = [dataset.episodes[index]["id"] for index in indices]
    scores = {"visual": [], "history": []}

    for episode_id in episode_ids:
        observations = cuda_observations(dataset.episode_observations(episode_id))
        keys = tuple(observations)
        chunks = zip(*(observations[key].split(cfg["batch_size"]) for key in keys))
        features = torch.cat([policy.encode_observations(dict(zip(keys, chunk))) for chunk in chunks])
        history, _ = policy.history_sequence(features.unsqueeze(0))
        vision, history = policy.event_features(features, history[0])
        for key, values in (("visual", vision), ("history", history)):
            similarity = F.cosine_similarity(values[:-1].float(), values[1:].float(), dim=-1)
            scores[key].append(similarity.cpu().numpy())
        print(json.dumps({"calibration_episode": episode_id, "frames": len(features)}), flush=True)

    quantile = calibration["adjacent_cosine_quantile"]
    thresholds, distributions = {}, {}
    for key, arrays in scores.items():
        values = np.concatenate(arrays)
        if not len(values) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid {key} calibration cosine distribution")
        thresholds[key] = float(np.quantile(values, quantile))
        distributions[key] = {
            "count": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }
    calibration_result = {
        "method": "Separate cosine quantiles on adjacent real observations; no evaluation data",
        "model": "Seeded from-scratch M3 in evaluation mode before optimizer updates",
        "seed": cfg["seed"],
        "episode_ids": episode_ids,
        "quantile": quantile,
        "distributions": distributions,
        "event_thresholds": thresholds,
    }
    write_json(args.root / "configs/event_threshold_calibration.json", calibration_result)
    cfg["event_thresholds"] = thresholds
    write_json(protocol, cfg)
    print(json.dumps(calibration_result), flush=True)


if __name__ == "__main__":
    main()
