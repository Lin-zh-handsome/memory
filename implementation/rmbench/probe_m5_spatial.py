"""Measure what the trained spatial auxiliary head can read from the causal shared state."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import EpisodeDataset
from models import HISTORY_DIM, build_policy
from train import configure_precision, cuda_observations


@torch.inference_mode()
def predictions(root, data, epoch, events):
    checkpoint = root / f"checkpoints/M5/epoch_{epoch:03d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["model_variant"] != "M5" or payload["epoch"] != epoch:
        raise ValueError(f"Wrong M5 checkpoint: {checkpoint}")
    policy = build_policy("M5", payload["shape_meta"], payload["config"])
    policy.load_state_dict(payload["ema"])
    policy.cuda().eval()
    result = {}
    for episode, items in events.items():
        observations = data.episode_observations(episode)
        # One mid-point of the supervised prefix per event. The head and labels are
        # called only after the same causal forward used by the policy.
        anchors = {int((item["start_frame"] + item["valid_mask"]["end_frame"] - 1) // 2):
                   item["goal_feature_index"] for item in items}
        carry = None
        for start in range(0, max(anchors) + 1, 64):
            stop = min(start + 64, max(anchors) + 1)
            obs = cuda_observations({k:v[start:stop] for k,v in observations.items()})
            conditions, _, carry = policy.segment_condition(obs, carry)
            to_read = [anchor for anchor in anchors if start <= anchor < stop]
            if to_read:
                latent = conditions[[anchor-start for anchor in to_read], -HISTORY_DIM:]
                prediction = policy.goal_head(latent).reshape(len(to_read), 9, 512)
                encoded = F.normalize(prediction, dim=-1).cpu().numpy()
                result.update({anchors[anchor]: value for anchor,value in zip(to_read, encoded)})
    return result


def compare(values, goals, audit):
    ids = sorted(values)
    received = np.stack([values[i] for i in ids])
    target = goals[ids]
    episodes = np.array([audit[i]["episode_id"] for i in ids])
    labels = np.array([audit[i]["diagnostic_state"] for i in ids])
    open_mask = np.array([audit[i]["diagnostic_operation"].startswith("Open ") for i in ids])
    open_labels = np.array([audit[i]["diagnostic_operation"] for i in ids])
    allowed = episodes[:, None] != episodes[None, :]
    matched = (received*target).sum(axis=-1).mean(axis=-1)
    prototypes = np.einsum("ij,jkd->ikd",allowed,target)
    prototypes /= np.linalg.norm(prototypes,axis=-1,keepdims=True).clip(min=1e-9)
    mean_baseline = (prototypes*target).sum(axis=-1).mean(axis=-1)
    similarity = received.reshape(len(ids),-1) @ target.reshape(len(ids),-1).T / 9
    whole_neighbor = np.where(allowed,similarity,-np.inf).argmax(axis=1)
    open_neighbor = np.where(allowed & open_mask[None,:],similarity,-np.inf).argmax(axis=1)
    return {"events":len(ids),"mean_matched_cosine":float(matched.mean()),
            "mean_other_episode_prototype_cosine":float(mean_baseline.mean()),
            "cross_episode_state_top1":float((labels==labels[whole_neighbor]).mean()),
            "cross_episode_uncover_position_top1":float((open_labels[open_mask]==open_labels[open_neighbor[open_mask]]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","data","targets"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    configure_precision()
    dataset = EpisodeDataset(args.data)
    goals = np.load(args.targets/"goal_features.npy",allow_pickle=False)
    audit = json.loads((args.targets/"keyframe_audit.json").read_text())
    events = {}
    for item in audit:
        if item["valid_mask"]["end_frame"] > item["start_frame"]:
            events.setdefault(item["episode_id"],[]).append(item)
    result = {}
    for epoch in (50,250):
        result[epoch] = compare(predictions(args.root,dataset,epoch,events),goals,audit)
        print(json.dumps({"epoch":epoch,**result[epoch]}),flush=True)
    (args.root/"goal_readout.json").write_text(json.dumps({
        "scope":"Training demonstrations; neighbor gallery excludes the query episode. No claim on held-out policy training data.",
        "checkpoints":result},indent=2)+"\n")


if __name__ == "__main__":
    main()
