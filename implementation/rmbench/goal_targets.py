"""Index official event endpoints and precompute frozen visual goal labels."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from data import DEMO_COUNT, EpisodeDataset, _read_expert, cover_phases, preprocess_rgb

GOAL_DIM = 512


def events_from_durations(episode_id, durations, paired_frames, label_source):
    """Project raw capture intervals onto XPolicyLab time t (state/RGB t, action t+1)."""
    if not durations or any(type(n) is not int or n <= 0 for n in durations):
        raise ValueError(f"Invalid capture durations for episode {episode_id}")
    raw_frames = sum(durations)
    if raw_frames != paired_frames + 1:
        raise ValueError(f"Episode {episode_id}: {raw_frames} annotated raw vs {paired_frames} paired frames")
    start = 0
    events = []
    for event_id, raw_length in enumerate(durations):
        raw_end = start + raw_length
        # The converter omits one terminal RGB globally, while keeping it as action[t-1].
        end = raw_end - int(raw_end == raw_frames)
        if end <= start:
            raise ValueError(f"Episode {episode_id} event {event_id} has no paired RGB")
        valid_until = start + (end - start) // 2
        events.append({
            "episode_id": episode_id, "event_id": event_id, "start_frame": start,
            "end_frame": end, "raw_end_frame": raw_end, "goal_frame_index": end - 1,
            "goal_feature": len(events), "label_source": label_source,
            "valid_mask": {"start_frame": start, "end_frame": valid_until},
        })
        start = raw_end
    return events


class GoalTargets:
    """Targets are batch labels keyed by chronological anchor, never observations."""

    def __init__(self, directory, episodes):
        metadata = json.loads((directory / "events.json").read_text())
        self.features = np.load(directory / "goal_features.npy", allow_pickle=False)
        if (self.features.shape != (len(metadata["events"]), GOAL_DIM)
                or not np.isfinite(self.features).all()
                or not np.allclose(np.linalg.norm(self.features, axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid frozen goal features")
        self.anchor_index = {ep["id"]: np.full(ep["steps"], -1, dtype=np.int64) for ep in episodes}
        end_by_episode = dict.fromkeys(self.anchor_index, 0)
        for feature_index, event in enumerate(metadata["events"]):
            ep = event["episode_id"]
            start, stop = event["start_frame"], event["end_frame"]
            mask_end = event["valid_mask"]["end_frame"]
            values = self.anchor_index[ep]
            if not (event["goal_feature"] == feature_index and start == end_by_episode[ep]
                    and event["valid_mask"]["start_frame"] == start
                    and start <= mask_end <= stop - 1 < len(values)
                    and event["goal_frame_index"] == stop - 1):
                raise ValueError(f"Invalid event index: {event}")
            values[start:mask_end] = feature_index
            end_by_episode[ep] = stop
        if any(end_by_episode[ep] != len(values) for ep, values in self.anchor_index.items()):
            raise ValueError("Goal metadata does not cover all prepared episodes")

    def batch(self, episode_id, start, stop):
        indices = self.anchor_index[episode_id][start:stop]
        valid = indices >= 0
        labels = np.zeros((len(indices), GOAL_DIM), dtype=self.features.dtype)
        labels[valid] = self.features[indices[valid]]
        return {"goal_feature": torch.from_numpy(labels), "goal_valid_mask": torch.from_numpy(valid)}


def collect_cover_events(archive, dataset, decode_image):
    if dataset.manifest["task"] != "cover_blocks" or len(dataset.episodes) != DEMO_COUNT:
        raise ValueError("Expected all 50 official Cover Blocks demonstrations")
    events, keyframes, trace = [], [], []
    with ZipFile(archive) as zipped:
        annotation_member, = (name for name in zipped.namelist() if name.endswith("/language_annotation.json"))
        annotations = json.loads(zipped.read(annotation_member))
        if set(annotations) != {f"episode_{ep['id']}" for ep in dataset.episodes}:
            raise ValueError("Official and prepared episode IDs differ")
        for ep in dataset.episodes:
            episode_id = ep["id"]
            phases = cover_phases(annotations, episode_id)
            if json.loads(json.dumps(phases)) != ep["official_cover_phases"]:
                raise ValueError(f"Official phase durations differ from prepared episode {episode_id}")
            durations = [phase["frames"] for phase in phases]
            official_rows = annotations[f"episode_{episode_id}"]
            if sum(durations) != sum(frames for _, frames in official_rows):
                raise ValueError(f"Annotations contain ignored or uncovered frames for episode {episode_id}")
            current = events_from_durations(episode_id, durations, ep["steps"], annotation_member)
            with h5py.File(io.BytesIO(zipped.read(ep["archive_member"])), "r") as h5:
                images, state, actions, source = _read_expert(h5)
                if (source["layout"] != "XPolicyLab" or len(images) != ep["steps"]
                        or not np.array_equal(actions[:-1], state[1:])
                        or not np.array_equal(state, dataset._load(episode_id, "state"))
                        or not np.array_equal(actions, dataset._load(episode_id, "action"))):
                    raise ValueError(f"State/RGB/action indexing mismatch for episode {episode_id}")
                prepared_rgb = dataset._load(episode_id, "rgb")
                if len(prepared_rgb) != len(images):
                    raise ValueError(f"Prepared RGB length differs in episode {episode_id}")
                for event in current:
                    start, last = event["start_frame"], event["goal_frame_index"]
                    rgb = decode_image(images[last])
                    if (not np.array_equal(preprocess_rgb(decode_image(images[start])), prepared_rgb[start])
                            or not np.array_equal(preprocess_rgb(rgb), prepared_rgb[last])):
                        raise ValueError(f"Prepared RGB boundary mismatch at episode {episode_id}: {start}, {last}")
                    event["goal_feature"] = len(keyframes)
                    keyframes.append(rgb)
            events.extend(current)
            trace.append({"episode_id": episode_id, "raw_frames": sum(durations),
                          "paired_frames": ep["steps"], "events": len(current)})
    return events, keyframes, trace


@torch.inference_mode()
def encode_keyframes(images, weights_path, device):
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import v2

    version = ResNet18_Weights.IMAGENET1K_V1
    digest = hashlib.file_digest if hasattr(hashlib, "file_digest") else None
    if digest is not None:
        with weights_path.open("rb") as handle:
            sha = digest(handle, "sha256").hexdigest()
    else:
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if not sha.startswith("f37072fd"):
        raise ValueError("Expected the official torchvision ResNet18 IMAGENET1K_V1 checkpoint")
    teacher = resnet18(weights=None)
    teacher.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    teacher.fc = torch.nn.Identity()
    teacher.eval().requires_grad_(False).to(device)
    transform = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])
    features = []
    for start in range(0, len(images), 32):
        batch = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images[start:start + 32]])
        features.append(F.normalize(teacher(transform(batch).to(device)), dim=-1).cpu().numpy())
    return np.concatenate(features), {
        "teacher": "torchvision ResNet18 IMAGENET1K_V1", "checkpoint_url": version.url, "sha256": sha,
        "preprocess": "native head RGB, 224x224 bilinear antialias resize, ImageNet normalization",
        "feature": "global average pool, 512 dimensions, unit L2 norm",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "data", "weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from XPolicyLab.utils.process_data import decode_image_bit
    torch.set_num_threads(4)
    dataset = EpisodeDataset(args.data)
    events, keyframes, trace = collect_cover_events(args.archive, dataset, decode_image_bit)
    features, teacher = encode_keyframes(keyframes, args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "goal_features.npy", features, allow_pickle=False)
    metadata = {"schema_version": 1, "task": dataset.manifest["task"], "teacher": teacher,
                "alignment": "native capture t maps to paired RGB[t], state[t], and action from t+1",
                "episodes": trace, "events": events}
    (args.output / "events.json").write_text(json.dumps(metadata, indent=2) + "\n")
    targets = GoalTargets(args.output, dataset.episodes)
    statistics = {
        "episodes": len(trace), "events": len(events),
        "raw_frames": sum(t["raw_frames"] for t in trace),
        "paired_frames": sum(t["paired_frames"] for t in trace),
        "valid_training_anchors": sum(np.count_nonzero(
            targets.anchor_index[ep["id"]][:dataset._anchor_count(ep["steps"], 16, 8)] >= 0)
            for ep in dataset.episodes),
        "all_training_anchors": dataset.num_samples(16, 8), "teacher": teacher,
    }
    (args.output / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics), flush=True)


if __name__ == "__main__":
    main()
