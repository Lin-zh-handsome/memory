"""Shared expert data and observation preprocessing for both RMBench policies."""

import argparse
import io
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import cv2
import numpy as np
import torch


IMAGE_SIZE = 96
CAMERA_KEY = "head_camera"
STATE_KEY = "agent_pos"
PAST_STEPS = 8
DEMO_COUNT = 50
JOINT_KEYS = (
    "left_arm_joint_states", "left_ee_joint_states",
    "right_arm_joint_states", "right_ee_joint_states",
)


def observation_meta(cfg):
    return {
        "action": {"shape": [14]},
        "obs": {
            **{camera: {"shape": [3, *cfg["vision"]["resolution"]], "type": "rgb"}
               for camera in cfg["cameras"]},
            "agent_pos": {"shape": [14], "type": "low_dim"},
        },
    }



def preprocess_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC uint8 RGB, got {image.shape}, {image.dtype}")
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def _joint_vector(group):
    return np.concatenate(
        [np.asarray(group[key], dtype=np.float32).reshape(len(group[key]), -1)
         for key in JOINT_KEYS],
        axis=1,
    )


def _read_expert(h5):
    if all(key in h5 for key in ("state", "action", "vision")):
        cameras = h5["vision"]
        images = cameras["cam_head/colors"]
        state = _joint_vector(h5["state"])
        action = _joint_vector(h5["action"])
        frequency = h5.get("additional_info/frequency")
        if frequency is not None:
            frequency = float(np.asarray(frequency).reshape(-1)[0])
        layout = "XPolicyLab"
    else:
        cameras = h5["observation"]
        images = cameras["head_camera/rgb"]
        joints = np.asarray(h5["joint_action/vector"], dtype=np.float32)
        state, action = joints[:-1], joints[1:]
        images = images[:len(state)]
        frequency = None
        layout = "legacy RMBench"

    if len(state) == 0 or state.shape != (len(images), 14) or action.shape != state.shape:
        raise ValueError(
            f"Unexpected expert shapes: RGB={len(images)}, state={state.shape}, action={action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite robot state or action in expert demonstration")

    return images, state, action, {
        "layout": layout,
        "cameras": list(cameras),
        "frequency_metadata_hz": frequency,
        "action_matches_next_state": bool(np.allclose(action[:-1], state[1:], atol=1e-5)),
    }


def prepare(archive, prepared_dir, rmbench):
    import h5py

    rmbench = Path(rmbench).resolve()
    sys.path.insert(0, str(rmbench))
    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ImportError:
        from data.decode_image_bit import decode_image_bit

    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "put_back_block",
        "config": "demo_clean",
        "split": "all 50 official demonstrations for training, no offline validation",
        "camera": CAMERA_KEY,
        "processed_rgb": [3, IMAGE_SIZE, IMAGE_SIZE],
        "preprocess": "decode_image_bit -> cv2.INTER_AREA resize -> uint8 CHW",
        "state_action": "absolute joint positions in radians plus normalized gripper commands",
        "joint_order": "left arm (6), left gripper, right arm (6), right gripper",
        "physics_timestep_s": 1 / 250,
        "expert_save_every_physics_steps": 15,
        "episodes": [],
    }
    cv2.setNumThreads(1)
    with ZipFile(archive) as zipped:
        members = {}
        for name in zipped.namelist():
            match = re.fullmatch(r"episode_?(\d+)\.hdf5", Path(name).name)
            if match and Path(name).parent.name == "data":
                episode_id = int(match.group(1))
                if episode_id in members:
                    raise ValueError(f"Duplicate expert episode {episode_id}")
                members[episode_id] = name
        if sorted(members) != list(range(DEMO_COUNT)):
            raise ValueError(f"Expected {DEMO_COUNT} official episode IDs, got {sorted(members)}")

        for episode_id, name in sorted(members.items()):
            with h5py.File(io.BytesIO(zipped.read(name)), "r") as h5:
                images, state, action, metadata = _read_expert(h5)
                first_rgb = decode_image_bit(images[0])
                frames = np.empty((len(state), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                frames[0] = preprocess_rgb(first_rgb)
                for time in range(1, len(state)):
                    frames[time] = preprocess_rgb(decode_image_bit(images[time]))
                metadata["source_rgb_hwc"] = list(first_rgb.shape)
                if episode_id == 0:
                    print(
                        f"First expert RGB {first_rgb.shape}; {metadata}; "
                        f"state/action shape {state.shape}",
                        flush=True,
                    )

            episode_dir = prepared_dir / f"episode_{episode_id:07d}"
            episode_dir.mkdir(exist_ok=True)
            np.save(episode_dir / "rgb.npy", frames, allow_pickle=False)
            np.save(episode_dir / "state.npy", state, allow_pickle=False)
            np.save(episode_dir / "action.npy", action, allow_pickle=False)
            manifest["episodes"].append({
                "id": episode_id, "steps": len(state), "archive_member": name, **metadata,
            })
            print(f"Prepared expert episode {episode_id + 1}/{DEMO_COUNT}: {len(state)} steps", flush=True)

    temporary = prepared_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(prepared_dir / "manifest.json")


class EpisodeDataset:
    def __init__(self, prepared_dir):
        self.root = Path(prepared_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.episodes = self.manifest["episodes"]

    def _episode_dir(self, episode_id):
        return self.root / f"episode_{episode_id:07d}"

    def _load(self, episode_id, name):
        return np.load(self._episode_dir(episode_id) / f"{name}.npy", mmap_mode="c", allow_pickle=False)

    def training_arrays(self):
        return {
            STATE_KEY: np.concatenate([self._load(ep["id"], "state") for ep in self.episodes]),
            "action": np.concatenate([self._load(ep["id"], "action") for ep in self.episodes]),
        }

    def normalizer(self):
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer

        normalizer = LinearNormalizer()
        normalizer.fit(self.training_arrays(), last_n_dims=1, mode="limits")
        normalizer[CAMERA_KEY] = get_image_range_normalizer()
        return normalizer

    @staticmethod
    def _anchor_count(length, horizon, n_action_steps):
        # Stanford SequenceSampler: pad_before=1 and pad_after=n_action_steps-1.
        return max(length - horizon + n_action_steps + 1, 0)

    def num_samples(self, horizon, n_action_steps):
        return sum(self._anchor_count(ep["steps"], horizon, n_action_steps) for ep in self.episodes)

    def num_batches(self, batch_size, horizon, n_action_steps):
        if batch_size < 1 or not 1 <= n_action_steps < horizon:
            raise ValueError("Expected batch_size>=1 and 1<=n_action_steps<horizon")
        return sum(
            (self._anchor_count(ep["steps"], horizon, n_action_steps) + batch_size - 1) // batch_size
            for ep in self.episodes
        )

    def episode_observations(self, episode_id):
        return {
            CAMERA_KEY: torch.from_numpy(self._load(episode_id, "rgb")),
            STATE_KEY: torch.from_numpy(self._load(episode_id, "state")),
        }

    def batches(self, batch_size, horizon, n_action_steps, epoch, seed=0):
        self.num_batches(batch_size, horizon, n_action_steps)
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        for episode_index in generator.permutation(len(self.episodes)):
            episode = self.episodes[episode_index]
            episode_id, length = episode["id"], episode["steps"]
            observations = self.episode_observations(episode_id)
            actions = self._load(episode_id, "action")
            anchor_count = self._anchor_count(length, horizon, n_action_steps)
            for start in range(0, anchor_count, batch_size):
                stop = min(start + batch_size, anchor_count)
                end_of_episode = stop == anchor_count
                # Include terminal observations in the last segment even when they
                # cannot anchor another complete executed action chunk.
                obs_stop = length if end_of_episode else stop
                anchors = np.arange(start, stop)
                action_indices = anchors[:, None] - 1 + np.arange(horizon)[None, :]
                past_indices = anchors[:, None] - PAST_STEPS + np.arange(PAST_STEPS)[None, :]
                past_mask = past_indices >= 0
                past_actions = actions[past_indices.clip(0, length - 1)]
                past_actions[~past_mask] = 0
                yield {
                    "obs": {key: values[start:obs_stop] for key, values in observations.items()},
                    "previous_obs": {key: values[max(start - 1, 0):max(start - 1, 0) + 1]
                                     for key, values in observations.items()},
                    "anchors": torch.arange(stop - start),
                    "action": torch.from_numpy(actions[action_indices.clip(0, length - 1)]),
                    "past_action": torch.from_numpy(past_actions),
                    "past_mask": torch.from_numpy(past_mask),
                    "episode_id": episode_id,
                    "segment_start": start,
                    "end_of_episode": end_of_episode,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("rmbench", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.prepared_dir, args.rmbench)
