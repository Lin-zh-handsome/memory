"""Validate the actual task's data and policy interface before training."""
import argparse
import io
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import h5py
import numpy as np
import torch

from data import EpisodeDataset, _read_expert, observation_meta, preprocess_rgb
from models import build_policy
from test_models import check_event_bank, check_mamba, check_policy
from train import configure_precision, move_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("rmbench", type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.root / "configs/protocol.json").read_text())
    task = cfg["task"]
    dataset = EpisodeDataset(args.root / "prepared")
    assert dataset.manifest["task"] == task and len(dataset.episodes) == 50
    assert torch.cuda.device_count() == 1
    sys.path.insert(0, str(args.rmbench))
    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ImportError:
        decoder = importlib.util.spec_from_file_location("official_rgb_decoder", args.rmbench / "data/decode_image_bit.py")
        module = importlib.util.module_from_spec(decoder)
        decoder.loader.exec_module(module)
        decode_image_bit = module.decode_image_bit

    archive = args.root / f"archive/data/{task}/demo_clean.zip"
    with ZipFile(archive) as zipped:
        for index in (0, 25, 49):
            entry = dataset.episodes[index]
            with h5py.File(io.BytesIO(zipped.read(entry["archive_member"])), "r") as demo:
                rgb, state, action, metadata = _read_expert(demo)
                assert metadata["layout"] == "XPolicyLab"
                assert np.array_equal(state, dataset._load(entry["id"], "state"))
                assert np.array_equal(action, dataset._load(entry["id"], "action"))
                expected = dataset._load(entry["id"], "rgb")
                for t in (0, len(state) // 2, len(state) - 1):
                    frame = preprocess_rgb(decode_image_bit(rgb[t]))
                    assert frame.shape == (3, 96, 96) and np.array_equal(frame, expected[t])
                    assert frame.std() > 0

    assert all(entry["action_matches_next_state"] for entry in dataset.episodes)
    phase_key = "official_cover_phases" if task == "cover_blocks" else "official_battery_phases"
    assert all(entry[phase_key] for entry in dataset.episodes)
    batches = dataset.batches(cfg["batch_size"], cfg["horizon"], cfg["n_action_steps"], epoch=1, seed=0)
    first, second = next(batches), next(batches)
    assert first["episode_id"] == second["episode_id"] and first["segment_start"] == 0
    assert second["segment_start"] == cfg["batch_size"]
    actions = dataset._load(first["episode_id"], "action")
    assert first["action"].shape == (cfg["batch_size"], cfg["horizon"], 14)
    np.testing.assert_array_equal(first["action"][0, 1].numpy(), actions[0])
    np.testing.assert_array_equal(second["past_action"][0].numpy(), actions[cfg["batch_size"]-8:cfg["batch_size"]])
    assert not first["past_mask"][0].any() and second["past_mask"][0].all()

    torch.set_num_threads(2)
    torch.manual_seed(0)
    configure_precision()
    normalizer = dataset.normalizer()
    for key, values in dataset.training_arrays().items():
        tensor = torch.from_numpy(values[::113])
        normalized = normalizer[key].normalize(tensor)
        assert torch.isfinite(normalized).all()
        torch.testing.assert_close(normalizer[key].unnormalize(normalized), tensor, atol=1e-5, rtol=1e-4)
    for model in ("M0", "M3"):
        policy = build_policy(model, observation_meta(cfg), cfg)
        policy.set_normalizer(normalizer)
        policy = policy.cuda().train()
        batch = move_batch(first, model)
        value = policy.compute_loss(batch)
        loss = value[0] if model == "M3" else value
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                   for p in policy.model.parameters())
        print(json.dumps({"model": model, "real_batch_loss": float(loss.detach())}), flush=True)
        del loss, value, policy, batch
        torch.cuda.empty_cache()

    device = torch.device("cuda")
    check_mamba(device)
    check_event_bank()
    check_policy(device, variants=("M3",))
    print(json.dumps({"task": task, "episodes": 50, "preflight_passed": True}), flush=True)


if __name__ == "__main__":
    main()
