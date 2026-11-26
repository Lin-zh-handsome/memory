"""Regression checks for the RGB range, missing checkpoints, and RNG fixes."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import Normalize

from data import CAMERA_KEY, EpisodeDataset
import run_ablation
from task_diagnostics import initial_state, report_state, transition_key

IMPLEMENTATION = Path(__file__).resolve().parents[1]


class CorrectionsTest(unittest.TestCase):
    def test_corrected_and_historical_rgb_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"episodes": [{"id": 0, "steps": 3}]}))
            episode = root / "episode_0000000"
            episode.mkdir()
            values = np.arange(42, dtype=np.float32).reshape(3, 14)
            np.save(episode / "state.npy", values)
            np.save(episode / "action.npy", values)
            dataset = EpisodeDataset(root)
            legacy = dataset.normalizer()
            corrected = dataset.normalizer({"rgb_normalization": "identity", "imagenet_norm": True})
            image = torch.full((1, 3, 2, 2), 0.5)
            standardize = Normalize([.485, .456, .406], [.229, .224, .225])
            torch.testing.assert_close(corrected[CAMERA_KEY].normalize(image), image)
            torch.testing.assert_close(legacy[CAMERA_KEY].normalize(image), torch.zeros_like(image))
            torch.testing.assert_close(standardize(corrected[CAMERA_KEY].normalize(image)), standardize(image))
            # Checkpoint loading restores the saved statistics over the new defaults.
            corrected.load_state_dict(legacy.state_dict())
            torch.testing.assert_close(corrected[CAMERA_KEY].normalize(image), torch.zeros_like(image))
            with self.assertRaises(ValueError):
                dataset.normalizer({"rgb_normalization": "invalid"})

    def test_missing_pending_checkpoint_never_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "configs").mkdir()
            (root / "logs/M5_complete.json").write_text('{"epoch":250}')
            spec = SimpleNamespace(root=root, model="M5")
            for split in (True, False):
                with self.assertRaisesRegex(FileNotFoundError, "epoch_050.pt"):
                    list(run_ablation.jobs([spec], split))
            (root / "logs/M5_complete.json").unlink()
            self.assertEqual([job[0] for job in run_ablation.jobs([spec], True)], ["train"])

    def test_completed_evaluations_can_survive_checkpoint_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "logs/M5_complete.json").write_text('{"epoch":250}')
            for epoch in (50,100,150,200,250):
                folder = root / f"evaluation/M5/epoch_{epoch:03d}"
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps({
                    "complete": True, "episodes": 50, "model": "M5", "epoch": epoch}))
            self.assertEqual(list(run_ablation.jobs([SimpleNamespace(root=root, model="M5")], True)), [])

    def test_scheduler_reports_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "logs/M5_complete.json").write_text('{"epoch":250}')
            queue = root / "queue.json"
            queue.write_text(json.dumps({
                "jobs": [{"root": str(root), "model": "M5"}],
                "fully_prepared": True}))
            args = SimpleNamespace(queue=queue, train_slots=2, gpu=[], split_evaluations=True)
            with patch.object(run_ablation, "priority", return_value=0), \
                 patch.object(run_ablation, "cards", return_value=[]), \
                 patch.object(run_ablation, "compute_processes", return_value={}):
                with self.assertRaises(FileNotFoundError):
                    run_ablation.run(args, None)
            status = json.loads((root / "logs/campaign_status.json").read_text())
            self.assertEqual(status["status"], "blocked")
            self.assertEqual(status["reason"], "missing_artifact")

    def test_task_diagnostics_read_values_without_advancing_task(self):
        def actor(position):
            pose = SimpleNamespace(p=np.array(position, dtype=float), q=np.array([1.,0,0,0]))
            return SimpleNamespace(get_pose=lambda: pose)
        env = SimpleNamespace(
            block1=actor([.1,0,.76]), block2=actor([.2,0,.76]),
            target_pose1=SimpleNamespace(p=np.array([.3,0,.76])),
            target_pose2=SimpleNamespace(p=np.array([.1,0,.76])),
            stage_id=1, press_cnt=1, take_action_cnt=10)
        data = initial_state("rearrange_blocks", env, np.zeros(14))
        state = report_state("rearrange_blocks", env)
        json.dumps(data)
        json.dumps(state)
        self.assertEqual(state["stage"], 1)
        self.assertEqual(env.stage_id, 1)
        self.assertEqual(env.press_cnt, 1)
        self.assertNotEqual(transition_key(state), transition_key({**state, "button_presses": 2}))

    def test_old_coffee_evaluation_preserves_training_rng(self):
        sys.path.insert(0, str(IMPLEMENTATION))
        spec = importlib.util.spec_from_file_location("coffee_eval_rng_regression", IMPLEMENTATION / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy = SimpleNamespace(device=torch.device("cpu"))
        random.seed(71)
        np.random.seed(72)
        torch.manual_seed(73)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        def sampled(*args):
            return random.random(), float(np.random.random()), torch.rand(3)

        def assert_preserved():
            self.assertEqual(random.getstate(), python_state)
            actual = np.random.get_state()
            np.testing.assert_array_equal(actual[1], numpy_state[1])
            self.assertEqual(actual[2:], numpy_state[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

        with patch.object(module, "_evaluate", side_effect=sampled):
            first = module.evaluate(policy, ".", "M0", 0, 50)
            assert_preserved()
            second = module.evaluate(policy, ".", "M0", 0, 50)
            self.assertEqual(first[:2], second[:2])
            self.assertTrue(torch.equal(first[2], second[2]))
            assert_preserved()
        with patch.object(module, "_evaluate", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                module.evaluate(policy, ".", "M0", 0, 50)
            assert_preserved()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

