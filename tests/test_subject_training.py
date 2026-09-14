import unittest
import uuid
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn

from src.models.pretrained_r3d import PretrainedR3D
from experiments.run_subject_benchmark import validate_cache_config, validate_run_options
from src.utils.subject_training import GlobalCheckpointTracker, stage_schedule


class SubjectTrainingTests(unittest.TestCase):
    def test_nonfinite_loss_cannot_replace_best_checkpoint(self):
        output = Path("tmp_tests_workspace") / f"checkpoint_{uuid.uuid4().hex}" / "best.pth"
        tracker = GlobalCheckpointTracker(output)
        model = nn.Linear(2, 1)
        tracker.update(0.5, model, stage="head", epoch=1, metadata={})
        original = output.read_bytes()
        for loss in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                tracker.update(loss, model, stage="full", epoch=2, metadata={})
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(tracker.best_loss, 0.5)

    def test_inconsistent_experiment_options_fail_before_training(self):
        config = {"training_seeds": [42], "temperature_scaling": False, "threshold": 0.5}
        options = dict(models=None, seeds=None, folds=None, temporal_ablations=["normal"], epoch_limit=None)
        for change in ({"seeds": [42, 42]}, {"temporal_ablations": ["static"]}, {"epoch_limit": 0}):
            with self.assertRaises(ValueError):
                validate_run_options(config, SimpleNamespace(**{**options, **change}))
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_run_options({**config, "temperature_scaling": True}, SimpleNamespace(**options))

    def test_primary_protocol_rejects_mismatched_cache(self):
        protocol = {"clip_cache_version": 2}
        cache = {
            "version": 2,
            "clips_per_video": 5,
            "clip_len": 60,
            "sample_fps": 15.0,
            "target_size": [112, 112],
            "pad_value": 128,
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_cache_config(cache, protocol)

    def test_r3d_stage_budgets_match_protocol(self):
        config = {
            "stages": [
                {"name": "head", "epochs": 15, "learning_rate": 1e-3},
                {"name": "layer4", "epochs": 20, "learning_rate": 1e-4},
                {"name": "full", "epochs": 8, "learning_rate": 1e-5},
            ]
        }

        stages = stage_schedule("r3d", config)

        self.assertEqual([stage["epochs"] for stage in stages], [15, 20, 8])
        self.assertEqual([stage["learning_rate"] for stage in stages], [1e-3, 1e-4, 1e-5])

    def test_r3d_batchnorm_stays_frozen_in_train_mode(self):
        model = PretrainedR3D(pretrained=False, freeze_batchnorm=True)
        model.configure_stage("full")
        model.train()
        batchnorms = [
            module for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]

        self.assertTrue(batchnorms)
        self.assertTrue(all(not module.training for module in batchnorms))
        self.assertTrue(
            all(not parameter.requires_grad for module in batchnorms for parameter in module.parameters())
        )

    def test_global_checkpoint_keeps_best_across_stages(self):
        output = Path("tmp_tests_workspace") / f"checkpoint_{uuid.uuid4().hex}" / "best.pth"
        model = nn.Linear(2, 1)
        tracker = GlobalCheckpointTracker(output)

        first = tracker.update(0.5, model, stage="head", epoch=1, metadata={})
        second = tracker.update(0.7, model, stage="layer4", epoch=2, metadata={})
        third = tracker.update(0.4, model, stage="full", epoch=3, metadata={})

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(third)
        self.assertEqual(tracker.best_stage, "full")
        payload = torch.load(output, weights_only=False)
        self.assertEqual(payload["validation_loss"], 0.4)


if __name__ == "__main__":
    unittest.main()
