import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from src.utils.subject_training import train_one_run


class SubjectBenchmarkEndToEndTests(unittest.TestCase):
    def test_synthetic_grouped_run_writes_traceable_artifacts(self):
        root = Path("tmp_tests_workspace") / f"grouped_e2e_{uuid.uuid4().hex}"
        cache_root = root / "cache"
        (cache_root / "clips").mkdir(parents=True)
        (cache_root / "masks").mkdir()
        rows = []
        for label, prefix, value in ((0, "negative", 30), (1, "positive", 220)):
            for index in range(3):
                subject = f"{prefix}-{index}"
                video_id = f"video-{subject}"
                clips = np.full((5, 16, 16, 16, 3), value, dtype=np.uint8)
                masks = np.ones(clips.shape[:-1], dtype=bool)
                clip_path = Path("clips") / f"{video_id}.npy"
                mask_path = Path("masks") / f"{video_id}.npy"
                np.save(cache_root / clip_path, clips)
                np.save(cache_root / mask_path, np.packbits(masks, axis=-1))
                rows.append(
                    {
                        "video_id": video_id,
                        "label": str(label),
                        "subject_id": subject,
                        "session_id": f"session-{subject}",
                        "cache_path": clip_path.as_posix(),
                        "mask_path": mask_path.as_posix(),
                    }
                )

        result = train_one_run(
            model_name="simple2dcnn",
            model_config={
                "max_epochs": 1,
                "patience": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.0,
            },
            protocol_config={"samples_per_subject": 1, "batch_size": 2, "num_workers": 0},
            rows=rows,
            cache_root=cache_root,
            train_subjects={"negative-0", "positive-0"},
            val_subjects={"negative-1", "positive-1"},
            test_subjects={"negative-2", "positive-2"},
            seed=42,
            fold=1,
            run_dir=root / "run",
            device=torch.device("cpu"),
            pretrained=False,
            temporal_ablations=("normal",),
        )

        self.assertEqual(result["threshold"], 0.5)
        self.assertFalse(result["temperature_scaling"])
        self.assertTrue((root / "run" / "best_checkpoint.pth").exists())
        self.assertTrue((root / "run" / "predictions_normal.csv").exists())
        self.assertTrue((root / "run" / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
