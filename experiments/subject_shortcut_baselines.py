#!/usr/bin/env python
"""Fixed grouped shortcut baselines for the completed subject clip cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.shortcut_baselines import (
    global_stats_features,
    motion_features,
    thumbnail_features,
)
from src.utils.subject_data import make_subject_folds, read_cache_manifest
from src.utils.subject_metrics import (
    compute_subject_metrics,
    hierarchical_aggregate,
    stratified_bootstrap_ci,
)


def metadata_features(row):
    extension = Path(row.get("relative_path", "")).suffix.lower()
    timestamp = row.get("capture_datetime", "")
    try:
        hour = datetime.fromisoformat(timestamp).hour
    except ValueError:
        hour = 0
    width = float(row.get("width") or 0)
    height = float(row.get("height") or 0)
    return np.asarray(
        [
            float(row.get("total_frames") or 0),
            float(row.get("fps") or 0),
            float(row.get("duration_seconds") or 0),
            width,
            height,
            width / height if height else 0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            float(extension == ".mp4"),
            float(extension == ".webm"),
            float(extension == ".avi"),
        ],
        dtype=np.float32,
    )


def extract_feature_families(rows, cache_root: Path, *, thumb_size=16, thumb_frames=5):
    metadata, appearance, motion, thumbnails = [], [], [], []
    for row in rows:
        clips = np.load(cache_root / row["cache_path"], mmap_mode="r")
        metadata.append(metadata_features(row))
        appearance.append(global_stats_features(clips).mean(axis=0))
        motion.append(motion_features(clips).mean(axis=0))
        thumbnails.append(
            thumbnail_features(clips, size=thumb_size, frames=thumb_frames).mean(axis=0)
        )
    return {
        "duration_resolution_extension_capture_time": np.stack(metadata),
        "color_brightness_padding": np.stack(appearance),
        "frame_difference_motion": np.stack(motion),
        "tiny_thumbnails": np.stack(thumbnails),
    }


def evaluate_family(features, rows, folds, *, seed):
    labels = np.asarray([int(row["label"]) for row in rows])
    subjects = np.asarray([row["subject_id"] for row in rows])
    video_counts = Counter(subjects.tolist())
    oof_rows = []
    for fold in folds:
        train_indices = np.flatnonzero(np.isin(subjects, list(fold["outer_train"])))
        test_indices = np.flatnonzero(np.isin(subjects, list(fold["outer_test"])))
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
                max_iter=2000,
            ),
        )
        sample_weights = np.asarray(
            [1.0 / video_counts[subjects[index]] for index in train_indices]
        )
        classifier.fit(
            features[train_indices],
            labels[train_indices],
            logisticregression__sample_weight=sample_weights,
        )
        logits = classifier.decision_function(features[test_indices])
        for index, logit in zip(test_indices, logits):
            row = rows[index]
            oof_rows.append(
                {
                    "video_id": row["video_id"],
                    "session_id": row["session_id"],
                    "subject_id": row["subject_id"],
                    "label": int(row["label"]),
                    "logit": float(logit),
                }
            )
    subject_rows = hierarchical_aggregate(oof_rows)
    return subject_rows, compute_subject_metrics(subject_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--thumb_size", type=int, default=16)
    parser.add_argument("--thumb_frames", type=int, default=5)
    parser.add_argument("--output", default="runs/subject_shortcut_baselines.json")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    rows = read_cache_manifest(Path(args.manifest or cache_root / "cache_manifest.csv"))
    folds = make_subject_folds(rows, max_folds=args.max_folds, split_seed=args.seed)
    feature_families = extract_feature_families(
        rows, cache_root, thumb_size=args.thumb_size, thumb_frames=args.thumb_frames
    )
    results = {}
    for name, features in feature_families.items():
        subject_rows, metrics = evaluate_family(features, rows, folds, seed=args.seed)
        results[name] = {
            "metrics": metrics,
            "bootstrap_95_ci": stratified_bootstrap_ci(
                subject_rows, samples=args.bootstrap_samples, seed=args.seed
            ),
        }
    output = {
        "unit": "subject",
        "threshold": 0.5,
        "folds": len(folds),
        "results": results,
        "interpretation": "Shortcut diagnostic; not a clinical model.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
