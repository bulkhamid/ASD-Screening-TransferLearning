#!/usr/bin/env python
"""Evaluate recording shortcuts while holding capture-date groups out."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.shortcut_baselines import (
    global_stats_features,
    motion_features,
    thumbnail_features,
)
from data.save_numpy_dataset import array_hash, file_hash


SPLITS = ("train", "val", "test")


def read_legacy_data(data_dir: Path, manifest_path: Path, *, manifest_git_ref=None):
    if manifest_git_ref:
        result = subprocess.run(
            ["git", "show", manifest_git_ref], check=True, capture_output=True, text=True,
        )
        manifest = list(csv.DictReader(io.StringIO(result.stdout)))
    else:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
    arrays = []
    labels = []
    rows = []
    for split in SPLITS:
        with np.load(data_dir / f"{split}.npz") as data:
            videos, cached_labels = data["X"], data["y"].astype(int)
        split_rows = [row for row in manifest if row.get("split") == split]
        if len(split_rows) != len(cached_labels):
            raise ValueError(f"{split}: manifest/cache row count mismatch")
        manifest_labels = np.asarray([int(row["label"]) for row in split_rows])
        if not np.array_equal(manifest_labels, cached_labels):
            raise ValueError(f"{split}: manifest/cache label order mismatch")
        for index, (row, video) in enumerate(zip(split_rows, videos)):
            if not row.get("decoded_hash") or array_hash(video) != row["decoded_hash"]:
                raise ValueError(f"{split} row {index}: decoded hash does not match manifest")
            if not row.get("raw_hash") or file_hash(row["path"]) != row["raw_hash"]:
                raise ValueError(f"{split} row {index}: raw hash does not match manifest")
        arrays.append(videos)
        labels.append(manifest_labels)
        rows.extend(split_rows)
    return np.concatenate(arrays), np.concatenate(labels), rows


def capture_group(path: str, granularity: str) -> str:
    stem = Path(path).stem
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    if not match:
        raise ValueError("Missing capture date; cannot safely assign a recording group")
    year, month, day = match.groups()
    return f"{year}-{month}" if granularity == "month" else f"{year}-{month}-{day}"


def raw_metadata_features(rows):
    values = []
    extensions = sorted({Path(row["path"]).suffix.lower() for row in rows})
    for row in rows:
        cap = cv2.VideoCapture(row["path"])
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        fps = fps if np.isfinite(fps) and fps > 0 else 0.0
        duration = total / fps if fps else 0.0
        stem = Path(row["path"]).stem
        match = re.search(r"20\d{6}([0-2]\d)([0-5]\d)([0-5]\d)", stem)
        hour = int(match.group(1)) if match else 0
        extension = Path(row["path"]).suffix.lower()
        values.append(
            [
                total,
                fps,
                duration,
                width,
                height,
                width / height if height else 0.0,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                *[float(extension == candidate) for candidate in extensions],
            ]
        )
    return np.asarray(values, dtype=np.float32)


def grouped_oof(features, labels, groups, *, folds: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(len(labels), np.nan, dtype=float)
    fold_info = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(features, labels, groups), 1
    ):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        if train_groups & test_groups:
            raise RuntimeError("Capture group leakage detected")
        if len(np.unique(labels[train_indices])) < 2:
            classifier = DummyClassifier(strategy="prior")
        else:
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
        classifier.fit(features[train_indices], labels[train_indices])
        prediction = classifier.predict_proba(features[test_indices])
        if prediction.shape[1] == 1:
            class_value = int(classifier.classes_[0])
            probabilities[test_indices] = float(class_value)
        else:
            positive_column = list(classifier.classes_).index(1)
            probabilities[test_indices] = prediction[:, positive_column]
        fold_info.append(
            {
                "fold": fold,
                "train_videos": int(len(train_indices)),
                "test_videos": int(len(test_indices)),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "test_positive": int(labels[test_indices].sum()),
                "test_negative": int(len(test_indices) - labels[test_indices].sum()),
            }
        )
    if not bool(np.isfinite(probabilities).all()):
        raise RuntimeError("Some videos did not receive an out-of-fold prediction")
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "folds": fold_info,
    }


def evaluate(args):
    X, labels, rows = read_legacy_data(
        Path(args.data_dir), Path(args.manifest),
        manifest_git_ref=getattr(args, "manifest_git_ref", None),
    )
    groups = np.asarray([capture_group(row["path"], args.group) for row in rows])
    unique_groups = sorted(set(groups))
    folds = min(args.folds, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two capture groups are required")
    group_labels = {
        group: sorted(set(labels[groups == group].tolist())) for group in unique_groups
    }
    confounded = sum(len(values) == 1 for values in group_labels.values())
    feature_sets = {
        "duration_resolution_extension_capture_time": raw_metadata_features(rows),
        "color_brightness_padding": global_stats_features(X),
        "frame_difference_motion": motion_features(X),
        "tiny_thumbnails": thumbnail_features(X, size=args.thumb_size, frames=args.thumb_frames),
    }
    results = {
        name: grouped_oof(values, labels, groups, folds=folds, seed=args.seed)
        for name, values in feature_sets.items()
    }
    return {
        "audit_scope": "Legacy caches treated as development/audit data",
        "grouping": args.group,
        "videos": int(len(labels)),
        "capture_groups": len(unique_groups),
        "single_class_capture_groups": confounded,
        "warning": (
            "Capture groups are label-confounded; grouped results diagnose recording shortcuts "
            "but cannot replace subject-independent evaluation."
        ),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_source_aware")
    parser.add_argument("--manifest", default="data_source_aware/split_manifest.csv")
    parser.add_argument(
        "--manifest_git_ref", default=None,
        help="Read a historical manifest in memory, e.g. HEAD:data_source_aware/split_manifest.csv",
    )
    parser.add_argument("--group", choices=["date", "month"], default="date")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb_size", type=int, default=16)
    parser.add_argument("--thumb_frames", type=int, default=5)
    parser.add_argument("--output", default="experiments/grouped_shortcut_date_results.json")
    args = parser.parse_args()
    try:
        output = evaluate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
