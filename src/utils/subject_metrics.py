"""Hierarchical subject-level metrics for binary video screening."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def hierarchical_aggregate(clip_rows):
    videos = defaultdict(list)
    video_meta = {}
    for row in clip_rows:
        key = row["video_id"]
        if key in video_meta and any(
            row[field] != video_meta[key][field]
            for field in ("subject_id", "session_id", "label")
        ):
            raise ValueError("Video has conflicting subject, session, or label metadata")
        videos[key].append(float(row["logit"]))
        video_meta[key] = row

    sessions = defaultdict(list)
    session_meta = {}
    for video_id, logits in videos.items():
        row = video_meta[video_id]
        key = (row["subject_id"], row["session_id"])
        sessions[key].append(float(np.mean(logits)))
        session_meta[key] = row

    subjects = defaultdict(list)
    subject_labels = defaultdict(set)
    for key, logits in sessions.items():
        row = session_meta[key]
        subjects[row["subject_id"]].append(float(np.mean(logits)))
        subject_labels[row["subject_id"]].add(int(row["label"]))

    output = []
    for subject_id, logits in sorted(subjects.items()):
        labels = subject_labels[subject_id]
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id!r} has inconsistent labels")
        output.append(
            {
                "subject_id": subject_id,
                "label": next(iter(labels)),
                "logit": float(np.mean(logits)),
                "sessions": len(logits),
            }
        )
    return output


def sigmoid(logits):
    logits = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_subject_metrics(subject_rows):
    if not subject_rows:
        raise ValueError("Subject metrics require at least one prediction")
    y = np.asarray([int(row["label"]) for row in subject_rows])
    logits = np.asarray([float(row["logit"]) for row in subject_rows], dtype=np.float64)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(logits).all():
        raise ValueError("Subject metrics require binary labels and finite logits")
    probabilities = sigmoid(logits)
    predictions = (logits >= 0.0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, logits)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, logits)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "nll": float(np.mean(np.logaddexp(0.0, logits) - y * logits)),
        "brier": float(np.mean(np.square(probabilities - y))),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "subjects": int(len(y)),
        "positive_subjects": int(y.sum()),
        "negative_subjects": int(len(y) - y.sum()),
        "threshold": 0.5,
    }


def stratified_bootstrap_ci(subject_rows, *, samples: int = 2000, seed: int = 42):
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    by_label = {
        label: [row for row in subject_rows if int(row["label"]) == label]
        for label in (0, 1)
    }
    if not by_label[0] or not by_label[1]:
        raise ValueError("Bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    scalar_keys = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity",
        "specificity", "precision", "f1", "nll", "brier",
    ]
    for _ in range(samples):
        sampled = []
        for label in (0, 1):
            indices = rng.integers(0, len(by_label[label]), size=len(by_label[label]))
            sampled.extend(by_label[label][index] for index in indices)
        metrics = compute_subject_metrics(sampled)
        for key in scalar_keys:
            draws[key].append(metrics[key])
    return {
        key: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }
