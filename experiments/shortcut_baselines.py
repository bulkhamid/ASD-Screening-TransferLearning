#!/usr/bin/env python
"""Run deliberately weak baselines to probe shortcut learning.

These checks are not meant to beat the deep models. They answer a narrower
question: can low-level image statistics, filenames, or coarse thumbnails
separate the current split? If yes, perfect neural-network results need more
skepticism.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPLITS = ("train", "val", "test")


def load_npz(data_dir: Path, split: str):
    data = np.load(data_dir / f"{split}.npz", mmap_mode="r")
    return data["X"], data["y"].astype(int)


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for split in SPLITS:
        out[split] = [r for r in rows if r.get("split") == split]
    return out


def metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = 0.5
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def fit_eval_numeric(name, X_train, y_train, X_val, y_val, X_test, y_test, C=1.0):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="liblinear", random_state=42, max_iter=1000),
    )
    clf.fit(X_train, y_train)
    return {
        "name": name,
        "val": metrics(y_val, clf.predict_proba(X_val)[:, 1]),
        "test": metrics(y_test, clf.predict_proba(X_test)[:, 1]),
    }


def fit_eval_text(name, train_text, y_train, val_text, y_val, test_text, y_test):
    clf = make_pipeline(
        CountVectorizer(token_pattern=r"(?u)\b[\w.-]+\b", lowercase=True, min_df=1),
        LogisticRegression(C=1.0, solver="liblinear", random_state=42, max_iter=1000),
    )
    clf.fit(train_text, y_train)
    return {
        "name": name,
        "val": metrics(y_val, clf.predict_proba(val_text)[:, 1]),
        "test": metrics(y_test, clf.predict_proba(test_text)[:, 1]),
    }


def global_stats_features(X):
    Xf = X.astype(np.float32) / 255.0
    mean_rgb = Xf.mean(axis=(1, 2, 3))
    std_rgb = Xf.std(axis=(1, 2, 3))
    gray = Xf.mean(axis=4)
    brightness_mean = gray.mean(axis=(1, 2, 3))[:, None]
    brightness_std = gray.std(axis=(1, 2, 3))[:, None]
    dark_frac = (gray < 0.08).mean(axis=(1, 2, 3))[:, None]
    bright_frac = (gray > 0.92).mean(axis=(1, 2, 3))[:, None]
    gray_pad_frac = (np.abs(Xf - (128 / 255.0)).max(axis=4) < (1 / 255.0)).mean(axis=(1, 2, 3))[:, None]
    return np.concatenate(
        [mean_rgb, std_rgb, brightness_mean, brightness_std, dark_frac, bright_frac, gray_pad_frac],
        axis=1,
    )


def motion_features(X):
    Xf = X.astype(np.float32) / 255.0
    gray = Xf.mean(axis=4)
    diffs = np.abs(np.diff(gray, axis=1))
    return np.stack(
        [
            diffs.mean(axis=(1, 2, 3)),
            diffs.std(axis=(1, 2, 3)),
            np.quantile(diffs.reshape(len(diffs), -1), 0.90, axis=1),
            np.quantile(diffs.reshape(len(diffs), -1), 0.99, axis=1),
        ],
        axis=1,
    )


def thumbnail_features(X, size=16, frames=5):
    frame_ids = np.linspace(0, X.shape[1] - 1, frames).round().astype(int)
    feats = []
    for video in X:
        selected = []
        for idx in frame_ids:
            small = cv2.resize(video[idx], (size, size), interpolation=cv2.INTER_AREA)
            selected.append(small.astype(np.float32).reshape(-1) / 255.0)
        feats.append(np.concatenate(selected))
    return np.stack(feats)


def path_text_features(manifest_rows):
    texts = {}
    for split in SPLITS:
        rows = manifest_rows.get(split, [])
        split_text = []
        for row in rows:
            path = Path(row.get("path", ""))
            stem = path.stem
            tokens = [path.suffix.lower().lstrip(".")]
            tokens.extend(re.findall(r"\d{4,}", stem))
            split_text.append(" ".join(tokens))
        texts[split] = split_text
    return texts


def evaluate(args):
    data_dir = Path(args.data_dir)
    X = {}
    y = {}
    for split in SPLITS:
        X[split], y[split] = load_npz(data_dir, split)

    results = []

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y["train"]), 1)), y["train"])
    results.append(
        {
            "name": "majority_prior",
            "val": metrics(y["val"], dummy.predict_proba(np.zeros((len(y["val"]), 1)))[:, 1]),
            "test": metrics(y["test"], dummy.predict_proba(np.zeros((len(y["test"]), 1)))[:, 1]),
        }
    )

    feature_sets = {
        "global_color_brightness_padding": global_stats_features,
        "frame_difference_motion_stats": motion_features,
        "tiny_thumbnail_static_frames": lambda arr: thumbnail_features(arr, size=args.thumb_size, frames=args.thumb_frames),
    }
    for name, extractor in feature_sets.items():
        feats = {split: extractor(X[split]) for split in SPLITS}
        results.append(fit_eval_numeric(name, feats["train"], y["train"], feats["val"], y["val"], feats["test"], y["test"]))

    manifest_rows = load_manifest(Path(args.manifest))
    if all(len(manifest_rows.get(split, [])) == len(y[split]) for split in SPLITS):
        texts = path_text_features(manifest_rows)
        results.append(
            fit_eval_text("path_suffix_date_tokens", texts["train"], y["train"], texts["val"], y["val"], texts["test"], y["test"])
        )

    output = {
        "data_dir": str(data_dir),
        "manifest": args.manifest,
        "splits": {
            split: {
                "count": int(len(y[split])),
                "positive": int(y[split].sum()),
                "negative": int((1 - y[split]).sum()),
            }
            for split in SPLITS
        },
        "results": results,
    }
    return output


def print_table(output):
    print("Shortcut baseline audit")
    for split, info in output["splits"].items():
        print(f"{split:5s}: n={info['count']:3d} pos={info['positive']:3d} neg={info['negative']:3d}")
    print()
    print(f"{'baseline':32s} {'val_auc':>7s} {'val_f1':>7s} {'test_auc':>8s} {'test_acc':>8s} {'test_f1':>7s}")
    for row in output["results"]:
        val = row["val"]
        test = row["test"]
        print(
            f"{row['name'][:32]:32s} "
            f"{val['auc']:7.3f} {val['f1']:7.3f} "
            f"{test['auc']:8.3f} {test['accuracy']:8.3f} {test['f1']:7.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--manifest", default="data/split_manifest.csv")
    parser.add_argument("--output", default="experiments/shortcut_baselines_results.json")
    parser.add_argument("--thumb_size", type=int, default=16)
    parser.add_argument("--thumb_frames", type=int, default=5)
    args = parser.parse_args()

    output = evaluate(args)
    print_table(output)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
