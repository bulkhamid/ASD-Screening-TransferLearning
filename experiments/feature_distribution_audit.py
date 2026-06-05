#!/usr/bin/env python
"""Summarize simple feature and source distributions by class/split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.shortcut_baselines import global_stats_features, load_npz, motion_features


SPLITS = ("train", "val", "test")
GLOBAL_NAMES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "brightness_mean",
    "brightness_std",
    "dark_frac",
    "bright_frac",
    "gray_pad_frac",
]
MOTION_NAMES = ["diff_mean", "diff_std", "diff_p90", "diff_p99"]


def cohen_d(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    if pooled == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def summarize_feature_matrix(X, y, names):
    rows = []
    for idx, name in enumerate(names):
        pos = X[y == 1, idx]
        neg = X[y == 0, idx]
        rows.append(
            {
                "feature": name,
                "positive_mean": float(pos.mean()) if len(pos) else None,
                "negative_mean": float(neg.mean()) if len(neg) else None,
                "positive_std": float(pos.std(ddof=0)) if len(pos) else None,
                "negative_std": float(neg.std(ddof=0)) if len(neg) else None,
                "cohen_d_pos_minus_neg": cohen_d(pos, neg),
            }
        )
    return sorted(rows, key=lambda r: abs(r["cohen_d_pos_minus_neg"]), reverse=True)


def load_manifest(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def month_token(path):
    stem = Path(path).stem
    match = re.search(r"(20\d{2})[-_]?(\d{2})", stem)
    if not match:
        return "unknown"
    return f"{match.group(1)}-{match.group(2)}"


def label_counts(rows, key_fn):
    counts = defaultdict(lambda: Counter())
    for row in rows:
        label = "positive" if int(row["label"]) == 1 else "negative"
        counts[key_fn(row)][label] += 1
    return {
        str(k): {"positive": int(v["positive"]), "negative": int(v["negative"])}
        for k, v in sorted(counts.items())
    }


def manifest_summary(rows):
    out = {}
    for split in SPLITS:
        split_rows = [r for r in rows if r.get("split") == split]
        out[split] = {
            "extension_counts": label_counts(split_rows, lambda r: Path(r["path"]).suffix.lower() or "none"),
            "month_counts": label_counts(split_rows, lambda r: month_token(r["path"])),
            "total_frames_by_label": numeric_by_label(split_rows, "total_frames"),
            "failed_reads_by_label": numeric_by_label(split_rows, "failed_reads"),
        }
    return out


def numeric_by_label(rows, field):
    out = {}
    for label_value, label_name in [(1, "positive"), (0, "negative")]:
        vals = [float(r[field]) for r in rows if int(r["label"]) == label_value and r.get(field) not in ("", None)]
        out[label_name] = {
            "count": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
        }
    return out


def evaluate(args):
    data_dir = Path(args.data_dir)
    split_features = {}
    for split in SPLITS:
        X, y = load_npz(data_dir, split)
        features = np.concatenate([global_stats_features(X), motion_features(X)], axis=1)
        split_features[split] = {
            "count": int(len(y)),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "features": summarize_feature_matrix(features, y, GLOBAL_NAMES + MOTION_NAMES),
        }

    rows = load_manifest(Path(args.manifest))
    return {
        "data_dir": str(data_dir),
        "manifest": args.manifest,
        "feature_summary": split_features,
        "manifest_summary": manifest_summary(rows) if rows else {},
    }


def print_summary(output, top_n):
    print("Feature distribution audit")
    for split, data in output["feature_summary"].items():
        print(f"\n{split}: n={data['count']} pos={data['positive']} neg={data['negative']}")
        print(f"{'feature':20s} {'pos_mean':>10s} {'neg_mean':>10s} {'cohen_d':>9s}")
        for row in data["features"][:top_n]:
            print(
                f"{row['feature']:20s} "
                f"{row['positive_mean']:10.4f} "
                f"{row['negative_mean']:10.4f} "
                f"{row['cohen_d_pos_minus_neg']:9.3f}"
            )

    manifest = output.get("manifest_summary", {})
    if manifest:
        print("\nManifest source/session hints")
        for split, data in manifest.items():
            print(f"\n{split} extension counts: {data['extension_counts']}")
            print(f"{split} month counts: {data['month_counts']}")
            print(f"{split} total frames: {data['total_frames_by_label']}")
            print(f"{split} failed reads: {data['failed_reads_by_label']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--manifest", default="data/split_manifest.csv")
    parser.add_argument("--output", default="experiments/feature_distribution_audit_results.json")
    parser.add_argument("--top_n", type=int, default=8)
    args = parser.parse_args()

    output = evaluate(args)
    print_summary(output, args.top_n)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
