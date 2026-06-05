#!/usr/bin/env python
# coding: utf-8
"""
Create cached train / val / test .npz files from raw videos.

The script deduplicates by full-file hash before decoding and by decoded tensor
hash after preprocessing. It also writes CSV manifests so every cached example
can be traced back to its source path and split.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.video_utils import resize_with_letterbox

VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi")
SPLITS = ("train", "val", "test")


def file_hash(path, algo="sha1", chunk_size=1024 * 1024):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array, algo="sha1"):
    h = hashlib.new(algo)
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def load_video(path, max_frames, target_size, unreadable_policy, pad_value=128):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total == 0:
        idxs = []
    elif total <= max_frames:
        idxs = list(range(total))
    else:
        idxs = sorted(random.sample(range(total), max_frames))

    frames = []
    failed_reads = 0
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_with_letterbox(frame, target_size, pad_value=pad_value)
        frames.append(frame)

    cap.release()

    if not frames:
        if unreadable_policy == "zeros":
            arr = np.zeros((max_frames, *target_size, 3), dtype=np.uint8)
            return arr, {
                "status": "unreadable_zero_filled",
                "total_frames": total,
                "decoded_frames": 0,
                "failed_reads": failed_reads,
            }
        return None, {
            "status": "excluded_unreadable",
            "total_frames": total,
            "decoded_frames": 0,
            "failed_reads": failed_reads,
        }

    if len(frames) < max_frames:
        pad = frames[-1]
        frames += [pad] * (max_frames - len(frames))

    arr = np.stack(frames).astype(np.uint8)
    return arr, {
        "status": "ok",
        "total_frames": total,
        "decoded_frames": min(len(idxs) - failed_reads, max_frames),
        "failed_reads": failed_reads,
    }


def crawl_videos(root):
    rows = []
    for label, folder in [(1, "positive"), (0, "negative")]:
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                rows.append({"path": path, "label": label})
    return rows


def write_manifest(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def capture_month(path):
    stem = Path(path).stem
    match = re.search(r"(20\d{2})[-_]?(\d{2})", stem)
    if not match:
        return "unknown"
    return f"{match.group(1)}-{match.group(2)}"


def add_source_fields(row):
    path = Path(row["path"])
    return {
        **row,
        "source_month": capture_month(path),
        "source_extension": path.suffix.lower() or "none",
    }


def split_counts(rows):
    return {
        "count": len(rows),
        "positive": sum(int(row["label"]) == 1 for row in rows),
        "negative": sum(int(row["label"]) == 0 for row in rows),
    }


def label_count_by(rows, field):
    out = defaultdict(lambda: Counter())
    for row in rows:
        label = "positive" if int(row["label"]) == 1 else "negative"
        out[str(row.get(field, "unknown"))][label] += 1
    return {
        key: {"positive": int(counts["positive"]), "negative": int(counts["negative"])}
        for key, counts in sorted(out.items())
    }


def numeric_summary(rows, field):
    out = {}
    for label_value, label_name in [(1, "positive"), (0, "negative")]:
        vals = [float(row[field]) for row in rows if int(row["label"]) == label_value]
        out[label_name] = {
            "count": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
        }
    return out


def source_distribution(rows, field):
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    total = len(rows) or 1
    return {key: count / total for key, count in counts.items()}


def distribution_l1(split_rows, all_rows, field):
    overall = source_distribution(all_rows, field)
    split = source_distribution(split_rows, field)
    keys = set(overall) | set(split)
    return sum(abs(split.get(k, 0.0) - overall.get(k, 0.0)) for k in keys)


def frame_mean_gap(split_rows, all_rows):
    if not split_rows or not all_rows:
        return 0.0
    all_mean = np.mean([float(row["total_frames"]) for row in all_rows])
    split_mean = np.mean([float(row["total_frames"]) for row in split_rows])
    scale = np.std([float(row["total_frames"]) for row in all_rows]) or 1.0
    return abs(split_mean - all_mean) / scale


def candidate_score(train_rows, val_rows, test_rows, all_rows):
    score = 0.0
    for rows in (train_rows, val_rows, test_rows):
        score += 4.0 * distribution_l1(rows, all_rows, "source_month")
        score += 2.0 * distribution_l1(rows, all_rows, "source_extension")
        score += frame_mean_gap(rows, all_rows)
    return float(score)


def split_random(rows, labels, args):
    train_rows, test_rows, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=args.test_frac,
        stratify=labels,
        random_state=args.seed,
    )
    val_ratio = args.val_frac / (1.0 - args.test_frac)
    train_rows, val_rows, y_train, y_val = train_test_split(
        train_rows,
        y_train,
        test_size=val_ratio,
        stratify=y_train,
        random_state=args.seed,
    )
    return train_rows, val_rows, test_rows


def split_source_aware(rows, labels, args):
    best = None
    best_score = math.inf
    val_ratio = args.val_frac / (1.0 - args.test_frac)
    for offset in range(args.source_aware_candidates):
        seed = args.seed + offset
        try:
            train_rows, test_rows, y_train, y_test = train_test_split(
                rows,
                labels,
                test_size=args.test_frac,
                stratify=labels,
                random_state=seed,
            )
            train_rows, val_rows, y_train, y_val = train_test_split(
                train_rows,
                y_train,
                test_size=val_ratio,
                stratify=y_train,
                random_state=seed,
            )
        except ValueError:
            continue
        score = candidate_score(train_rows, val_rows, test_rows, rows)
        if score < best_score:
            best_score = score
            best = (train_rows, val_rows, test_rows)
    if best is None:
        raise RuntimeError("Could not create a stratified source-aware split.")
    return best, best_score


def split_warnings(rows):
    warnings = []
    for field in ("source_month", "source_extension"):
        counts = label_count_by(rows, field)
        one_class = {
            key: value for key, value in counts.items()
            if (value["positive"] == 0 or value["negative"] == 0)
            and (value["positive"] + value["negative"]) > 1
        }
        if one_class:
            warnings.append(
                f"{field} is class-confounded for {one_class}; source-aware splitting can reduce but not eliminate shortcut bias."
            )
    return warnings


def write_split_diagnostics(output_dir, strategy, all_rows, split_rows, score=None):
    diagnostics = {
        "split_strategy": strategy,
        "source_aware_score": score,
        "warnings": split_warnings(all_rows),
        "overall": {
            "label_counts": split_counts(all_rows),
            "month_counts": label_count_by(all_rows, "source_month"),
            "extension_counts": label_count_by(all_rows, "source_extension"),
            "total_frames_by_label": numeric_summary(all_rows, "total_frames"),
            "failed_reads_by_label": numeric_summary(all_rows, "failed_reads"),
        },
        "splits": {},
    }
    for split, rows in split_rows:
        diagnostics["splits"][split] = {
            "label_counts": split_counts(rows),
            "month_counts": label_count_by(rows, "source_month"),
            "extension_counts": label_count_by(rows, "source_extension"),
            "total_frames_by_label": numeric_summary(rows, "total_frames"),
            "failed_reads_by_label": numeric_summary(rows, "failed_reads"),
        }
    with open(output_dir / "split_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size)

    raw_rows = crawl_videos(root)
    by_raw_hash = {}
    raw_clashes = defaultdict(list)
    raw_unique = []

    for row in raw_rows:
        raw_sha1 = file_hash(row["path"])
        row = {**row, "raw_hash": raw_sha1}
        if raw_sha1 not in by_raw_hash:
            by_raw_hash[raw_sha1] = row
            raw_unique.append(row)
        else:
            previous = by_raw_hash[raw_sha1]
            if previous["label"] != row["label"]:
                raw_clashes[raw_sha1].extend([previous, row])

    if raw_clashes:
        print("Conflicting raw-file duplicates found; keeping the first copy only:")
        for raw_sha1, items in raw_clashes.items():
            print(f"  raw_hash={raw_sha1}")
            for item in items:
                print(f"    {item['path'].name} label={item['label']}")

    decoded_unique = []
    skipped_rows = []
    by_decoded_hash = {}
    decoded_clashes = defaultdict(list)

    for row in raw_unique:
        video, meta = load_video(
            row["path"],
            max_frames=args.max_frames,
            target_size=target_size,
            unreadable_policy=args.unreadable_policy,
            pad_value=args.pad_value,
        )
        base = {
            "path": str(row["path"]),
            "label": row["label"],
            "raw_hash": row["raw_hash"],
            "decoded_hash": "",
            **meta,
        }
        if video is None:
            skipped_rows.append(base)
            continue

        decoded_sha1 = array_hash(video)
        base["decoded_hash"] = decoded_sha1
        base["is_all_zero"] = int(np.all(video == 0))

        if base["is_all_zero"] and not args.allow_all_zero:
            base["status"] = "excluded_all_zero"
            skipped_rows.append(base)
            continue

        if decoded_sha1 in by_decoded_hash:
            previous = by_decoded_hash[decoded_sha1]
            if previous["label"] != row["label"]:
                decoded_clashes[decoded_sha1].extend([previous, base])
            base["status"] = "excluded_decoded_duplicate"
            skipped_rows.append(base)
            continue

        base = add_source_fields(base)
        by_decoded_hash[decoded_sha1] = base
        decoded_unique.append({**base, "video": video})

    if decoded_clashes:
        print("Conflicting decoded-array duplicates found; keeping the first copy only:")
        for decoded_sha1, items in decoded_clashes.items():
            print(f"  decoded_hash={decoded_sha1}")
            for item in items:
                print(f"    {Path(item['path']).name} label={item['label']}")

    if not decoded_unique:
        raise RuntimeError("No readable unique videos found.")

    labels = [row["label"] for row in decoded_unique]
    split_score = None
    if args.split_strategy == "source_aware":
        (train_rows, val_rows, test_rows), split_score = split_source_aware(
            decoded_unique, labels, args
        )
    else:
        train_rows, val_rows, test_rows = split_random(decoded_unique, labels, args)

    split_rows = [("train", train_rows), ("val", val_rows), ("test", test_rows)]
    manifest_rows = []
    for split, rows in split_rows:
        X = np.stack([row["video"] for row in rows])
        y = np.array([row["label"] for row in rows], dtype=np.uint8)
        np.savez_compressed(output_dir / f"{split}.npz", X=X, y=y)
        print(f"{split}: {X.shape[0]} videos  (pos={int(y.sum())}, neg={len(y)-int(y.sum())})")
        for row in rows:
            manifest_rows.append({
                "split": split,
                "path": row["path"],
                "label": row["label"],
                "raw_hash": row["raw_hash"],
                "decoded_hash": row["decoded_hash"],
                "total_frames": row["total_frames"],
                "decoded_frames": row["decoded_frames"],
                "failed_reads": row["failed_reads"],
                "status": row["status"],
                "is_all_zero": row["is_all_zero"],
                "resize_mode": "letterbox",
                "pad_value": args.pad_value,
                "source_month": row["source_month"],
                "source_extension": row["source_extension"],
            })

    manifest_fields = [
        "split", "path", "label", "raw_hash", "decoded_hash", "total_frames",
        "decoded_frames", "failed_reads", "status", "is_all_zero",
        "resize_mode", "pad_value", "source_month", "source_extension",
    ]
    write_manifest(output_dir / "split_manifest.csv", manifest_rows, manifest_fields)
    write_split_diagnostics(output_dir, args.split_strategy, decoded_unique, split_rows, split_score)

    if skipped_rows:
        skipped_fields = [
            "path", "label", "raw_hash", "decoded_hash", "total_frames",
            "decoded_frames", "failed_reads", "status", "is_all_zero",
        ]
        for row in skipped_rows:
            row.setdefault("is_all_zero", "")
        write_manifest(output_dir / "skipped_manifest.csv", skipped_rows, skipped_fields)
        print(f"skipped: {len(skipped_rows)} videos; see skipped_manifest.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="dataset root containing positive/ and negative/ folders")
    parser.add_argument("--output_dir", default="data",
                        help="directory where train.npz, val.npz, test.npz, and manifests are written")
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    parser.add_argument("--pad_value", type=int, default=128,
                        help="padding value for aspect-ratio letterbox resize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.20)
    parser.add_argument("--split_strategy", choices=["random", "source_aware"], default="random",
                        help="random stratified split or deterministic candidate search that reduces source/month imbalance")
    parser.add_argument("--source_aware_candidates", type=int, default=500,
                        help="number of deterministic stratified split candidates to score for source_aware")
    parser.add_argument("--unreadable_policy", choices=["exclude", "zeros"], default="exclude",
                        help="exclude unreadable videos by default, or zero-fill them explicitly")
    parser.add_argument("--allow_all_zero", action="store_true",
                        help="keep all-zero decoded videos instead of excluding them")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
