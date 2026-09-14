"""Exclude strongly verified repeated footage from an experiment-only manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.utils.subject_data import read_cache_manifest


def overlap_evidence(left, right):
    if left.shape != right.shape:
        return {"verified_overlap": False, "reason": "different clip shapes"}
    identical_frames = np.all(left == right, axis=(2, 3, 4))
    identical_clips = identical_frames.all(axis=1)
    error = float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean())
    fraction = float(identical_frames.mean())
    verified = len(left) >= 5 and identical_clips.sum() >= 4 and fraction >= 0.9 and error < 0.1
    return {
        "verified_overlap": bool(verified),
        "identical_frames": int(identical_frames.sum()),
        "sampled_frames": int(identical_frames.size),
        "identical_clips": int(identical_clips.sum()),
        "mean_absolute_pixel_error": error,
    }


def build_reviewed_manifest(cache_root, review_path, output):
    source = cache_root / "cache_manifest.csv"
    rows = read_cache_manifest(source)
    by_id = {row["video_id"]: row for row in rows}
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidates = {
        tuple(sorted((pair["left"]["video_id"], pair["right"]["video_id"])))
        for pair in review["pairs"]
        if pair["similarity"] >= review["cross_subject_pairs_review_threshold"]
    }
    excluded, decisions = set(), []
    for left_id, right_id in sorted(candidates):
        left, right = by_id[left_id], by_id[right_id]
        left_array = np.load(cache_root / left["cache_path"], mmap_mode="r")
        right_array = np.load(cache_root / right["cache_path"], mmap_mode="r")
        evidence = overlap_evidence(left_array, right_array)
        if not evidence["verified_overlap"]:
            raise ValueError("A high-similarity candidate requires manual review before creating the training manifest")
        if left["label"] != right["label"]:
            raise ValueError("Repeated footage has conflicting labels; resolve metadata before training")
        retained, removed = sorted((left, right), key=lambda row: row["relative_path"])
        excluded.add(removed["video_id"])
        decisions.append({
            **evidence, "retained_video_id": retained["video_id"],
            "excluded_video_id": removed["video_id"],
            "selection_rule": "Retain lexically first relative path; independent of model outcomes.",
        })
    selected = [row for row in rows if row["video_id"] not in excluded]
    if output.resolve() == source.resolve():
        raise ValueError("Reviewed output must not overwrite the original cache manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reviewed_manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_videos": len(rows), "selected_videos": len(selected),
        "excluded_repeated_footage": len(excluded), "decisions": decisions,
        "scope": "Verified sampled footage overlap; no biometric identity inference. Raw files preserved.",
    }
    output.with_suffix(".review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", default="private/reviewed_cache_manifest.csv")
    args = parser.parse_args()
    report = build_reviewed_manifest(Path(args.cache_dir), Path(args.review), Path(args.output))
    print(json.dumps({key: report[key] for key in ("source_videos", "selected_videos", "excluded_repeated_footage")}, indent=2))


if __name__ == "__main__":
    main()
