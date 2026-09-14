"""Deterministic, split-independent clip cache for subject-grouped experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from data.subject_metadata import is_included, read_metadata, validate_metadata
from data.video_utils import resize_with_letterbox


CACHE_FIELDS = [
    "video_id",
    "relative_path",
    "label",
    "subject_id",
    "session_id",
    "capture_datetime",
    "site_id",
    "device_id",
    "decoded_sha256",
    "cache_path",
    "mask_path",
    "fps",
    "total_frames",
    "duration_seconds",
    "width",
    "height",
    "clip_frame_indices",
    "content_boxes",
    "mask_width",
    "failed_reads",
    "status",
]


def deterministic_clip_indices(
    total_frames: int,
    native_fps: float,
    *,
    clips_per_video: int = 5,
    clip_len: int = 16,
    sample_fps: float = 15.0,
) -> list[list[int]]:
    if total_frames <= 0 or not math.isfinite(native_fps) or native_fps <= 0:
        raise ValueError("Video metadata must contain positive frame count and FPS")
    if sample_fps <= 0:
        raise ValueError("Sample FPS must be positive")
    offsets = np.rint(np.arange(clip_len) * native_fps / sample_fps).astype(int)
    span = int(offsets[-1]) + 1
    max_start = max(0, total_frames - span)
    starts = np.linspace(0, max_start, clips_per_video).round().astype(int)
    return [
        [min(int(start + offset), total_frames - 1) for offset in offsets]
        for start in starts
    ]


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _read_frame(cap: cv2.VideoCapture, requested: int, previous: tuple[np.ndarray, int] | None):
    for candidate in [requested, requested + 1, requested - 1, requested + 2, requested - 2]:
        if candidate < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = cap.read()
        if ok:
            return frame, candidate, candidate != requested
    if previous is not None:
        return previous[0].copy(), previous[1], True
    return None, requested, True


def decode_clips(
    path: Path,
    *,
    clips_per_video: int,
    clip_len: int,
    sample_fps: float,
    target_size: tuple[int, int],
    pad_value: int,
    max_failure_fraction: float = 0.10,
):
    cap = cv2.VideoCapture(str(path))
    raw_total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(max(0, cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(max(0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not math.isfinite(raw_total) or raw_total <= 0:
        cap.release()
        raise RuntimeError("invalid frame-count metadata")
    if not math.isfinite(raw_fps) or raw_fps <= 0:
        cap.release()
        raise RuntimeError("invalid FPS metadata")
    total = int(raw_total)
    requested_indices = deterministic_clip_indices(
        total,
        raw_fps,
        clips_per_video=clips_per_video,
        clip_len=clip_len,
        sample_fps=sample_fps,
    )

    clips = []
    masks = []
    actual_indices = []
    failures = 0
    for clip_indices in requested_indices:
        previous = None
        clip_frames = []
        clip_masks = []
        clip_actual = []
        for requested in clip_indices:
            frame, actual, failed = _read_frame(cap, requested, previous)
            failures += int(failed)
            if frame is None:
                cap.release()
                raise RuntimeError(f"could not decode frame near index {requested}")
            previous = (frame, actual)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, mask = resize_with_letterbox(
                frame, target_size, pad_value=pad_value, return_mask=True
            )
            clip_frames.append(frame)
            clip_masks.append(mask)
            clip_actual.append(actual)
        clips.append(clip_frames)
        masks.append(clip_masks)
        actual_indices.append(clip_actual)
    cap.release()

    requested_count = clips_per_video * clip_len
    if failures / requested_count > max_failure_fraction:
        raise RuntimeError(
            f"decoded with {failures}/{requested_count} failed or substituted frame reads"
        )
    clip_array = np.asarray(clips, dtype=np.uint8)
    mask_array = np.asarray(masks, dtype=bool)
    packed_masks = np.packbits(mask_array, axis=-1)
    metadata = {
        "fps": raw_fps,
        "total_frames": total,
        "duration_seconds": total / raw_fps,
        "width": width,
        "height": height,
        "clip_frame_indices": actual_indices,
        "content_boxes": [
            [
                [
                    int(np.where(mask)[1].min()),
                    int(np.where(mask)[0].min()),
                    int(np.where(mask)[1].max()) + 1,
                    int(np.where(mask)[0].max()) + 1,
                ]
                for mask in clip_masks
            ]
            for clip_masks in mask_array
        ],
        "failed_reads": failures,
        "mask_width": target_size[1],
    }
    return clip_array, packed_masks, metadata


def build_clip_cache(
    root: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    clips_per_video: int = 5,
    clip_len: int = 16,
    sample_fps: float = 15.0,
    target_size: tuple[int, int] = (112, 112),
    pad_value: int = 128,
) -> dict:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    validate_metadata(root, metadata_path, require_complete=True, verify_hashes=True)
    rows = [row for row in read_metadata(metadata_path) if is_included(row["include"])]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    excluded = []
    for video_number, row in enumerate(rows, 1):
        path = root / Path(row["relative_path"])
        try:
            clips, masks, video_meta = decode_clips(
                path,
                clips_per_video=clips_per_video,
                clip_len=clip_len,
                sample_fps=sample_fps,
                target_size=target_size,
                pad_value=pad_value,
            )
        except RuntimeError as exc:
            excluded.append({"video_id": row["video_id"], "reason": str(exc)})
            print(f"Cache {video_number}/{len(rows)}: excluded ({exc})", flush=True)
            continue
        clip_rel = Path("clips") / f"{row['video_id']}.npy"
        mask_rel = Path("masks") / f"{row['video_id']}.npy"
        _atomic_save(output_dir / clip_rel, clips)
        _atomic_save(output_dir / mask_rel, masks)
        manifest_rows.append(
            {
                **{key: row[key] for key in (
                    "video_id", "relative_path", "label", "subject_id", "session_id",
                    "capture_datetime", "site_id", "device_id"
                )},
                "decoded_sha256": hashlib.sha256(clips.tobytes(order="C")).hexdigest(),
                "cache_path": clip_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "fps": f"{video_meta['fps']:.8f}",
                "total_frames": video_meta["total_frames"],
                "duration_seconds": f"{video_meta['duration_seconds']:.8f}",
                "width": video_meta["width"],
                "height": video_meta["height"],
                "clip_frame_indices": json.dumps(video_meta["clip_frame_indices"], separators=(",", ":")),
                "content_boxes": json.dumps(video_meta["content_boxes"], separators=(",", ":")),
                "mask_width": video_meta["mask_width"],
                "failed_reads": video_meta["failed_reads"],
                "status": "ok",
            }
        )
        print(f"Cache {video_number}/{len(rows)}: saved; substituted reads={video_meta['failed_reads']}", flush=True)

    manifest_path = output_dir / "cache_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    config = {
        "version": 2,
        "clips_per_video": clips_per_video,
        "clip_len": clip_len,
        "sample_fps": sample_fps,
        "target_size": list(target_size),
        "pad_value": pad_value,
    }
    config_blob = json.dumps(config, sort_keys=True).encode("utf-8")
    config["config_sha256"] = hashlib.sha256(config_blob).hexdigest()
    (output_dir / "cache_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    diagnostics = {
        "included_metadata_rows": len(rows),
        "cached_videos": len(manifest_rows),
        "excluded_videos": len(excluded),
        "excluded_reasons": sorted({item["reason"] for item in excluded}),
        "label_counts": {
            str(label): sum(int(row["label"]) == label for row in manifest_rows)
            for label in (0, 1)
        },
    }
    (output_dir / "public_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    if excluded:
        (output_dir / "excluded_private.json").write_text(
            json.dumps(excluded, indent=2), encoding="utf-8"
        )
    return diagnostics
