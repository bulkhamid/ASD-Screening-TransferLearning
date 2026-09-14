"""Clip-cache datasets and subject-level split helpers."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


def read_cache_manifest(path: Path, *, require_complete: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "video_id", "label", "subject_id", "session_id", "cache_path", "mask_path"
    }
    if not required.issubset(fields):
        raise ValueError(f"Cache manifest is missing columns: {sorted(required - fields)}")
    if not rows:
        raise ValueError("Cache manifest contains no videos")
    if require_complete:
        incomplete = [row["video_id"] for row in rows if not row["subject_id"] or not row["session_id"]]
        if incomplete:
            raise ValueError(
                f"Cache manifest has {len(incomplete)} included videos without subject/session IDs"
            )
    return rows


def subject_table(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if not row["subject_id"].strip() or int(row["label"]) not in (0, 1):
            raise ValueError("Every subject must have a nonempty ID and a binary label")
        labels[row["subject_id"]].add(int(row["label"]))
    inconsistent = {subject: values for subject, values in labels.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Subjects with inconsistent labels: {inconsistent}")
    return sorted((subject, next(iter(values))) for subject, values in labels.items())


def make_subject_folds(
    rows: list[dict[str, str]], *, max_folds: int = 5, split_seed: int = 42
) -> list[dict[str, set[str]]]:
    table = subject_table(rows)
    subjects = np.asarray([item[0] for item in table])
    labels = np.asarray([item[1] for item in table])
    counts = Counter(labels.tolist())
    minimum = min(counts.get(0, 0), counts.get(1, 0))
    if minimum < 3:
        raise ValueError("At least three subjects in each class are required")
    folds = min(max_folds, minimum)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    output = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(subjects, labels), 1
    ):
        output.append(
            {
                "fold": fold_index,
                "outer_train": set(subjects[train_indices].tolist()),
                "outer_test": set(subjects[test_indices].tolist()),
            }
        )
    return output


def split_inner_subjects(
    rows: list[dict[str, str]],
    outer_train: set[str],
    *,
    seed: int,
    validation_fraction: float = 0.20,
) -> tuple[set[str], set[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    table = [(subject, label) for subject, label in subject_table(rows) if subject in outer_train]
    if {subject for subject, _label in table} != outer_train:
        raise ValueError("Outer-training split contains unknown subjects")
    subjects = np.asarray([item[0] for item in table])
    labels = np.asarray([item[1] for item in table])
    counts = Counter(labels.tolist())
    if min(counts.get(0, 0), counts.get(1, 0)) < 2:
        raise ValueError("Inner splitting requires at least two outer-training subjects per class")
    validation_count = max(2, int(math.ceil(len(subjects) * validation_fraction)))
    validation_count = min(validation_count, len(subjects) - 2)
    # Constrain the proportional allocation so BOTH partitions retain both classes.
    minimum_negative = max(1, validation_count - (counts[1] - 1))
    maximum_negative = min(counts[0] - 1, validation_count - 1)
    negative_count = int(np.clip(
        round(validation_count * counts[0] / len(subjects)),
        minimum_negative, maximum_negative,
    ))
    rng = np.random.default_rng(seed)
    val_subjects = set()
    for label, count in ((0, negative_count), (1, validation_count - negative_count)):
        candidates = subjects[labels == label]
        val_subjects.update(rng.permutation(candidates)[:count].tolist())
    return outer_train - val_subjects, val_subjects


def assert_group_isolation(
    rows: list[dict[str, str]],
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
) -> None:
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise ValueError("Subject overlap detected between train, validation, and test")
    session_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject = row["subject_id"]
        split = (
            "train" if subject in train_subjects else
            "val" if subject in val_subjects else
            "test" if subject in test_subjects else
            "unused"
        )
        if split != "unused":
            session_splits[row["session_id"]].add(split)
    leaked = {session: splits for session, splits in session_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Session overlap detected: {leaked}")


def rows_for_subjects(rows: list[dict[str, str]], subjects: set[str]):
    return [row for row in rows if row["subject_id"] in subjects]


def unpack_masks(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1)[..., :width].astype(bool)


def _load_video_arrays(row: dict[str, str], cache_root: Path):
    clips = np.load(cache_root / row["cache_path"], mmap_mode="r")
    packed = np.load(cache_root / row["mask_path"], mmap_mode="r")
    masks = unpack_masks(packed, clips.shape[3])
    return clips, masks


def compute_content_normalization(rows: list[dict[str, str]], cache_root: Path):
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for row in rows:
        clips, masks = _load_video_arrays(row, cache_root)
        for clip, mask in zip(clips, masks):
            pixels = np.asarray(clip)[mask]
            if not len(pixels):
                continue
            values = pixels.astype(np.float64) / 255.0
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += len(values)
    if count == 0:
        raise ValueError("No content pixels were available for normalization")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return tuple(mean.tolist()), tuple(np.sqrt(variance).tolist())


def audit_clip_cache(
    rows: list[dict[str, str]],
    cache_root: Path,
    *,
    expected_clips: int = 5,
    expected_frames: int = 16,
    expected_size: tuple[int, int] = (112, 112),
    pad_value: int = 128,
) -> dict:
    """Verify per-video cache shape, masks, padding, and exact duplicates."""
    errors = []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    zero_videos = []
    for row in rows:
        try:
            clips, masks = _load_video_arrays(row, cache_root)
        except (OSError, ValueError) as exc:
            errors.append(f"{row['video_id']}: cannot load cache ({exc})")
            continue
        expected_shape = (expected_clips, expected_frames, *expected_size, 3)
        if tuple(clips.shape) != expected_shape:
            errors.append(f"{row['video_id']}: clip shape {tuple(clips.shape)} != {expected_shape}")
            continue
        if clips.dtype != np.uint8:
            errors.append(f"{row['video_id']}: expected uint8 clips, got {clips.dtype}")
        if tuple(masks.shape) != expected_shape[:-1]:
            errors.append(f"{row['video_id']}: content mask shape {tuple(masks.shape)} is invalid")
            continue
        if not bool(masks.reshape(expected_clips, -1).any(axis=1).all()):
            errors.append(f"{row['video_id']}: one or more clips have an empty content mask")
        if not bool(np.asarray(clips).any()):
            zero_videos.append(row["video_id"])
        padding = np.asarray(clips)[~masks]
        if len(padding) and not bool(np.all(padding == pad_value)):
            errors.append(f"{row['video_id']}: padding is not uniformly {pad_value}")
        digest = hashlib.sha256(np.asarray(clips).tobytes(order="C")).hexdigest()
        recorded_digest = row.get("decoded_sha256", "").strip()
        if recorded_digest and recorded_digest != digest:
            errors.append(f"{row['video_id']}: decoded SHA-256 does not match manifest")
        hashes[digest].append((row["video_id"], row["subject_id"]))

    exact_cross_subject = []
    for digest, items in hashes.items():
        if len({subject for _video, subject in items}) > 1:
            exact_cross_subject.append({"sha256": digest, "videos": [item[0] for item in items]})
    if zero_videos:
        errors.append(f"all-zero videos: {zero_videos}")
    if exact_cross_subject:
        errors.append(f"{len(exact_cross_subject)} exact decoded duplicates cross subject boundaries")
    diagnostics = {
        "videos": len(rows),
        "unique_decoded_hashes": len(hashes),
        "all_zero_videos": zero_videos,
        "exact_cross_subject_duplicates": exact_cross_subject,
        "errors": errors,
    }
    if errors:
        raise ValueError("Clip-cache integrity failed:\n" + "\n".join(errors))
    return diagnostics


class ClipTransform:
    """Apply one spatial/color transform consistently to every clip frame."""

    def __init__(
        self,
        mean,
        std,
        *,
        training: bool,
        seed: int = 42,
        temporal_ablation: str = "normal",
    ):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.training = training
        self.seed = seed
        self.temporal_ablation = temporal_ablation

    def __call__(self, clip: np.ndarray, mask: np.ndarray, *, sample_seed: int = 0):
        frames = torch.from_numpy(np.asarray(clip).copy()).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # T,C,H,W
        masks = torch.from_numpy(np.asarray(mask).copy()).bool().unsqueeze(1)
        generator = torch.Generator().manual_seed(self.seed + sample_seed)

        if self.temporal_ablation == "reversed":
            frames, masks = frames.flip(0), masks.flip(0)
        elif self.temporal_ablation == "shuffled":
            order = torch.randperm(frames.shape[0], generator=generator)
            frames, masks = frames[order], masks[order]
        elif self.temporal_ablation == "static":
            middle = frames.shape[0] // 2
            frames = frames[middle:middle + 1].expand_as(frames).clone()
            masks = masks[middle:middle + 1].expand_as(masks).clone()
        elif self.temporal_ablation != "normal":
            raise ValueError(f"Unknown temporal ablation: {self.temporal_ablation}")

        if self.training:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + sample_seed)
                top, left, height, width = RandomResizedCrop.get_params(
                    frames[0], scale=(0.9, 1.0), ratio=(0.95, 1.05)
                )
                do_flip = bool(torch.rand(()) < 0.5)
                jitter = ColorJitter.get_params((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), None)
            output_size = [frames.shape[-2], frames.shape[-1]]
            frames = TF.resized_crop(
                frames, top, left, height, width, output_size,
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            )
            masks = TF.resized_crop(
                masks.float(), top, left, height, width, output_size,
                interpolation=InterpolationMode.NEAREST,
            ).bool()
            if do_flip:
                frames, masks = TF.hflip(frames), TF.hflip(masks)
            fn_idx, brightness, contrast, saturation, hue = jitter
            for fn_id in fn_idx.tolist():
                if fn_id == 0 and brightness is not None:
                    frames = TF.adjust_brightness(frames, brightness)
                elif fn_id == 1 and contrast is not None:
                    frames = TF.adjust_contrast(frames, contrast)
                elif fn_id == 2 and saturation is not None:
                    frames = TF.adjust_saturation(frames, saturation)
                elif fn_id == 3 and hue is not None:
                    frames = TF.adjust_hue(frames, hue)

        frames = (frames - self.mean) / self.std
        frames = torch.where(masks.expand_as(frames), frames, torch.zeros_like(frames))
        return frames.permute(1, 0, 2, 3).contiguous()


class SubjectBalancedClipDataset(Dataset):
    def __init__(
        self,
        rows,
        cache_root: Path,
        transform: ClipTransform,
        *,
        samples_per_subject: int = 4,
        seed: int = 42,
        label_overrides: dict[str, int] | None = None,
    ):
        self.cache_root = cache_root
        self.transform = transform
        self.samples_per_subject = samples_per_subject
        self.seed = seed
        self.epoch = 0
        self.label_overrides = label_overrides or {}
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["subject_id"]][row["session_id"]].append(row)
        self.subjects = sorted(grouped)
        self.rows_by_session = {
            subject: {session: session_rows for session, session_rows in sessions.items()}
            for subject, sessions in grouped.items()
        }
        self.sessions_by_subject = {
            subject: sorted(sessions) for subject, sessions in self.rows_by_session.items()
        }

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.subjects) * self.samples_per_subject

    def sample_row(self, subject, rng):
        session = rng.choice(self.sessions_by_subject[subject])
        return rng.choice(self.rows_by_session[subject][session])

    def __getitem__(self, index):
        subject = self.subjects[index // self.samples_per_subject]
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        row = self.sample_row(subject, rng)
        clips, masks = _load_video_arrays(row, self.cache_root)
        clip_index = rng.randrange(len(clips))
        sample_seed = self.seed * 1_000_003 + self.epoch * 10_007 + index
        video = self.transform(clips[clip_index], masks[clip_index], sample_seed=sample_seed)
        label = self.label_overrides.get(subject, int(row["label"]))
        return video, torch.tensor(float(label), dtype=torch.float32)


class EvaluationClipDataset(Dataset):
    def __init__(self, rows, cache_root: Path, transform: ClipTransform):
        self.rows = rows
        self.cache_root = cache_root
        self.transform = transform
        self.items = []
        for row_index, row in enumerate(rows):
            clips = np.load(cache_root / row["cache_path"], mmap_mode="r")
            self.items.extend((row_index, clip_index) for clip_index in range(len(clips)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_index, clip_index = self.items[index]
        row = self.rows[row_index]
        clips, masks = _load_video_arrays(row, self.cache_root)
        video = self.transform(clips[clip_index], masks[clip_index], sample_seed=index)
        return (
            video,
            torch.tensor(float(row["label"]), dtype=torch.float32),
            row["video_id"],
            row["session_id"],
            row["subject_id"],
            clip_index,
        )


def shuffled_training_labels(rows, subjects: set[str], seed: int) -> dict[str, int]:
    table = [(subject, label) for subject, label in subject_table(rows) if subject in subjects]
    subject_ids = [item[0] for item in table]
    labels = [item[1] for item in table]
    rng = random.Random(seed)
    rng.shuffle(labels)
    return dict(zip(subject_ids, labels))
