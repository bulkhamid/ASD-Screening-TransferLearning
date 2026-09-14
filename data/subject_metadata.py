"""Private subject/session metadata generation and validation."""

from __future__ import annotations

import csv
import hashlib
import math
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import cv2


VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi"}
METADATA_FIELDS = [
    "video_id",
    "relative_path",
    "label",
    "subject_id",
    "session_id",
    "capture_datetime",
    "site_id",
    "device_id",
    "include",
    "notes",
]


class MetadataValidationError(ValueError):
    """Raised when metadata cannot support subject-independent evaluation."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crawl_labeled_videos(root: Path) -> list[tuple[Path, int]]:
    videos: list[tuple[Path, int]] = []
    for label, folder_name in ((1, "positive"), (0, "negative")):
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        videos.extend(
            (path, label)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    return videos


def capture_datetime_from_name(path: Path) -> str:
    match = re.search(
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})",
        path.stem,
    )
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


def _video_is_readable(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _frame = cap.read()
    cap.release()
    return bool(ok and math.isfinite(frame_count) and frame_count > 0)


def write_metadata_template(root: Path, output: Path) -> dict:
    """Create a private mapping template without inventing subject identities."""
    root = root.expanduser().resolve()
    rows = []
    unreadable = 0
    for path, label in crawl_labeled_videos(root):
        readable = _video_is_readable(path)
        unreadable += int(not readable)
        rows.append(
            {
                "video_id": sha256_file(path),
                "relative_path": path.relative_to(root).as_posix(),
                "label": label,
                "subject_id": "",
                "session_id": "",
                "capture_datetime": capture_datetime_from_name(path),
                "site_id": "",
                "device_id": "",
                "include": "1" if readable else "0",
                "notes": "" if readable else "unreadable video; excluded pending review",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"videos": len(rows), "readable": len(rows) - unreadable, "unreadable": unreadable}


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(METADATA_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise MetadataValidationError(f"Metadata is missing columns: {sorted(missing)}")
        return list(reader)


def assign_unique_subjects(root: Path, metadata_path: Path) -> dict:
    """Apply the dataset owner's confirmation of one distinct participant per video."""
    validate_metadata(root, metadata_path, require_complete=False, verify_hashes=True)
    original = metadata_path.read_bytes()
    rows = read_metadata(metadata_path)
    included = [row for row in rows if is_included(row["include"])]
    for row in included:
        subject = "S_" + row["video_id"][:16]
        session = subject + "_V01"
        if row["subject_id"].strip() not in ("", subject) or row["session_id"].strip() not in ("", session):
            raise MetadataValidationError("Existing subject/session assignments require manual reconciliation")
        row["subject_id"], row["session_id"] = subject, session
        note = "Dataset owner confirmed one distinct main participant per video; helpers may appear."
        if note not in row["notes"]:
            row["notes"] = "; ".join(filter(None, [row["notes"], note]))
    if len({row["subject_id"] for row in included}) != len(included):
        raise MetadataValidationError("Generated subject IDs are not unique")
    temporary = metadata_path.with_suffix(".assigned.tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result = validate_metadata(root, temporary, require_complete=True, verify_hashes=False)
    backup = metadata_path.with_name(metadata_path.stem + ".before_unique_subjects.csv")
    if not backup.exists():
        backup.write_bytes(original)
    os.replace(temporary, metadata_path)
    provenance = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "Explicit dataset-owner confirmation: each main participant appears in only one video.",
        "session_definition": "One participant-specific recording session per included video.",
        "helper_people": "May appear; not assigned as target participants.",
        "identity_independence": "Owner-confirmed; not independently established through biometric identification.",
        "included_subjects": len(included),
        "metadata_sha256": sha256_file(metadata_path),
    }
    metadata_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return result


def is_included(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_metadata(
    root: Path,
    metadata_path: Path,
    *,
    require_complete: bool = True,
    verify_hashes: bool = True,
    minimum_subjects_per_class: int = 3,
) -> dict:
    """Validate privacy, identity, label, and dataset-coverage invariants."""
    root = root.expanduser().resolve()
    rows = read_metadata(metadata_path)
    errors: list[str] = []

    expected = {
        path.relative_to(root).as_posix(): (path, label)
        for path, label in crawl_labeled_videos(root)
    }
    seen_paths: Counter[str] = Counter()
    seen_ids: Counter[str] = Counter()
    subject_labels: dict[str, set[int]] = defaultdict(set)
    session_subjects: dict[str, set[str]] = defaultdict(set)
    included_rows = []
    missing_subject_lines = []
    missing_session_lines = []

    for line_number, row in enumerate(rows, 2):
        rel = row["relative_path"].strip().replace("\\", "/")
        seen_paths[rel] += 1
        seen_ids[row["video_id"].strip()] += 1
        candidate = Path(rel)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"line {line_number}: relative_path must stay below the dataset root")
            continue
        if rel not in expected:
            errors.append(f"line {line_number}: unknown video path {rel!r}")
            continue
        path, expected_label = expected[rel]
        try:
            label = int(row["label"])
        except ValueError:
            errors.append(f"line {line_number}: label must be 0 or 1")
            continue
        if label not in (0, 1) or label != expected_label:
            errors.append(f"line {line_number}: label does not match the class folder")
        if verify_hashes and row["video_id"].strip() != sha256_file(path):
            errors.append(f"line {line_number}: video_id SHA-256 does not match {rel}")
        if not is_included(row["include"]):
            continue

        subject = row["subject_id"].strip()
        session = row["session_id"].strip()
        if require_complete and not subject:
            missing_subject_lines.append(line_number)
        if require_complete and not session:
            missing_session_lines.append(line_number)
        if subject:
            subject_labels[subject].add(label)
        if session:
            session_subjects[session].add(subject)
        included_rows.append(row)

    missing_paths = sorted(set(expected) - set(seen_paths))
    duplicate_paths = sorted(path for path, count in seen_paths.items() if count != 1)
    duplicate_ids = sorted(video_id for video_id, count in seen_ids.items() if count != 1)
    if missing_paths:
        errors.append(f"metadata omits {len(missing_paths)} raw videos")
    if duplicate_paths:
        errors.append(f"metadata contains duplicate paths: {duplicate_paths[:5]}")
    if duplicate_ids:
        errors.append(f"metadata contains duplicate video IDs: {duplicate_ids[:5]}")
    if missing_subject_lines:
        errors.append(
            f"{len(missing_subject_lines)} included videos have no subject_id "
            f"(first lines: {missing_subject_lines[:5]})"
        )
    if missing_session_lines:
        errors.append(
            f"{len(missing_session_lines)} included videos have no session_id "
            f"(first lines: {missing_session_lines[:5]})"
        )
    for subject, labels in subject_labels.items():
        if len(labels) != 1:
            errors.append(f"subject {subject!r} has inconsistent labels: {sorted(labels)}")
    for session, subjects in session_subjects.items():
        if len(subjects) != 1 or "" in subjects:
            errors.append(f"session {session!r} is not assigned to exactly one subject")

    subject_counts = Counter()
    for labels in subject_labels.values():
        if len(labels) == 1:
            subject_counts[next(iter(labels))] += 1
    if require_complete:
        for label in (0, 1):
            if subject_counts[label] < minimum_subjects_per_class:
                errors.append(
                    f"class {label} has {subject_counts[label]} subjects; "
                    f"at least {minimum_subjects_per_class} are required"
                )

    diagnostics = {
        "raw_videos": len(expected),
        "metadata_rows": len(rows),
        "included_videos": len(included_rows),
        "excluded_videos": len(rows) - len(included_rows),
        "subjects": len(subject_labels),
        "sessions": len(session_subjects),
        "subjects_by_label": {str(label): int(subject_counts[label]) for label in (0, 1)},
        "errors": errors,
    }
    if errors:
        raise MetadataValidationError("\n".join(errors))
    return diagnostics
