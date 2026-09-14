"""CLI for private metadata and deterministic clip-cache preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.clip_cache import build_clip_cache
from data.subject_metadata import (
    MetadataValidationError,
    validate_metadata,
    write_metadata_template,
    assign_unique_subjects,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="create a private mapping template")
    template.add_argument("--root", required=True)
    template.add_argument("--output", default="private/subject_sessions.csv")

    unique = subparsers.add_parser("assign_unique_subjects", help="assign IDs after owner confirmation of one distinct person per video")
    unique.add_argument("--root", required=True)
    unique.add_argument("--metadata", default="private/subject_sessions.csv")
    unique.add_argument("--confirmed_one_video_per_subject", action="store_true", required=True)

    validate = subparsers.add_parser("validate", help="validate a completed private mapping")
    validate.add_argument("--root", required=True)
    validate.add_argument("--metadata", default="private/subject_sessions.csv")
    validate.add_argument("--skip_hash_check", action="store_true")
    validate.add_argument("--allow_incomplete", action="store_true")

    cache = subparsers.add_parser("cache", help="build a split-independent per-video clip cache")
    cache.add_argument("--root", required=True)
    cache.add_argument("--metadata", default="private/subject_sessions.csv")
    cache.add_argument("--output_dir", default="data_grouped")
    cache.add_argument("--clips_per_video", type=int, default=5)
    cache.add_argument("--clip_len", type=int, default=16)
    cache.add_argument("--sample_fps", type=float, default=15.0)
    cache.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    cache.add_argument("--pad_value", type=int, default=128)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            result = write_metadata_template(Path(args.root), Path(args.output))
        elif args.command == "assign_unique_subjects":
            result = assign_unique_subjects(Path(args.root), Path(args.metadata))
        elif args.command == "validate":
            result = validate_metadata(
                Path(args.root),
                Path(args.metadata),
                require_complete=not args.allow_incomplete,
                verify_hashes=not args.skip_hash_check,
            )
        else:
            result = build_clip_cache(
                Path(args.root),
                Path(args.metadata),
                Path(args.output_dir),
                clips_per_video=args.clips_per_video,
                clip_len=args.clip_len,
                sample_fps=args.sample_fps,
                target_size=tuple(args.target_size),
                pad_value=args.pad_value,
            )
    except (MetadataValidationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
