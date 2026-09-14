#!/usr/bin/env python
"""Run nested subject-grouped ASD video benchmarks from the per-video clip cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.utils.subject_data import (
    assert_group_isolation,
    audit_clip_cache,
    make_subject_folds,
    read_cache_manifest,
    split_inner_subjects,
    subject_table,
)
from src.utils.subject_metrics import compute_subject_metrics, stratified_bootstrap_ci
from src.utils.subject_training import MODEL_NAMES, train_one_run


PRIMARY_MODELS = ("simple2dcnn", "simple3dlstm_v2", "r3d")


def validate_run_options(config: dict, args) -> None:
    if float(config.get("threshold", 0.5)) != 0.5:
        raise ValueError("Primary subject benchmark requires the fixed probability threshold 0.5")
    if config.get("temperature_scaling", False):
        raise ValueError("Primary subject benchmark does not support temperature scaling")
    if "normal" not in args.temporal_ablations:
        raise ValueError("Temporal ablations must include normal for the primary result")
    for name, values in (("models", args.models), ("seeds", args.seeds),
                         ("folds", args.folds), ("temporal_ablations", args.temporal_ablations)):
        if values is not None and len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} are not allowed")
    seeds = args.seeds or config["training_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be nonempty and unique")
    if args.epoch_limit is not None and args.epoch_limit < 1:
        raise ValueError("epoch_limit must be positive")


def validate_cache_config(cache_config: dict, protocol_config: dict) -> None:
    expected = {
        "version": int(protocol_config["clip_cache_version"]),
        "clips_per_video": 5,
        "clip_len": 16,
        "sample_fps": 15.0,
        "target_size": [112, 112],
        "pad_value": 128,
    }
    mismatches = {
        key: {"expected": value, "actual": cache_config.get(key)}
        for key, value in expected.items()
        if cache_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Clip cache does not match the primary protocol: {mismatches}")


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_tree_hash(root: Path = Path(".")) -> str:
    """Hash executable project sources, including uncommitted and untracked code."""
    digest = hashlib.sha256()
    excluded = {"asd", "private", "data_grouped", "runs", "reports", "tmp_tests_workspace"}
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix == ".py" or path.as_posix() in {"requirements.txt", "configs/subject_cv.json"}:
            candidates.append(path)
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_state() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_fold_assignments(path: Path, rows, folds, inner_splits) -> None:
    labels = dict(subject_table(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "subject_id", "label", "role"])
        writer.writeheader()
        for fold in folds:
            fold_index = fold["fold"]
            train_subjects, val_subjects = inner_splits[fold_index]
            for role, subjects in (
                ("train", train_subjects),
                ("validation", val_subjects),
                ("test", fold["outer_test"]),
            ):
                for subject in sorted(subjects):
                    writer.writerow(
                        {"fold": fold_index, "subject_id": subject, "label": labels[subject], "role": role}
                    )


def _read_subject_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "subject_id": row["subject_id"],
                "label": int(row["label"]),
                "logit": float(row["logit"]),
            }
            for row in csv.DictReader(handle)
            if row["level"] == "subject"
        ]


def aggregate_seed_predictions(
    run_dirs: list[Path], *, expected_predictions_per_subject: int | None = None
) -> list[dict]:
    grouped = defaultdict(list)
    labels = defaultdict(set)
    for run_dir in run_dirs:
        for row in _read_subject_predictions(run_dir / "predictions_normal.csv"):
            grouped[row["subject_id"]].append(row["logit"])
            labels[row["subject_id"]].add(row["label"])
    output = []
    for subject, logits in sorted(grouped.items()):
        if len(labels[subject]) != 1:
            raise ValueError(f"Subject {subject!r} has inconsistent out-of-fold labels")
        if expected_predictions_per_subject is not None and len(logits) != expected_predictions_per_subject:
            raise ValueError(
                f"Subject {subject!r} has {len(logits)} seed predictions; "
                f"expected {expected_predictions_per_subject}"
            )
        output.append(
            {
                "subject_id": subject,
                "label": next(iter(labels[subject])),
                "logit": float(np.mean(logits)),
                "seed_predictions": len(logits),
            }
        )
    return output


def _init_wandb(mode: str, *, project: str, config: dict, name: str):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WandB mode was enabled but wandb is not installed") from exc
    os.environ.setdefault("WANDB_MODE", mode)
    return wandb.init(project=project, name=name, config=config, mode=mode, reinit=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run(args) -> Path:
    cache_root = Path(args.cache_dir).resolve()
    manifest_path = Path(args.manifest or cache_root / "cache_manifest.csv").resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_run_options(config, args)

    cache_config = json.loads((cache_root / "cache_config.json").read_text(encoding="utf-8"))
    validate_cache_config(cache_config, config)
    rows = read_cache_manifest(manifest_path)
    integrity = audit_clip_cache(
        rows,
        cache_root,
        expected_clips=int(cache_config["clips_per_video"]),
        expected_frames=int(cache_config["clip_len"]),
        expected_size=tuple(cache_config["target_size"]),
        pad_value=int(cache_config["pad_value"]),
    )
    folds = make_subject_folds(
        rows,
        max_folds=int(config["max_outer_folds"]),
        split_seed=int(config["split_seed"]),
    )
    if args.folds:
        selected = set(args.folds)
        folds = [fold for fold in folds if fold["fold"] in selected]
        missing = selected - {fold["fold"] for fold in folds}
        if missing:
            raise ValueError(f"Unknown fold numbers: {sorted(missing)}")
    inner_splits = {}
    for fold in folds:
        train_subjects, val_subjects = split_inner_subjects(
            rows,
            fold["outer_train"],
            seed=int(config["split_seed"]) + int(fold["fold"]),
        )
        assert_group_isolation(rows, train_subjects, val_subjects, fold["outer_test"])
        inner_splits[fold["fold"]] = (train_subjects, val_subjects)

    models = tuple(args.models or PRIMARY_MODELS)
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    seeds = args.seeds or [int(seed) for seed in config["training_seeds"]]
    state = git_state()
    run_spec = {
        "protocol_config": config,
        "models": models,
        "seeds": seeds,
        "folds": [fold["fold"] for fold in folds],
        "cache_config": cache_config,
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git_commit": state["commit"],
        "source_tree_sha256": source_tree_hash(),
        "permutation_seed": args.permutation_seed,
        "temporal_ablations": args.temporal_ablations,
        "pretrained": not args.no_pretrained,
        "epoch_limit_per_stage": args.epoch_limit,
    }
    config_hash = stable_hash(run_spec)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_dir = Path(args.output_dir) / f"{timestamp}_{config_hash[:12]}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    _write_json(experiment_dir / "configuration.json", {**run_spec, "configuration_sha256": config_hash})
    _write_json(experiment_dir / "environment.json", environment_state())
    _write_json(experiment_dir / "git_state.json", state)
    _write_json(experiment_dir / "cache_integrity.json", integrity)
    _write_fold_assignments(experiment_dir / "fold_assignments.csv", rows, folds, inner_splits)

    if args.validate_only:
        print(f"Validation passed; no training run. Artifacts: {experiment_dir}")
        return experiment_dir

    device = resolve_device(args.device)
    run_dirs_by_model = defaultdict(list)
    run_summaries = []
    for model_name in models:
        if model_name not in config["models"]:
            raise ValueError(f"No configuration exists for model {model_name}")
        for fold in folds:
            train_subjects, val_subjects = inner_splits[fold["fold"]]
            for seed in seeds:
                relative = Path(model_name) / f"fold_{fold['fold']}" / f"seed_{seed}"
                run_dir = experiment_dir / relative
                wandb_run = _init_wandb(
                    args.wandb_mode,
                    project=args.wandb_project,
                    config={**run_spec, "model": model_name, "fold": fold["fold"], "seed": seed},
                    name=f"{model_name}-fold{fold['fold']}-seed{seed}-{config_hash[:8]}",
                )
                try:
                    summary = train_one_run(
                        model_name=model_name,
                        model_config=config["models"][model_name],
                        protocol_config=config,
                        rows=rows,
                        cache_root=cache_root,
                        train_subjects=train_subjects,
                        val_subjects=val_subjects,
                        test_subjects=fold["outer_test"],
                        seed=int(seed),
                        fold=int(fold["fold"]),
                        run_dir=run_dir,
                        device=device,
                        pretrained=not args.no_pretrained,
                        permutation_seed=args.permutation_seed,
                        temporal_ablations=tuple(args.temporal_ablations),
                        epoch_limit=args.epoch_limit,
                        wandb_run=wandb_run,
                    )
                finally:
                    if wandb_run is not None:
                        wandb_run.finish()
                run_dirs_by_model[model_name].append(run_dir)
                run_summaries.append(summary)

    model_summaries = {}
    expected_test_subjects = set().union(*(fold["outer_test"] for fold in folds))
    for model_name, run_dirs in run_dirs_by_model.items():
        subject_rows = aggregate_seed_predictions(
            run_dirs, expected_predictions_per_subject=len(seeds)
        )
        if {row["subject_id"] for row in subject_rows} != expected_test_subjects:
            raise ValueError(f"{model_name}: out-of-fold subject coverage is incomplete")
        model_summaries[model_name] = {
            "subject_metrics": compute_subject_metrics(subject_rows),
            "bootstrap_95_ci": stratified_bootstrap_ci(
                subject_rows,
                samples=int(config["bootstrap_samples"]),
                seed=int(config["split_seed"]),
            ),
            "subjects": subject_rows,
            "interpretation": "Internal subject-grouped validation; not external or clinical generalization.",
        }
    _write_json(
        experiment_dir / "summary.json",
        {"configuration_sha256": config_hash, "runs": run_summaries, "models": model_summaries},
    )
    print(f"Completed subject benchmark: {experiment_dir}")
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested subject-grouped benchmark. Requires completed private metadata and clip cache."
    )
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="configs/subject_cv.json")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--permutation_seed", type=int, default=None)
    parser.add_argument(
        "--temporal_ablations",
        nargs="+",
        choices=["normal", "reversed", "shuffled", "static"],
        default=["normal", "reversed", "shuffled", "static"],
    )
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="ASD-SubjectCV")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--epoch_limit", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
