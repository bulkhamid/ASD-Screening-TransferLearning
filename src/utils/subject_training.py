"""Training primitives for leakage-resistant subject-grouped experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models import PretrainedR3D, Simple2DCNN, Simple3DLSTM, Simple3DLSTMV2
from src.utils.subject_data import (
    KINETICS_MEAN,
    KINETICS_STD,
    ClipTransform,
    EvaluationClipDataset,
    SubjectBalancedClipDataset,
    compute_content_normalization,
    rows_for_subjects,
    shuffled_training_labels,
    assert_group_isolation,
    subject_table,
)
from src.utils.subject_metrics import compute_subject_metrics, hierarchical_aggregate


MODEL_NAMES = ("simple2dcnn", "simple3dlstm_v2", "simple3dlstm_legacy", "r3d")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str, model_config: dict, *, pretrained: bool = True) -> nn.Module:
    dropout = float(model_config.get("dropout", 0.3))
    if name == "simple2dcnn":
        return Simple2DCNN(p_drop=dropout)
    if name == "simple3dlstm_v2":
        return Simple3DLSTMV2(p_drop=dropout)
    if name == "simple3dlstm_legacy":
        return Simple3DLSTM(16, (112, 112), p_drop=dropout)
    if name == "r3d":
        return PretrainedR3D(pretrained=pretrained, freeze_batchnorm=True)
    raise ValueError(f"Unknown model: {name}")


def stage_schedule(model_name: str, model_config: dict) -> list[dict]:
    if model_name == "r3d":
        stages = [dict(stage) for stage in model_config["stages"]]
        expected = ["head", "layer4", "full"]
        names = [stage.get("name") for stage in stages]
        if names != expected:
            raise ValueError(f"R3D stages must be {expected}, got {names}")
        return stages
    return [
        {
            "name": "full",
            "epochs": int(model_config["max_epochs"]),
            "learning_rate": float(model_config["learning_rate"]),
        }
    ]


def configure_stage(model: nn.Module, model_name: str, stage_name: str) -> None:
    if model_name == "r3d":
        model.configure_stage(stage_name)
        return
    if stage_name != "full":
        raise ValueError(f"{model_name} only supports the full stage")
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GlobalCheckpointTracker:
    path: Path
    minimum_delta: float = 1e-6
    best_loss: float = math.inf
    best_stage: str | None = None
    best_epoch: int | None = None

    def update(
        self,
        validation_loss: float,
        model: nn.Module,
        *,
        stage: str,
        epoch: int,
        metadata: dict,
    ) -> bool:
        if not math.isfinite(validation_loss):
            raise ValueError("Validation loss must be finite; checkpoint was not updated")
        if validation_loss >= self.best_loss - self.minimum_delta:
            return False
        self.best_loss = float(validation_loss)
        self.best_stage = stage
        self.best_epoch = int(epoch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "validation_loss": self.best_loss,
                "stage": stage,
                "epoch": int(epoch),
                **metadata,
            },
            self.path,
        )
        return True


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    clip_rows = []
    for batch in loader:
        videos, labels, video_ids, session_ids, subject_ids, clip_indices = batch
        videos = videos.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(videos)
        logits = logits.detach().float().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        clip_indices = clip_indices.detach().cpu().numpy()
        for index in range(len(logits)):
            clip_rows.append(
                {
                    "video_id": str(video_ids[index]),
                    "session_id": str(session_ids[index]),
                    "subject_id": str(subject_ids[index]),
                    "clip_index": int(clip_indices[index]),
                    "label": int(labels[index]),
                    "logit": float(logits[index]),
                }
            )
    subject_rows = hierarchical_aggregate(clip_rows)
    return compute_subject_metrics(subject_rows), clip_rows, subject_rows


def _make_eval_loader(
    rows,
    cache_root: Path,
    mean,
    std,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    temporal_ablation: str = "normal",
):
    transform = ClipTransform(
        mean,
        std,
        training=False,
        temporal_ablation=temporal_ablation,
    )
    dataset = EvaluationClipDataset(rows, cache_root, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _write_predictions(path: Path, clip_rows: list[dict], subject_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "level", "subject_id", "session_id", "video_id", "clip_index",
            "label", "logit", "probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in clip_rows:
            logit = float(row["logit"])
            writer.writerow(
                {
                    "level": "clip",
                    **{key: row.get(key, "") for key in fields if key not in {"level", "probability"}},
                    "probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit)))),
                }
            )
        for row in subject_rows:
            logit = float(row["logit"])
            writer.writerow(
                {
                    "level": "subject",
                    "subject_id": row["subject_id"],
                    "label": row["label"],
                    "logit": logit,
                    "probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit)))),
                }
            )


def train_one_run(
    *,
    model_name: str,
    model_config: dict,
    protocol_config: dict,
    rows: list[dict[str, str]],
    cache_root: Path,
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
    seed: int,
    fold: int,
    run_dir: Path,
    device: torch.device,
    pretrained: bool = True,
    permutation_seed: int | None = None,
    temporal_ablations: tuple[str, ...] = ("normal", "reversed", "shuffled", "static"),
    epoch_limit: int | None = None,
    wandb_run=None,
) -> dict:
    """Train one model/seed/fold without accessing test data before selection."""
    assert_group_isolation(rows, train_subjects, val_subjects, test_subjects)
    labels_by_subject = dict(subject_table(rows))
    for role, subjects in (("train", train_subjects), ("validation", val_subjects), ("test", test_subjects)):
        if not subjects <= labels_by_subject.keys():
            raise ValueError(f"{role} split contains unknown subjects")
        if {labels_by_subject[subject] for subject in subjects} != {0, 1}:
            raise ValueError(f"{role} split must contain subjects from both classes")
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    train_rows = rows_for_subjects(rows, train_subjects)
    val_rows = rows_for_subjects(rows, val_subjects)
    test_rows = rows_for_subjects(rows, test_subjects)

    if model_name == "r3d":
        mean, std = KINETICS_MEAN, KINETICS_STD
        normalization_source = "torchvision R3D-18 Kinetics-400 V1 weights"
    else:
        mean, std = compute_content_normalization(train_rows, cache_root)
        normalization_source = "content pixels from inner-training subjects only"
    _json_dump(
        run_dir / "normalization.json",
        {"mean": mean, "std": std, "source": normalization_source, "padding_after_normalization": 0},
    )

    label_overrides = None
    if permutation_seed is not None:
        label_overrides = shuffled_training_labels(rows, train_subjects, permutation_seed)
        if any(subject in label_overrides for subject in val_subjects | test_subjects):
            raise RuntimeError("Permutation labels escaped the training subjects")

    train_transform = ClipTransform(mean, std, training=True, seed=seed)
    train_dataset = SubjectBalancedClipDataset(
        train_rows,
        cache_root,
        train_transform,
        samples_per_subject=int(protocol_config["samples_per_subject"]),
        seed=seed,
        label_overrides=label_overrides,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(protocol_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(protocol_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = _make_eval_loader(
        val_rows,
        cache_root,
        mean,
        std,
        batch_size=int(protocol_config["batch_size"]),
        num_workers=int(protocol_config["num_workers"]),
        device=device,
    )

    model = build_model(model_name, model_config, pretrained=pretrained).to(device)
    schedule = stage_schedule(model_name, model_config)
    tracker = GlobalCheckpointTracker(run_dir / "best_checkpoint.pth")
    criterion = nn.BCEWithLogitsLoss()
    history = []
    global_epoch = 0

    for stage in schedule:
        stage_name = stage["name"]
        configure_stage(model, model_name, stage_name)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"Stage {stage_name} has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(stage["learning_rate"]),
            weight_decay=float(model_config.get("weight_decay", 1e-4)),
        )
        stage_epochs = int(stage["epochs"])
        if epoch_limit is not None:
            stage_epochs = min(stage_epochs, epoch_limit)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage_epochs), eta_min=min(1e-6, float(stage["learning_rate"]) / 10)
        )
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        stage_best = math.inf
        stage_wait = 0
        stage_checkpoint = run_dir / f"stage_{stage_name}_best.pth"

        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            train_dataset.set_epoch(global_epoch)
            model.train()
            running_loss = 0.0
            examples = 0
            started = time.time()
            for videos, labels in train_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(videos), labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite training loss; stopped before optimizer update")
                scaler.scale(loss).backward()
                gradient_clip = model_config.get("gradient_clip")
                if gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach().cpu()) * len(labels)
                examples += len(labels)
            scheduler.step()

            val_metrics, _val_clips, _val_subjects = evaluate_model(model, val_loader, device)
            record = {
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "global_epoch": global_epoch,
                "train_loss": running_loss / max(1, examples),
                "validation": val_metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - started,
            }
            history.append(record)
            _json_dump(run_dir / "history.json", history)
            print(
                f"{model_name} fold={fold} seed={seed} {stage_name} "
                f"epoch={stage_epoch}/{stage_epochs} train_loss={record['train_loss']:.5f} "
                f"val_nll={val_metrics['nll']:.5f} seconds={record['seconds']:.1f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": record["train_loss"],
                        "validation/nll": val_metrics["nll"],
                        "validation/roc_auc": val_metrics["roc_auc"],
                        "stage": stage_name,
                        "global_epoch": global_epoch,
                    },
                    step=global_epoch,
                )

            val_loss = float(val_metrics["nll"])
            metadata = {
                "model_name": model_name,
                "normalization": {"mean": mean, "std": std},
                "fold": fold,
                "seed": seed,
            }
            tracker.update(
                val_loss,
                model,
                stage=stage_name,
                epoch=global_epoch,
                metadata=metadata,
            )
            if val_loss < stage_best - 1e-6:
                stage_best = val_loss
                stage_wait = 0
                torch.save(model.state_dict(), stage_checkpoint)
            else:
                stage_wait += 1
                if stage_wait >= int(model_config.get("patience", 8)):
                    break

        if stage_checkpoint.exists():
            model.load_state_dict(torch.load(stage_checkpoint, map_location=device, weights_only=True))

    if not tracker.path.exists():
        raise RuntimeError("Training did not produce a validation checkpoint")
    checkpoint = torch.load(tracker.path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_hash = _checkpoint_sha256(tracker.path)
    (run_dir / "best_checkpoint.sha256").write_text(checkpoint_hash + "\n", encoding="ascii")
    _json_dump(run_dir / "history.json", history)

    test_results = {}
    for ablation in temporal_ablations:
        test_loader = _make_eval_loader(
            test_rows,
            cache_root,
            mean,
            std,
            batch_size=int(protocol_config["batch_size"]),
            num_workers=int(protocol_config["num_workers"]),
            device=device,
            temporal_ablation=ablation,
        )
        metrics, clip_rows, subject_rows = evaluate_model(model, test_loader, device)
        _write_predictions(run_dir / f"predictions_{ablation}.csv", clip_rows, subject_rows)
        test_results[ablation] = metrics

    summary = {
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "permutation_seed": permutation_seed,
        "selection": {
            "metric": "subject_nll",
            "best_validation_loss": tracker.best_loss,
            "best_stage": tracker.best_stage,
            "best_epoch": tracker.best_epoch,
        },
        "test": test_results,
        "checkpoint_sha256": checkpoint_hash,
        "threshold": 0.5,
        "temperature_scaling": False,
    }
    _json_dump(run_dir / "metrics.json", summary)
    return summary
