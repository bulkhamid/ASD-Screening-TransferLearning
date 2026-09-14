#!/usr/bin/env python
"""Legacy holdout label-permutation sanity check.

Only training labels are permuted. Validation and test labels remain untouched,
so their metrics still measure whether the model generalizes under the null.
For inferential experiments, use the grouped permutation option in
``experiments.run_subject_benchmark`` after subject metadata is completed.
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.pretrained_r3d import PretrainedR3D
from src.models.simple2dcnn import Simple2DCNN
from src.models.simple3dlstm import Simple3DLSTM
from src.utils.data_cached import _normalize_tensor, load_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(name, max_frames, target_size, r3d_stage):
    if name == "simple3dlstm":
        return Simple3DLSTM(max_frames, target_size)
    if name == "simple2dcnn":
        return Simple2DCNN()
    if name == "r3d":
        model = PretrainedR3D()
        model.configure_stage(r3d_stage)
        return model
    raise ValueError(f"Unknown model: {name}")


def build_loader(
    split,
    batch_size,
    *,
    data_path,
    permute_training_labels=False,
    shuffle_batches=False,
    seed=42,
):
    dataset = load_split(split, data_dir=data_path)
    if permute_training_labels:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset.tensors[1]), generator=generator)
        dataset.tensors = (dataset.tensors[0], dataset.tensors[1][order])

    def collate(batch):
        videos, labels = zip(*batch)
        normalized = torch.stack([_normalize_tensor(video.float()) for video in videos])
        return normalized, torch.stack(labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_batches,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels_all, logits_all = [], []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(videos)
            loss = criterion(logits, labels)
        labels_all.extend(labels.cpu().numpy())
        logits_all.extend(logits.float().cpu().numpy())
        total_loss += float(loss.cpu())
    labels_array = np.asarray(labels_all)
    logits_array = np.asarray(logits_all)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -60, 60)))
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": float(roc_auc_score(labels_array, logits_array)) if len(np.unique(labels_array)) > 1 else 0.5,
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple2dcnn", "simple3dlstm", "r3d"], default="simple3dlstm")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r3d_stage", choices=["head", "layer4", "full"], default="layer4")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(
        "train",
        args.batch_size,
        data_path=args.data_path,
        permute_training_labels=True,
        shuffle_batches=True,
        seed=args.seed,
    )
    val_loader = build_loader("val", args.batch_size, data_path=args.data_path)
    test_loader = build_loader("test", args.batch_size, data_path=args.data_path)

    model = get_model(args.model, args.max_frames, tuple(args.target_size), args.r3d_stage).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(videos), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d} train_auc={train_metrics['auc']:.3f} "
            f"val_auc={val_metrics['auc']:.3f} seconds={time.time() - started:.1f}"
        )

    test_metrics = evaluate(model, test_loader, device)
    print("\nLegacy training-label permutation result (validation/test labels unchanged)")
    print(
        f"train AUC {train_metrics['auc']:.3f} val AUC {val_metrics['auc']:.3f} "
        f"test AUC {test_metrics['auc']:.3f} test loss {test_metrics['loss']:.3f}"
    )
    print("This legacy video-level result is diagnostic only, not subject-independent evidence.")


if __name__ == "__main__":
    main()
