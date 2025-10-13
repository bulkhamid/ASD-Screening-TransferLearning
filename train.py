#!/usr/bin/env python
# coding: utf-8
"""
Cross-validated training for ASD video screening.

This script trains deep learning models for autism spectrum disorder (ASD)
screening from video data. It supports two model architectures:
  - Simple3DLSTM: Custom 3D CNN + LSTM architecture with optional dropout
  - Pre-trained R3D-18: Transfer learning from ResNet3D (head-only or fine-tuned)

Key Features
------------
* GroupKFold 5-fold cross-validation (--cv) or 70/10/20 holdout split
* Supports cached .npz tensor files (--cached) or raw video processing
* AdamW optimizer with cosine learning rate scheduling
* Mixed precision training (AMP) for faster training
* Early stopping with configurable patience
* Temperature scaling calibration (--temp) for better probability estimates
* Automatic threshold selection based on validation F1 score
* Generates loss/AUC curves and ROC plots for each fold
* Weights & Biases (wandb) logging support

Usage Examples
--------------
Train Simple3DLSTM with cached data:
    python train.py --model simple3dlstm --cached --batch_size 8 --epochs 50

Train R3D with fine-tuning:
    python train.py --model r3d --cached --finetune --batch_size 4 --epochs 40
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)

import matplotlib
import matplotlib.pyplot as plt
import wandb  # Comment out if not needed

# Local imports
from src.models.simple3dlstm import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D
from src.utils.metrics import compute_metrics
from src.utils.temperature_scaling import TemperatureScaler

# Data loading pipelines
from data.dataloader import VideoDataset, VideoFrameTransform, build_frame_aug
from src.utils.data_cached import make_train_loader, make_eval_loader


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gather_video_paths(root):
    """
    Collect video file paths and labels from positive/negative directories.
    
    Args:
        root: Root directory containing 'positive' and 'negative' subdirectories
    
    Returns:
        Tuple of (paths array, labels array) where labels are 1 for positive, 0 for negative
    """
    pos, neg = Path(root) / "positive", Path(root) / "negative"
    paths, labels = [], []
    for lab, folder in [(1, pos), (0, neg)]:
        for fn in folder.glob("*"):
            if fn.suffix.lower() in (".mp4", ".webm", ".avi"):
                paths.append(str(fn))
                labels.append(lab)
    return np.array(paths), np.array(labels)

@torch.no_grad()
def eval_loader(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Computes predictions, probabilities, and loss for all batches in the loader.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader instance
        device: Device to run evaluation on (cuda/cpu)
    
    Returns:
        Tuple of (metrics dict, true labels, probabilities, logits)
    """
    model.eval()
    ys, logits = [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    running_loss = 0.0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with autocast(device_type=device.type):
            logit = model(X)
            loss = bce(logit, y)
        logits.extend(logit.cpu().numpy())
        ys.extend(y.cpu().numpy())
        running_loss += loss.item()
    
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    metrics = compute_metrics(np.array(ys), probs)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics, np.array(ys), probs, np.array(logits)


def calibrate_and_threshold(model, val_loader, device, apply_ts):
    """
    Optionally apply temperature scaling and find optimal threshold.
    
    Temperature scaling improves probability calibration. The threshold is
    chosen to maximize F1 score on the validation set.
    
    Args:
        model: PyTorch model to calibrate
        val_loader: Validation data loader
        device: Device to run calibration on
        apply_ts: Whether to apply temperature scaling
    
    Returns:
        Tuple of (calibrated model, optimal threshold)
    """
    net = (TemperatureScaler(model)
           .set_temperature(val_loader, device)) if apply_ts else model
    _, y, p, _ = eval_loader(net, val_loader, device)
    ths = np.linspace(0.05, 0.95, 19)
    thr = float(ths[np.argmax([f1_score(y, p >= t) for t in ths])])
    return net, thr


def plot_curves(curve_dict, fpath_prefix):
    """
    Plot training and validation loss/AUC curves.
    
    Args:
        curve_dict: Dictionary with keys: epoch, train_loss, val_loss, train_auc, val_auc
        fpath_prefix: File path prefix for saving plots
    """
    ep = curve_dict["epoch"]
    tl = curve_dict["train_loss"]
    vl = curve_dict["val_loss"]
    ta = curve_dict["train_auc"]
    va = curve_dict["val_auc"]

    # Loss plot
    plt.figure()
    plt.plot(ep, tl, label="train")
    plt.plot(ep, vl, label="val")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fpath_prefix}_loss.png")
    plt.close()

    # AUC plot
    plt.figure()
    plt.plot(ep, ta, label="train")
    plt.plot(ep, va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fpath_prefix}_auc.png")
    plt.close()


def plot_roc(y_true, y_prob, fpath):
    """
    Plot ROC curve and save to file.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        fpath: File path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    A = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {A:.3f}")
    plt.plot([0, 1], [0, 1], "--k")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


# ---------------------------------------------------------------------------
# Training Stage
# ---------------------------------------------------------------------------
def train_stage(model, stage_name, tr_loader, vl_loader,
                cfg, device, lr, n_epochs, fold):
    """
    Train model for one stage (e.g., head-only or fine-tuning).
    
    Args:
        model: PyTorch model to train
        stage_name: Name of training stage (for logging)
        tr_loader: Training data loader
        vl_loader: Validation data loader
        cfg: Configuration object with hyperparameters
        device: Device to train on (cuda/cpu)
        lr: Learning rate
        n_epochs: Number of epochs to train
        fold: Current fold number (for logging)
    
    Returns:
        Dictionary with training curves (epoch, train_loss, val_loss, train_auc, val_auc)
    """
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.wd
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_loss, wait = np.inf, 0
    ckpt = f"checkpoints/{cfg.model}_fold{fold}_{stage_name}.pth"

    curve = defaultdict(list)  # For plotting

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                loss = crit(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        
        sched.step()

        # Evaluate on train and validation sets
        tr_m, _, _, _ = eval_loader(model, tr_loader, device)
        vl_m, _, _, _ = eval_loader(model, vl_loader, device)

        # Record curves
        curve["epoch"].append(ep)
        curve["train_loss"].append(tr_m["loss"])
        curve["val_loss"].append(vl_m["loss"])
        curve["train_auc"].append(tr_m["auc"])
        curve["val_auc"].append(vl_m["auc"])

        # Log to wandb (optional)
        wandb.log({
            f"fold{fold}/{stage_name}/train_loss": tr_m["loss"],
            f"fold{fold}/{stage_name}/val_loss": vl_m["loss"],
            f"fold{fold}/{stage_name}/val_auc": vl_m["auc"],
            f"fold{fold}/{stage_name}/epoch": ep
        })

        print(f"[Fold {fold}][{stage_name}] ep {ep:03d}  "
              f"train={tr_m['loss']:.4f}  val={vl_m['loss']:.4f}  "
              f"valAUC={vl_m['auc']:.3f}  time={time.time()-t0:.1f}s")

        # Early stopping
        if vl_m["loss"] < best_loss:
            best_loss, wait = vl_m["loss"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping {stage_name}")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return curve


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------
def main():
    """Main training loop with cross-validation or holdout split."""
    parser = argparse.ArgumentParser(
        description="Train ASD screening models with cross-validation"
    )
    
    # Data and cross-validation arguments
    parser.add_argument("--data_path", default="data",
                        help="Path to data directory")
    parser.add_argument("--cv", action="store_true", default=True,
                        help="Use 5-fold GroupKFold cross-validation (default: True)")
    parser.add_argument("--cached", action="store_true",
                        help="Load pre-processed .npz tensors instead of raw videos")
    parser.add_argument("--splits", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    
    # Model arguments
    parser.add_argument("--model", required=True,
                        choices=["simple3dlstm", "r3d"],
                        help="Model architecture to use")
    parser.add_argument("--finetune", action="store_true",
                        help="Unfreeze layer4 for R3D fine-tuning (ignored for LSTM)")
    parser.add_argument("--drop", type=float, default=0.0,
                        help="3D dropout probability for Simple3DLSTM (default: 0.0)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--temp", action="store_true",
                        help="Enable temperature scaling calibration")
    
    # Miscellaneous arguments
    parser.add_argument("--max_frames", type=int, default=60,
                        help="Maximum frames per video (default: 60)")
    parser.add_argument("--target_size", nargs=2, type=int, default=[112, 112],
                        help="Target frame size [height width] (default: 112 112)")
    
    cfg = parser.parse_args()

    # Setup
    set_seed(42)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    wandb.init(
        project="ASD-SubjectCV",
        name=f"{cfg.model}_{'cv' if cfg.cv else 'hold'}",
        config=vars(cfg)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data splits
    if cfg.cached is False:
        paths, labels = gather_video_paths(cfg.data_path)
        subs = np.array([Path(p).stem for p in paths])

        splits = []
        if cfg.cv:
            # 5-fold cross-validation with GroupKFold
            gkf = GroupKFold(cfg.splits)
            for i, (tr, rest) in enumerate(gkf.split(paths, labels, subs), 1):
                val, test = train_test_split(
                    rest, test_size=0.5,
                    stratify=labels[rest], random_state=i
                )
                splits.append((tr, val, test))
        else:
            # Single 70/10/20 holdout split
            uniq = np.unique(subs)
            strat = np.array([labels[subs == u][0] for u in uniq])
            tr_s, te_s = train_test_split(
                uniq, test_size=0.20,
                stratify=strat, random_state=42
            )
            tr_s, vl_s = train_test_split(
                tr_s, test_size=0.125,
                stratify=[strat[list(uniq).index(u)] for u in tr_s],
                random_state=42
            )
            tr = np.where(np.isin(subs, tr_s))[0]
            val = np.where(np.isin(subs, vl_s))[0]
            test = np.where(np.isin(subs, te_s))[0]
            splits.append((tr, val, test))

    # Loop over folds
    fold_metrics = []
    for fold_idx, split in enumerate(range(cfg.splits) if cfg.cv else [0], 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}")
        print(f"{'='*60}")

        # Create data loaders
        if cfg.cached:
            tr_loader = make_train_loader(
                cfg.batch_size,
                target_size=tuple(cfg.target_size)
            )
            vl_loader = make_eval_loader("val", cfg.batch_size)
            te_loader = make_eval_loader("test", cfg.batch_size)
        else:
            tr_idx, val_idx, test_idx = splits[fold_idx - 1]
            aug = VideoFrameTransform(build_frame_aug(tuple(cfg.target_size)))

            tr_loader = DataLoader(
                VideoDataset(
                    paths[tr_idx], labels[tr_idx],
                    cfg.max_frames, tuple(cfg.target_size),
                    transform=aug
                ),
                batch_size=cfg.batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            vl_loader = DataLoader(
                VideoDataset(
                    paths[val_idx], labels[val_idx],
                    cfg.max_frames, tuple(cfg.target_size)
                ),
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            te_loader = DataLoader(
                VideoDataset(
                    paths[test_idx], labels[test_idx],
                    cfg.max_frames, tuple(cfg.target_size)
                ),
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

        # Build model
        if cfg.model == "simple3dlstm":
            model = Simple3DLSTM(
                cfg.max_frames, tuple(cfg.target_size),
                p_drop=cfg.drop
            ).to(device)
            stages = [dict(name="full", epochs=cfg.epochs, lr=1e-4)]
        else:  # R3D-18
            model = PretrainedR3D().to(device)
            # Freeze all parameters
            for p in model.parameters():
                p.requires_grad_(False)
            model.net.fc.requires_grad_(True)
            
            stages = [dict(name="head", epochs=cfg.epochs // 2, lr=1e-3)]
            if cfg.finetune:
                stages.append(
                    dict(name="ft", epochs=cfg.epochs // 2, lr=1e-4,
                         unfreeze=["layer4", "fc"])
                )

        # Train in stages
        curves = defaultdict(list)
        for st in stages:
            # Unfreeze layers if needed
            if "unfreeze" in st:
                for n, p in model.net.named_parameters():
                    if any(k in n for k in st["unfreeze"]):
                        p.requires_grad_(True)

            c = train_stage(
                model, st["name"], tr_loader, vl_loader,
                cfg, device, st["lr"], st["epochs"], fold_idx
            )
            for k, v in c.items():
                curves[k].extend(v)

        # Calibration and threshold selection
        infer_net, thr = calibrate_and_threshold(
            model, vl_loader, device, cfg.temp
        )

        # Evaluation
        vl_m, _, _, _ = eval_loader(infer_net, vl_loader, device)
        te_m, ys_te, ps_te, _ = eval_loader(infer_net, te_loader, device)
        print(f"TEST  AUC={te_m['auc']:.3f}  F1={te_m['f1']:.3f}  "
              f"loss={te_m['loss']:.4f}")
        fold_metrics.append(te_m)

        # Save plots and checkpoints
        prefix = f"plots/{cfg.model}_fold{fold_idx}"
        plot_curves(curves, prefix)
        plot_roc(ys_te, ps_te, f"{prefix}_roc.png")
        torch.save(
            model.state_dict(),
            f"checkpoints/{cfg.model}_fold{fold_idx}.pth"
        )

    # Aggregate results across folds
    df_keys = ["auc", "accuracy", "precision", "recall", "f1", "loss"]
    df = {k: [m[k] for m in fold_metrics] for k in df_keys}
    mean = {k: float(np.mean(v)) for k, v in df.items()}
    std = {k: float(np.std(v)) for k, v in df.items()}

    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary ({cfg.splits} folds)")
    print(f"{'='*60}")
    for k in df_keys:
        print(f"{k:9s}: {mean[k]:.3f} ± {std[k]:.3f}")
    
    wandb.log({f"cv_mean_{k}": mean[k] for k in df_keys})
    wandb.log({f"cv_std_{k}": std[k] for k in df_keys})
    wandb.finish()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
