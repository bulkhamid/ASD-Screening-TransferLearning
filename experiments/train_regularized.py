#!/usr/bin/env python
# coding: utf-8
"""
Regularized training script for ASD video screening models.

This is a simplified training script for quick experiments with either
Simple3DLSTM or Pretrained R3D models. It includes:
  - Weight decay regularization (AdamW optimizer)
  - Optional 3D dropout for LSTM (ignored for R3D)
  - Cosine learning rate scheduling with mixed precision training
  - Early stopping based on validation loss
  - Temperature scaling calibration (optional)
  - Threshold selection based on validation F1 score

Usage Examples
--------------
Train Simple3DLSTM:
    python experiments/train_regularized.py --model simple3dlstm --epochs 25

Train R3D with fine-tuning:
    python experiments/train_regularized.py --model r3d --finetune --epochs 30
"""
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

# Local imports
from src.utils.data_cached import make_train_loader, make_eval_loader
from src.utils.temperature_scaling import TemperatureScaler
from src.models.simple3dlstm import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, thr=0.5):
    """
    Compute classification metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thr: Decision threshold
    
    Returns:
        Dictionary of metrics (auc, acc, prec, rec, f1)
    """
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        auc=float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0)
    )


@torch.no_grad()
def eval_loader(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model: PyTorch model
        loader: DataLoader instance
        device: Device to run on
    
    Returns:
        Tuple of (true labels, predicted probabilities)
    """
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        logit = model(X.to(device))
        ys.extend(y.numpy())
        ps.extend(torch.sigmoid(logit).cpu().numpy())
    return np.array(ys), np.array(ps)

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Regularized training for ASD screening models"
    )
    parser.add_argument("--model", choices=["simple3dlstm", "r3d"],
                        required=True,
                        help="Model architecture")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs (default: 25)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--drop", type=float, default=0.0,
                        help="3D dropout for Simple3DLSTM (default: 0.0)")
    parser.add_argument("--target", nargs=2, type=int, default=[112, 112],
                        help="Target frame size (default: 112 112)")
    parser.add_argument("--temp", action="store_true",
                        help="Enable temperature scaling")
    parser.add_argument("--finetune", action="store_true",
                        help="Unfreeze layer4 for R3D fine-tuning")
    
    cfg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    tr_loader = make_train_loader(cfg.batch, target_size=tuple(cfg.target))
    vl_loader = make_eval_loader("val", cfg.batch)
    te_loader = make_eval_loader("test", cfg.batch)

    # Build model and optimizer
    if cfg.model == "simple3dlstm":
        model = Simple3DLSTM(60, tuple(cfg.target), p_drop=cfg.drop).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=cfg.wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=1e-6
        )
        stages = [("full", cfg.epochs, model.parameters())]

    else:  # R3D model
        model = PretrainedR3D().to(device)

        # Stage 1: Train only the newly-initialized FC layer
        for p in model.parameters():
            p.requires_grad_(False)
        model.net.fc.requires_grad_(True)

        opt = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=cfg.wd
        )
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs // 2, eta_min=1e-6
        )
        stages = [
            ("head", cfg.epochs // 2,
             filter(lambda p: p.requires_grad, model.parameters()))
        ]

        # Stage 2: Optionally unfreeze layer4 for light fine-tuning
        if cfg.finetune:
            stages.append(("ft", cfg.epochs // 2, []))  # Will be filled later

    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Training stages
    for stage_name, n_ep, params in stages:
        if stage_name == "ft":  # Unlock layer4 and FC
            for n, p in model.net.named_parameters():
                if "layer4" in n or "fc" in n:
                    p.requires_grad_(True)
                    params.append(p)
            opt = optim.AdamW(params, lr=1e-4, weight_decay=cfg.wd)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_ep, eta_min=1e-6
            )

        best_val, wait = float("inf"), 0
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name} (epochs={n_ep}, wd={cfg.wd})")
        print(f"{'='*60}")

        for ep in range(1, n_ep + 1):
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

            # Validation loss
            ys_val, ps_val = eval_loader(model, vl_loader, device)
            val_loss = crit(
                torch.from_numpy(ps_val).logit(),
                torch.from_numpy(ys_val)
            ).item()
            
            print(f"[{stage_name}] ep {ep:02d}  "
                  f"val_loss={val_loss:.4f}  "
                  f"time={time.time()-t0:.1f}s")

            # Early stopping
            if val_loss < best_val:
                best_val, wait = val_loss, 0
                torch.save(model.state_dict(), f"best_{stage_name}.pth")
            else:
                wait += 1
                if wait >= cfg.patience:
                    print("Early stopping this stage")
                    break

        # Reload best checkpoint
        model.load_state_dict(
            torch.load(f"best_{stage_name}.pth", map_location=device)
        )

    # Calibration with temperature scaling
    if cfg.temp:
        infer_net = TemperatureScaler(model).set_temperature(vl_loader, device)
        print(f"Temperature scaling applied (T = {infer_net.temperature.item():.2f})")
    else:
        infer_net = model

    # Pick decision threshold on calibrated validation set
    ys_val, ps_val = eval_loader(infer_net, vl_loader, device)
    ths = np.linspace(0.05, 0.95, 19)
    best_t = float(ths[int(np.argmax([f1_score(ys_val, ps_val >= t) for t in ths]))])
    print(f"\nChosen threshold: {best_t:.2f}")

    # Post-temperature-scaling validation diagnostics
    with torch.no_grad():
        logits_val, ys_val_t = [], []
        for X, y in vl_loader:
            logits_val.append(infer_net(X.to(device)))
            ys_val_t.append(y.to(device))

        logits_val = torch.cat(logits_val)
        ys_val_t = torch.cat(ys_val_t)

    val_loss_post = crit(logits_val, ys_val_t).item()
    val_auc_post = roc_auc_score(
        ys_val_t.cpu().numpy(),
        torch.sigmoid(logits_val).cpu().numpy()
    )
    print(f"Validation: loss={val_loss_post:.3f}, AUC={val_auc_post:.3f}")

    # Test set evaluation
    ys_te, ps_te = eval_loader(infer_net, te_loader, device)
    mets = compute_metrics(ys_te, ps_te, thr=best_t)
    
    print(f"\n{'='*60}")
    print("TEST METRICS")
    print(f"{'='*60}")
    for k, v in mets.items():
        print(f"{k:7s}: {v:.3f}")


if __name__ == "__main__":
    main()

