#!/usr/bin/env python
"""
Label‑shuffle sanity‑check
─────────────────────────
Train exactly the same architecture you use in *train.py* but **after shuffling the
labels** inside every split.  If the pipeline is healthy the network should be
unable to exceed random‑chance (AUC ≈ 0.50, loss ≈ 0.69).

Usage
-----
$ python debug_shuffle_labels.py --model simple3dlstm --epochs 10 --batch_size 8

The script prints AUC / loss for train, val and test.  If you still see AUC ≫ 0.5
there is information leakage (frames duplicated across splits, subject overlap,
etc.).
"""

import argparse, random, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# ── local imports (assumes we run from repo root) ────────────────────────────
from src.models.simple3dlstm   import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D
from src.utils.data_cached     import load_split, _normalize_tensor  # _normalize_tensor is used inside custom collate

# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── reproducibility ─────────────────────────────────────────
SEED = 42                        # pick your favourite prime
import random, numpy as np, torch

random.seed(SEED)                # Python stdlib
np.random.seed(SEED)             # NumPy
torch.manual_seed(SEED)          # CPU RNG
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True   # slower but repeatable
torch.backends.cudnn.benchmark     = False
# ---------------------------------------------------------------------------


def get_model(name, max_frames, target_size):
    if name == "simple3dlstm":
        return Simple3DLSTM(max_frames, target_size)
    if name == "r3d":
        return PretrainedR3D()
    raise ValueError(f"Unknown model: {name}")

# ---------------------------------------------------------------------------

def build_loader(split, batch_size, shuffle_labels=False):
    """Return DataLoader built from cached .npz split.
    If *shuffle_labels* is True, permute the labels within this split."""
    ds = load_split(split)                     # TensorDataset(vids, labels)
    if shuffle_labels:
        idx = torch.randperm(len(ds.tensors[1]))
        ds.tensors = (ds.tensors[0], ds.tensors[1][idx])

    def collate(batch):
        vids, lbls = zip(*batch)
        vids = torch.stack([_normalize_tensor(v.float() / 255.) for v in vids])
        return vids, torch.stack(lbls)

    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate, num_workers=0, pin_memory=True)

# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    ys, pr, losses = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with autocast(device_type=device.type):
                log = model(X)
                loss = bce(log, y)
            probs = torch.sigmoid(log).detach().cpu().numpy()
            ys.extend(y.cpu().numpy()); pr.extend(probs); losses.append(loss.item())
    ys = np.array(ys); pr = np.array(pr)
    auc = roc_auc_score(ys, pr) if len(np.unique(ys)) > 1 else 0.5
    y_pred = (pr >= 0.5).astype(int)
    return {
        "loss": np.sum(losses)/len(loader.dataset),
        "auc": auc,
        "accuracy": accuracy_score(ys, y_pred),
        "precision": precision_score(ys, y_pred, zero_division=0),
        "recall": recall_score(ys, y_pred, zero_division=0),
        "f1": f1_score(ys, y_pred, zero_division=0),
    }

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simple3dlstm", "r3d"], default="simple3dlstm")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ───────────────────────────────────────────────────────────────
    tr_loader = build_loader("train", args.batch_size, shuffle_labels=True)
    vl_loader = build_loader("val",   args.batch_size, shuffle_labels=True)
    te_loader = build_loader("test",  args.batch_size, shuffle_labels=True)

    # ── model & optimiser ──────────────────────────────────────────────────
    model = get_model(args.model, args.max_frames, tuple(args.target_size)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce   = nn.BCEWithLogitsLoss()
    scaler= GradScaler()

    # ── training loop ──────────────────────────────────────────────────────
    for ep in range(1, args.epochs+1):
        t0 = time.time(); model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                loss = bce(model(X), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sec = time.time()-t0

        tr = evaluate(model, tr_loader, device)
        vl = evaluate(model, vl_loader, device)
        print(f"ep {ep:02d}  train AUC={tr['auc']:.3f}  val AUC={vl['auc']:.3f}  time={sec:.1f}s")

    # ── final test ─────────────────────────────────────────────────────────
    te = evaluate(model, te_loader, device)
    print("\n=== Shuffled‑label performance ===")
    print(f"Train AUC {tr['auc']:.3f}   Val AUC {vl['auc']:.3f}   Test AUC {te['auc']:.3f}")
    print(f"Test loss {te['loss']:.3f}  Accuracy {te['accuracy']:.3f}")

    if te['auc'] > 0.65:
        print("AUC is well above chance – investigate potential leakage!")
    else:
        print(" Looks good – model can’t learn from scrambled labels.")

if __name__ == "__main__":
    main()
