#!/usr/bin/env python
# coding: utf-8
"""
Subject‑level GroupKFold training for ASD video screening.

* **r3d**: two‑stage schedule (head‑only → fine‑tune last block)
* **simple3dlstm**: single‑stage end‑to‑end training ("full" stage)

Examples
--------
python train.py --model simple3dlstm
python train.py --model r3d --num_workers 2
"""

import os, argparse, json, platform, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold, train_test_split
import wandb
import re
from pathlib import Path
from typing import Tuple
import numpy as np

# ─── repo imports ─────────────────────────────────────────────────────────────
from data.dataloader import VideoDataset, VideoFrameTransform, build_frame_aug
from src.models.simple3dlstm   import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D
from src.utils.metrics         import compute_metrics

# ─── helper functions ─────────────────────────────────────────────────────────

def get_model(name: str, max_frames: int, target_size):
    if name == "simple3dlstm":
        return Simple3DLSTM(max_frames, target_size)
    if name == "r3d":
        return PretrainedR3D()
    raise ValueError(name)


def eval_split(model, loader, criterion, device):
    """Evaluate one dataloader and return metrics dict."""
    ys, ps, losses = [], [], []
    with torch.no_grad():
        model.eval()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            p    = model(X)
            loss = criterion(p, y)
            losses.append(loss.item() * y.size(0))
            ys.extend(y.cpu().numpy())
            ps.extend(p.cpu().numpy())
    out = compute_metrics(np.array(ys), np.array(ps))
    out["loss"] = sum(losses) / len(loader.dataset)
    return out


def train_stage(model, tr_loader, vl_loader,
                epochs, lr, stage, cfg, device):
    """Generic training loop with early‑stopping on val loss."""
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=cfg.weight_decay)
    sch = ReduceLROnPlateau(opt, "min", factor=0.5, patience=cfg.patience // 2)
    crit = nn.BCELoss()
    best, no_imp = np.inf, 0
    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()

        vl = eval_split(model, vl_loader, crit, device)
        sch.step(vl["loss"])
        wandb.log({f"{stage}/epoch": ep,
                   f"{stage}/val_auc": vl["auc"],
                   f"{stage}/val_loss": vl["loss"]})
        print(f"[{stage}] ep {ep:03d}  val_auc={vl['auc']:.3f}  loss={vl['loss']:.4f}")

        if vl["loss"] < best:
            best, no_imp = vl["loss"], 0
            torch.save(model.state_dict(), f"best_{stage}.pth")
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                break

    model.load_state_dict(torch.load(f"best_{stage}.pth"))


def _extract_id(fn: str) -> str:
    """
    Get a subject/session identifier from a filename.

    •  video_YYYY-MM-DD_HH-MM-SS.mp4  →  YYYY-MM-DD
    •  20240726072914.webm           →  20240726072914
    """
    stem = Path(fn).stem  # strip extension
    if stem.startswith("video_"):
        # video_2024-10-20_03-10-34 → 2024-10-20
        m = re.match(r"video_(\d{4}-\d{2}-\d{2})_", stem)
        return m.group(1) if m else stem          # fallback = whole stem
    return stem                                   # timestamp-only case


def gather_paths(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays: paths, labels, subject_ids."""
    pos = os.path.join(root, "positive")
    neg = os.path.join(root, "negative")
    paths, labels, subjects = [], [], []
    for lab, folder in [(1, pos), (0, neg)]:
        for fn in os.listdir(folder):
            if fn.lower().endswith((".mp4", ".webm", ".avi")):
                paths.append(os.path.join(folder, fn))
                labels.append(lab)
                subjects.append(_extract_id(fn))
    return np.array(paths), np.array(labels), np.array(subjects)

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    default_workers = 0 if platform.system() == "Windows" else 4

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",   default=r"C:\Users\zhams\HWs\Autsim")
    ap.add_argument("--model",       choices=["simple3dlstm", "r3d"], required=True)
    ap.add_argument("--max_frames",  type=int, default=60)
    ap.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--epochs_head", type=int, default=50)
    ap.add_argument("--epochs_ft",   type=int, default=50)
    ap.add_argument("--lr_head",     type=float, default=1e-3)
    ap.add_argument("--lr_ft",       type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience",     type=int,   default=8)
    ap.add_argument("--splits",       type=int,   default=5)
    ap.add_argument("--num_workers",  type=int,   default=default_workers)
    cfg = ap.parse_args()

    wandb.init(project="ASD-SubjectCV", config=vars(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data prep ────────────────────────────────────────────────────────────
    paths, labels, subs = gather_paths(cfg.data_path)
    gkf = GroupKFold(n_splits=cfg.splits)
    crit = nn.BCELoss()
    fold_results = []

    for fold, (tr_idx, tt_idx) in enumerate(gkf.split(paths, labels, groups=subs), 1):
        print(f"\n=== Fold {fold} ===")
        tr_p, tr_l = paths[tr_idx], labels[tr_idx]
        tt_p, tt_l = paths[tt_idx], labels[tt_idx]
        val_p, te_p, val_l, te_l = train_test_split(
            tt_p, tt_l, test_size=0.5, stratify=tt_l, random_state=fold)
        
        train_ids = {fn.split('_')[0] for fn in tr_p}
        val_ids   = {fn.split('_')[0] for fn in val_p}
        test_ids  = {fn.split('_')[0] for fn in te_p}

        assert train_ids.isdisjoint(val_ids),  "Train ↔ Val leakage!"
        assert train_ids.isdisjoint(test_ids), "Train ↔ Test leakage!"

        # --- quick leakage debug ---------------------------------
        leak_train_val = train_ids & val_ids
        leak_train_test = train_ids & test_ids
        print("Train ↔ Val overlap:", leak_train_val)
        print("Train ↔ Test overlap:", leak_train_test)
        # ---------------------------------------------------------
        # --- exploratory: how many unique subject IDs? ---
        print("Unique IDs in TRAIN:", len(train_ids))
        print("Unique IDs in VAL:  ", len(val_ids))
        print("Unique IDs in TEST: ", len(test_ids))
        # How many files per ID in train?
        from collections import Counter
        print("Most common train IDs:", Counter([_extract_id(os.path.basename(p)) for p in tr_p]).most_common(5))




        aug = VideoFrameTransform(build_frame_aug(tuple(cfg.target_size)))
        tr_loader = DataLoader(VideoDataset(tr_p, tr_l, cfg.max_frames, tuple(cfg.target_size), transform=aug),
                               batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
        vl_loader = DataLoader(VideoDataset(val_p, val_l, cfg.max_frames, tuple(cfg.target_size)),
                               batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True)
        te_loader = DataLoader(VideoDataset(te_p, te_l, cfg.max_frames, tuple(cfg.target_size)),
                               batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True)

        model = get_model(cfg.model, cfg.max_frames, tuple(cfg.target_size)).to(device)
        wandb.watch(model, log="all", log_freq=10)

        # ── training schedules ──────────────────────────────────────────────
        if cfg.model == "r3d":
            # Stage 1: head‑only
            for p in model.parameters():
                p.requires_grad = False
            model.net.fc.requires_grad_(True)
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_head, cfg.lr_head, f"{fold}_head", cfg, device)

            # Stage 2: fine‑tune last block
            for n, p in model.net.named_parameters():
                if "layer4" in n or "fc" in n:
                    p.requires_grad_(True)
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_ft, cfg.lr_ft, f"{fold}_ft", cfg, device)
        else:  # simple3dlstm – single full stage
            for p in model.parameters():
                p.requires_grad = True
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_ft, cfg.lr_ft, f"{fold}_full", cfg, device)

        # ── evaluation on held‑out test split ───────────────────────────────
        res = eval_split(model, te_loader, crit, device)
        print(f"Fold {fold}: test_auc {res['auc']:.3f}")
        fold_results.append(res)

    summary = {k: float(np.mean([fr[k] for fr in fold_results]))
               for k in ["auc", "accuracy", "f1", "precision", "recall", "loss"]}
    print("\n==== Overall (subject‑level CV) ====")
    print(json.dumps(summary, indent=2))
    wandb.log({f"overall_{k}": v for k, v in summary.items()})
    wandb.finish()

# ─── entry point (multiprocessing guard for Windows) ─────────────────────────
if __name__ == "__main__":
    main()
