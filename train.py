#!/usr/bin/env python
# coding: utf-8
"""
Subject‑level training for ASD video screening.

Key features
------------
* 70 / 10 / 20 split **or** GroupKFold CV toggled by --cv
* Cosine‑annealing learning‑rate schedule
* Mixed precision (torch.cuda.amp) **with** BCEWithLogitsLoss (safe)
* Fast DataLoader: pin_memory, persistent_workers, prefetch_factor
* W&B logging (loss, AUC) + optional confusion‑matrix / ROC / PR plots
* Optional Grad‑CAM visualisation (if torchcam is installed)
"""

import os, argparse, json, re, time
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
import wandb

from data.dataloader import VideoDataset, VideoFrameTransform, build_frame_aug
from src.models.simple3dlstm   import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D
from src.utils.metrics         import compute_metrics

# ────────────────────────── helpers ───────────────────────────────────────────


def get_model(name: str, max_frames: int, target_size):
    if name == "simple3dlstm":
        return Simple3DLSTM(max_frames, target_size)          # returns **raw logits**
    if name == "r3d":
        return PretrainedR3D()                                # returns **raw logits**
    raise ValueError(f"Unknown model: {name}")


def _extract_id(fn: str) -> str:
    """Derive a subject/session id from filename."""
    stem = Path(fn).stem
    # if stem.startswith("video_"):
    #     m = re.match(r"video_(\d{4}-\d{2}-\d{2})_", stem)
    #     return m.group(1) if m else stem
    return stem


def gather_paths(root: str):
    """Return arrays: video paths | labels | subject ids."""
    pos, neg = os.path.join(root, "positive"), os.path.join(root, "negative")
    paths, labels, subs = [], [], []
    for lab, folder in [(1, pos), (0, neg)]:
        for fn in os.listdir(folder):
            if fn.lower().endswith((".mp4", ".webm", ".avi")):
                paths.append(os.path.join(folder, fn))
                labels.append(lab)
                subs.append(_extract_id(fn))
    return np.array(paths), np.array(labels), np.array(subs)


@torch.no_grad()
def eval_split(model, loader, device):
    """Run model on loader and compute metrics."""
    model.eval()
    ys, logits, losses = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with autocast(device_type=device.type):
            logit = model(X)
            loss = bce(logit, y)
        ys.extend(y.cpu().numpy())
        logits.extend(logit.cpu().numpy())
        losses.append(loss.item())
    ys = np.array(ys)
    logits = np.array(logits)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()

    out = compute_metrics(ys, probs)
    out["loss"] = np.sum(losses) / len(loader.dataset)
    out["y_true"], out["y_score"], out["y_pred"] = ys, probs.tolist(), (probs >= 0.5).astype(int).tolist()
    return out


def train_stage(model, tr_loader, vl_loader,
                epochs, lr, fold, stage, cfg, device):
    """Generic training loop."""
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_loss, patience_counter = float("inf"), 0
    ckpt_path = f"checkpoints/{cfg.model}_fold{fold}_{stage}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, epochs+1):
        t0 = time.time()

        # ── training loop ──────────────────────────────────────────────────
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                loss = crit(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        scheduler.step()

        # ── eval ──────────────────────────────────────────────────────────
        tr_met = eval_split(model, tr_loader, device)
        vl_met = eval_split(model, vl_loader, device)

        # ── measure time ─────────────────────────────────────────────────
        epoch_time = time.time() - t0

        wandb.log({
            f"fold{fold}/{stage}/epoch":        ep,
            f"fold{fold}/{stage}/train_loss":   tr_met["loss"],
            f"fold{fold}/{stage}/val_loss":     vl_met["loss"],
            f"fold{fold}/{stage}/val_auc":      vl_met["auc"],
            f"fold{fold}/{stage}/epoch_time":   epoch_time,
        })
        print(f"[Fold {fold}][{stage}] ep {ep:03d}  "
              f"train_loss={tr_met['loss']:.4f}  "
              f"val_loss={vl_met['loss']:.4f}  "
              f"val_auc={vl_met['auc']:.3f}  "
              f"time={epoch_time:.1f}s")

        # ── early‑stopping ───────────────────────────────────────────────────
        if vl_met["loss"] < best_val_loss:
            best_val_loss, patience_counter = vl_met["loss"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"→ Early‑stopping fold {fold} {stage} at epoch {ep}")
                break

    # load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


# ────────────────────────── main ─────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=r"C:\Users\zhams\HWs\Autsim")
    ap.add_argument("--model", choices=["simple3dlstm", "r3d"], required=True)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--target_size", nargs=2, type=int, default=[112, 112])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs_head", type=int, default=50)
    ap.add_argument("--epochs_ft", type=int, default=50)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--splits", type=int, default=5, help="folds if --cv")
    ap.add_argument("--cv", action="store_true", help="GroupKFold CV (else hold‑out)")
    cfg = ap.parse_args()

    # ─── reproducibility ─────────────────────────────────────────────────────────
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project="ASD-SubjectCV",
               name=f"{cfg.model}_{'GKF' if cfg.cv else 'HOLD'}{cfg.splits}",
               config=vars(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── build splits ─────────────────────────────────────────────────────────
   # ── build explicit (train, val, test) splits ───────────────────────────────
    paths, labels, subs = gather_paths(cfg.data_path)
    splits = []

    if cfg.cv:
        gkf = GroupKFold(n_splits=cfg.splits)
        for fold, (tr_idx, rest_idx) in enumerate(
                gkf.split(paths, labels, groups=subs), 1):
            # split the “rest” 50/50 into val vs test
            val_idx, test_idx = train_test_split(
                rest_idx,
                test_size=0.50,
                stratify=labels[rest_idx],
                random_state=fold
            )
            splits.append((fold, tr_idx, val_idx, test_idx))
    else:
        # ── subject-level hold-out split: 70/10/20 ────────────────────────────
        # get unique subjects and their labels
        unique_subs, inv = np.unique(subs, return_inverse=True)
        # assign each subject the label of its first video
        subj_label_map = {s: int(labels[subs==s][0]) for s in unique_subs}

        # split subjects into train vs test (80/20)
        train_subs, test_subs = train_test_split(
            unique_subs,
            test_size=0.20,
            stratify=[subj_label_map[s] for s in unique_subs],
            random_state=42
        )
        # split train_subs into train vs val (87.5/12.5 of the 80% → 70/10 overall)
        train_subs, val_subs = train_test_split(
            train_subs,
            test_size=0.125,  # = 0.10 / 0.80
            stratify=[subj_label_map[s] for s in train_subs],
            random_state=42
        )

        # now collect video‐indices for each split
        tr_idx  = np.where(np.isin(subs, train_subs))[0]
        val_idx = np.where(np.isin(subs,   val_subs))[0]
        te_idx  = np.where(np.isin(subs,  test_subs))[0]

        splits.append((1, tr_idx, val_idx, te_idx))

    crit = nn.BCEWithLogitsLoss()
    all_results = []

    # ── train / val / test loop ────────────────────────────────────────────────
    for fold, train_idx, val_idx, test_idx in splits:
        if cfg.cv:
            print(f"\n=== Fold {fold} ===")
        else:
            print("\n=== Hold-out split ===")

        # ── leakage checks ────────────────────────────────────────────────────
        train_ids = set(subs[train_idx])
        val_ids   = set(subs[val_idx])
        test_ids  = set(subs[test_idx])
        assert train_ids.isdisjoint(val_ids),  f"Train⇄Val leakage: {train_ids & val_ids}"
        assert train_ids.isdisjoint(test_ids), f"Train⇄Test leakage: {train_ids & test_ids}"
        assert val_ids.isdisjoint(test_ids),   f"Val⇄Test leakage:   {val_ids   & test_ids}"

        # ── build loaders ─────────────────────────────────────────────────────
        aug = VideoFrameTransform(build_frame_aug(tuple(cfg.target_size)))
        tr_loader = DataLoader(
            VideoDataset(paths[train_idx], labels[train_idx],
                         cfg.max_frames, tuple(cfg.target_size),
                         transform=aug),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )
        vl_loader = DataLoader(
            VideoDataset(paths[val_idx], labels[val_idx],
                         cfg.max_frames, tuple(cfg.target_size)),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )
        te_loader = DataLoader(
            VideoDataset(paths[test_idx], labels[test_idx],
                         cfg.max_frames, tuple(cfg.target_size)),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        )

        # ── model ────────────────────────────────────────────────────────────
        model = get_model(cfg.model, cfg.max_frames, tuple(cfg.target_size)).to(device)
        wandb.watch(model, log="all", log_freq=10)

        if cfg.model == "r3d":
            # stage 1 – head only
            for p in model.parameters(): p.requires_grad = False
            model.net.fc.requires_grad_(True)
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_head, cfg.lr_head, fold, "head", cfg, device)
            # stage 2 – fine‑tune last block
            for n, p in model.net.named_parameters():
                if "layer4" in n or "fc" in n:
                    p.requires_grad_(True)
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_ft, cfg.lr_ft, fold, "ft", cfg, device)
        else:
            # single stage
            train_stage(model, tr_loader, vl_loader,
                        cfg.epochs_ft, cfg.lr_ft, fold, "full", cfg, device)

        # ── final TEST evaluation ───────────────────────────────────────────
        test_met = eval_split(model, te_loader, device)
        print(f"\nFold {fold} TEST → "
              f"AUC={test_met['auc']:.4f}  "
              f"Prec={test_met['precision']:.4f}  "
              f"Rec={test_met['recall']:.4f}  "
              f"F1={test_met['f1']:.4f}")

        
        # W&B plots (only if both classes present)
        if len(set(test_met["y_true"])) == 2:
            # build a (N,2) array of [P(neg), P(pos)]
            pos = np.array(test_met["y_score"], dtype=np.float32)
            neg = 1.0 - pos
            probas = np.vstack([neg, pos]).T

            wandb.log({
                f"fold{fold}/test_confmat":
                    wandb.plot.confusion_matrix(
                        y_true=test_met["y_true"],
                        preds=test_met["y_pred"],
                        class_names=["neg", "pos"]
                    ),
                f"fold{fold}/test_roc":
                    wandb.plot.roc_curve(
                        y_true=test_met["y_true"],
                        y_probas=probas
                    ),
                f"fold{fold}/test_pr":
                    wandb.plot.pr_curve(
                        y_true=test_met["y_true"],
                        y_probas=probas
                    ),
            })


        # optional Grad-CAM
        try:
            from torchcam.methods import CAM

            # 1) Pick your backbone and the layer to inspect
            backbone     = model.net     if cfg.model == "r3d" else model.cnn
            target_layer = (backbone.layer4[-1] 
                            if cfg.model == "r3d" 
                            else backbone[-1])

            # 2) Instantiate the CAM extractor
            cam_extractor = CAM(
                model=backbone,
                target_layer=target_layer,
                fc_layer="fc"              # torchvision’s classifier is named "fc"
            )

            # 3) Grab exactly one sample from the test loader
            X_batch, y_batch = next(iter(te_loader))
            X1     = X_batch[:1].to(device)    # shape [1, C, T, H, W]
            label1 = int(y_batch[0].item())    # 0 or 1

            # 4) Forward + get the CAM map
            #    CAM’s signature is CAM(input_tensor, class_idx_list)
            cams = cam_extractor(X1, [label1])

            # 5) Extract the middle frame’s heatmap and overlay it
            heat  = cams[target_layer][0, cfg.max_frames // 2].cpu().numpy()
            frame = X1[0].permute(1,2,3,0)[cfg.max_frames // 2].cpu().numpy()
            heat  = np.clip(heat, 0, 1)[..., None]
            overlay = np.uint8((frame + heat * np.array([1, 0, 0])) 
                               / np.max(frame + heat) * 255)

            # 6) Log the CAM image to W&B
            wandb.log({f"fold{fold}/cam": wandb.Image(overlay)})
                
        except ImportError:
            print("torchcam not installed - skipping visualization")
        except Exception as e:
            print(f"GradCAM visualization skipped: {str(e)}")

        all_results.append(test_met)

    # ── global summary ───────────────────────────────────────────────────────
    summary = {k: float(np.mean([r[k] for r in all_results]))
               for k in ["auc", "accuracy", "precision", "recall", "f1", "loss"]}
    print("\n==== Overall TEST summary ====")
    print(json.dumps(summary, indent=2))
    wandb.log({f"overall_test_{k}": v for k, v in summary.items()})
    wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
