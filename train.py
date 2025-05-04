#!/usr/bin/env python
# coding: utf-8
"""
Subject‑level training for ASD video screening.

Highlights
----------
* 70/10/20 hold‑out **or** GroupKFold CV (‑‑cv)
* Optional cached tensors (‑‑cached)  → 3‑4 × faster I/O
* Cosine‑annealing LR, mixed‑precision, early‑stopping
* W&B logging + optional Grad‑CAM
"""

import os, time, json, random, argparse
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
import wandb

# ── project imports ──────────────────────────────────────────────────────────
from src.models.simple3dlstm   import Simple3DLSTM
from src.models.pretrained_r3d import PretrainedR3D
from src.utils.metrics         import compute_metrics

# “raw” pipeline
from data.dataloader  import VideoDataset, VideoFrameTransform, build_frame_aug
# cached pipeline
from src.utils.data_cached  import make_train_loader, make_eval_loader

# ─────────────────────────── helpers ─────────────────────────────────────────
def get_model(name, max_frames, target_size):
    if name == "simple3dlstm": return Simple3DLSTM(max_frames, target_size)
    if name == "r3d":          return PretrainedR3D()
    raise ValueError(f"Unknown model {name}")

def gather_paths(root):
    pos, neg = Path(root)/"positive", Path(root)/"negative"
    paths, labels = [], []
    for lab, folder in [(1,pos),(0,neg)]:
        for fn in folder.glob("*"):
            if fn.suffix.lower() in (".mp4",".webm",".avi"):
                paths.append(str(fn));  labels.append(lab)
    return np.array(paths), np.array(labels)

@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    ys, logits, losses = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        with autocast(device_type=device.type):
            logit = model(X);  loss = bce(logit, y)
        ys.extend(y.cpu().numpy());     logits.extend(logit.cpu().numpy())
        losses.append(loss.item())
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    out   = compute_metrics(np.array(ys), probs)
    out["loss"] = np.sum(losses) / len(loader.dataset)
    return out

def train_stage(model, tr_loader, vl_loader,
                epochs, lr, fold, stage, cfg, device):

    opt       = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=cfg.weight_decay)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
    crit      = nn.BCEWithLogitsLoss()
    scaler    = GradScaler()
    best_loss = np.inf;  wait = 0
    ckpt      = f"checkpoints/{cfg.model}_fold{fold}_{stage}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, epochs+1):
        t0 = time.time();  model.train()
        for X,y in tr_loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                loss = crit(model(X), y)
            scaler.scale(loss).backward();  scaler.step(opt);  scaler.update()
        sched.step()

        tr = eval_split(model, tr_loader, device)
        vl = eval_split(model, vl_loader, device)
        wandb.log({f"fold{fold}/{stage}/epoch":ep,
                   f"fold{fold}/{stage}/train_loss":tr["loss"],
                   f"fold{fold}/{stage}/val_loss":  vl["loss"],
                   f"fold{fold}/{stage}/val_auc":   vl["auc"],
                   f"fold{fold}/{stage}/epoch_time":time.time()-t0})

        if vl["loss"] < best_loss:
            best_loss, wait = vl["loss"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"→ Early stop fold{fold} {stage} @ epoch {ep}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))

# ─────────────────────────── main ────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=r"C:\Users\zhams\HWs\Autsim")
    p.add_argument("--model", required=True, choices=["simple3dlstm","r3d"])
    p.add_argument("--max_frames", type=int, default=60)
    p.add_argument("--target_size", nargs=2, type=int, default=[224,224])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs_head", type=int, default=30)
    p.add_argument("--epochs_ft",   type=int, default=50)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft",   type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--cv",    action="store_true")
    p.add_argument("--cached",action="store_true",
                   help="use pre‑saved .npz tensors instead of raw videos")
    p.add_argument("--gradcam",action="store_true")
    cfg = p.parse_args()

    # Reproducibility
    SEED = 42
    random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    wandb.init(project="ASD-SubjectCV",
               name=f"{cfg.model}_{'GKF' if cfg.cv else 'HOLD'}{cfg.splits}"
                    +("_cached" if cfg.cached else ""),
               config=vars(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build splits (only needed for raw mode)
    if not cfg.cached:
        paths, labels = gather_paths(cfg.data_path)
        subs   = np.array([Path(p).stem for p in paths])
        splits = []
        if cfg.cv:
            gkf = GroupKFold(cfg.splits)
            for f,(tr,rest) in enumerate(gkf.split(paths,labels,subs),1):
                val,test = train_test_split(rest, test_size=.5,
                                            stratify=labels[rest], random_state=f)
                splits.append((f,tr,val,test))
        else:
            uniq,inv = np.unique(subs, return_inverse=True)
            subj_lbl = {u:int(labels[subs==u][0]) for u in uniq}
            tr_s,te_s = train_test_split(uniq, test_size=.20,
                                         stratify=[subj_lbl[u] for u in uniq],
                                         random_state=SEED)
            tr_s,vl_s= train_test_split(tr_s, test_size=.125,
                                        stratify=[subj_lbl[u] for u in tr_s],
                                        random_state=SEED)
            tr = np.where(np.isin(subs,tr_s))[0]
            vl = np.where(np.isin(subs,vl_s))[0]
            te = np.where(np.isin(subs,te_s))[0]
            splits.append((1,tr,vl,te))

    all_results = []

    # ── loop over folds or hold‑out ──────────────────────────────────────────
    fold_ids = range(1, cfg.splits+1) if cfg.cv else [1]
    for f in fold_ids:
        print(f"\n=== Fold {f} ===" if cfg.cv else "\n=== Hold‑out ===")

        # ------------------------------------------------------------------ #
        # 1.  DATA LOADERS
        # ------------------------------------------------------------------ #
        if cfg.cached:
            tr_loader = make_train_loader(cfg.batch_size,
                                          target_size=tuple(cfg.target_size))
            vl_loader = make_eval_loader("val",  cfg.batch_size)
            te_loader = make_eval_loader("test", cfg.batch_size)
        else:
            tr_idx,val_idx,te_idx = splits[f-1][1:]
            aug = VideoFrameTransform(build_frame_aug(tuple(cfg.target_size)))
            tr_loader = DataLoader(VideoDataset(paths[tr_idx],  labels[tr_idx],
                                                cfg.max_frames, tuple(cfg.target_size),
                                                transform=aug),
                                   batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
            vl_loader = DataLoader(VideoDataset(paths[val_idx], labels[val_idx],
                                                cfg.max_frames, tuple(cfg.target_size)),
                                   batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)
            te_loader = DataLoader(VideoDataset(paths[te_idx], labels[te_idx],
                                                cfg.max_frames, tuple(cfg.target_size)),
                                   batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # ------------------------------------------------------------------ #
        # 2.  MODEL
        # ------------------------------------------------------------------ #
        model = get_model(cfg.model, cfg.max_frames, tuple(cfg.target_size)).to(device)
        wandb.watch(model, log="all", log_freq=10)

        if cfg.model == "r3d":
            # stage 1: head only
            for p in model.parameters(): p.requires_grad = False
            model.net.fc.requires_grad_(True)
            train_stage(model,tr_loader,vl_loader,
                        cfg.epochs_head,cfg.lr_head,f,"head",cfg,device)
            # stage 2: unfreeze last block
            for n,p in model.net.named_parameters():
                if "layer4" in n or "fc" in n: p.requires_grad_(True)
            train_stage(model,tr_loader,vl_loader,
                        cfg.epochs_ft,cfg.lr_ft,f,"ft",cfg,device)
        else:
            train_stage(model,tr_loader,vl_loader,
                        cfg.epochs_ft,cfg.lr_ft,f,"full",cfg,device)

        # ------------------------------------------------------------------ #
        # 3.  FINAL TEST
        # ------------------------------------------------------------------ #
        te = eval_split(model, te_loader, device)
        print(f"Fold {f} TEST → AUC={te['auc']:.4f}  "
              f"Prec={te['precision']:.4f}  Rec={te['recall']:.4f}  F1={te['f1']:.4f}")
        all_results.append(te)

    # ---------------------------------------------------------------------- #
    # 4.  SUMMARY
    # ---------------------------------------------------------------------- #
    summary = {k:float(np.mean([r[k] for r in all_results]))
               for k in ["auc","accuracy","precision","recall","f1","loss"]}
    print("\n==== Overall TEST summary ====")
    print(json.dumps(summary, indent=2))
    wandb.log({f"overall_{k}":v for k,v in summary.items()});  wandb.finish()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
