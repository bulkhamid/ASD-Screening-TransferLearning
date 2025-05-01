#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import wandb

# 1. Configuration
CONFIG = {
    "data_path":        r"C:\Users\zhams\HWs\Autsim",
    "target_size":      (112, 112),
    "max_frames":       60,
    "batch_size":       8,
    "epochs_head":      100,
    "epochs_finetune":  100,
    "lr_head":          1e-3,
    "lr_finetune":      1e-4,
    "weight_decay":     1e-5,
    "patience":         10,
    "n_splits":         5,
    "random_seed":      42
}

# 2. Initialize W&B
wandb.init(project="ASD-video-screening", config=CONFIG)
config   = wandb.config
run_id   = wandb.run.id
ckpt_root = os.path.join("checkpoints", run_id)
os.makedirs(ckpt_root, exist_ok=True)

# 3. Dataset
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels      = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path  = self.video_paths[idx]
        label = self.labels[idx]
        cap   = cv2.VideoCapture(path)
        frames = []
        while len(frames) < config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, config.target_size)
            frames.append(frame)
        cap.release()
        if len(frames) < config.max_frames:
            pad = frames[-1] if frames else np.zeros((*config.target_size,3),np.uint8)
            frames += [pad] * (config.max_frames - len(frames))
        video_np = np.array(frames, dtype=np.float32) / 255.0
        video    = torch.from_numpy(video_np).permute(3,0,1,2)  # (C,T,H,W)
        return video, torch.tensor(label, dtype=torch.float32)

# 4. Transfer model
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)        # (B,1)
        return torch.sigmoid(logits).view(-1)  # (B,)

# 5. Metrics
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            probs      = model(vids)
            loss       = criterion(probs, labs)
            total_loss += loss.item() * labs.size(0)
            preds      = (probs >= 0.5).int()
            all_labels.extend(labs.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    n = len(loader.dataset)
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float("nan")
    return {
        "loss":      total_loss / n,
        "auc":       auc,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1_score":  f1_score(all_labels, all_preds, zero_division=0),
    }

# 6. Training with early stopping
def train_with_earlystop(model, tr_loader, vl_loader, criterion,
                         optimizer, scheduler, max_epochs,
                         stage, device, ckpt_prefix):
    best_auc  = -np.inf
    no_improve = 0
    for epoch in range(1, max_epochs+1):
        start = time.time()
        model.train()
        for vids, labs in tr_loader:
            vids, labs = vids.to(device), labs.to(device)
            optimizer.zero_grad()
            probs = model(vids)
            loss  = criterion(probs, labs)
            loss.backward()
            optimizer.step()

        # evaluate once per epoch
        tr_met = evaluate_metrics(model, tr_loader, criterion, device)
        vl_met = evaluate_metrics(model, vl_loader, criterion, device)
        scheduler.step(vl_met["loss"])

        # log to W&B
        wandb.log({
            f"{stage}/epoch":       epoch,
            f"{stage}/train_loss":  tr_met["loss"],
            f"{stage}/val_loss":    vl_met["loss"],
            f"{stage}/train_auc":   tr_met["auc"],
            f"{stage}/val_auc":     vl_met["auc"],
        })

        elapsed = time.time() - start
        print(f"[{stage}] Epoch {epoch}/{max_epochs} — "
              f"train_loss {tr_met['loss']:.4f}, "
              f"val_loss   {vl_met['loss']:.4f}, "
              f"val_auc    {vl_met['auc']:.4f} "
              f"({elapsed:.1f}s)")

        auc = vl_met["auc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc    = auc
            no_improve  = 0
            best_path   = os.path.join(ckpt_prefix, f"best_{stage}.pth")
            os.makedirs(ckpt_prefix, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            if not np.isnan(auc):
                no_improve += 1
            if no_improve >= config.patience:
                print(f"→ Early stopping {stage} at epoch {epoch}")
                break

# 7. 5-fold CV
if __name__ == "__main__":
    # gather data
    pos = os.path.join(config.data_path, "Positive_Trimmed")
    neg = os.path.join(config.data_path, "Negative_Trimmed")
    paths, labels = [], []
    for d, lab in [(pos,1), (neg,0)]:
        for fn in os.listdir(d):
            if fn.lower().endswith((".mp4", ".webm", ".avi")):
                paths.append(os.path.join(d, fn))
                labels.append(lab)

    skf       = StratifiedKFold(n_splits=config.n_splits,
                                shuffle=True, random_state=config.random_seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    fold_results = []

    for fold, (train_idx, temp_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n=== Fold {fold} ===")
        ckpt_prefix = os.path.join(ckpt_root, f"fold{fold}")
        os.makedirs(ckpt_prefix, exist_ok=True)

        # split into train / val+test
        t_paths = [paths[i] for i in train_idx]
        t_lbls  = [labels[i] for i in train_idx]
        tp_paths = [paths[i] for i in temp_idx]
        tp_lbls  = [labels[i] for i in temp_idx]
        val_p, test_p, val_l, test_l = train_test_split(
            tp_paths, tp_lbls, test_size=0.5,
            stratify=tp_lbls, random_state=config.random_seed)

        # loaders
        tr_loader = DataLoader(VideoDataset(t_paths, t_lbls),
                               batch_size=config.batch_size, shuffle=True)
        vl_loader = DataLoader(VideoDataset(val_p, val_l),
                               batch_size=config.batch_size, shuffle=False)
        te_loader = DataLoader(VideoDataset(test_p, test_l),
                               batch_size=config.batch_size, shuffle=False)

        # build fresh model
        model = AutismDetectionModel().to(device)
        wandb.watch(model, log="all", log_freq=10)

        # Stage 1: transfer head
        opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.lr_head, weight_decay=config.weight_decay)
        sch1 = ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=5)
        train_with_earlystop(model, tr_loader, vl_loader, criterion,
                             opt1, sch1, config.epochs_head,
                             "transfer_head", device, ckpt_prefix)

        # reload best head
        model.load_state_dict(torch.load(
            os.path.join(ckpt_prefix, "best_transfer_head.pth"),
            map_location=device))
        head_val  = evaluate_metrics(model, vl_loader, criterion, device)
        head_test = evaluate_metrics(model, te_loader, criterion, device)
        print(f" Transfer head  → val AUC {head_val['auc']:.4f}, test AUC {head_test['auc']:.4f}")

        # Stage 2: fine-tune last block
        for name, p in model.backbone.named_parameters():
            p.requires_grad = ("layer4" in name) or ("fc" in name)
        opt2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.lr_finetune, weight_decay=config.weight_decay)
        sch2 = ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=5)
        train_with_earlystop(model, tr_loader, vl_loader, criterion,
                             opt2, sch2, config.epochs_finetune,
                             "transfer_finetune", device, ckpt_prefix)

        # reload best finetune
        model.load_state_dict(torch.load(
            os.path.join(ckpt_prefix, "best_transfer_finetune.pth"),
            map_location=device))
        ft_val  = evaluate_metrics(model, vl_loader, criterion, device)
        ft_test = evaluate_metrics(model, te_loader, criterion, device)
        print(f" Transfer FT    → val AUC {ft_val['auc']:.4f}, test AUC {ft_test['auc']:.4f}")

        fold_results.append({
            "fold":           fold,
            "transfer_head_val":  head_val,
            "transfer_head_test": head_test,
            "transfer_ft_val":    ft_val,
            "transfer_ft_test":   ft_test
        })

    # aggregate
    summary = {}
    phases  = ["transfer_head_val","transfer_head_test","transfer_ft_val","transfer_ft_test"]
    metrics = ["loss","auc","accuracy","precision","recall","f1_score"]
    for ph in phases:
        for m in metrics:
            vals = [fr[ph][m] for fr in fold_results]
            summary[f"{ph}_{m}_mean"] = float(np.mean(vals))
            summary[f"{ph}_{m}_std"]  = float(np.std(vals))

    # print summary
    print("\n=== CV Summary (mean ± std) ===")
    for ph in phases:
        line = f"{ph:20s}: "
        for m in ["auc","accuracy","f1_score"]:
            mean, std = summary[f"{ph}_{m}_mean"], summary[f"{ph}_{m}_std"]
            line += f"{m} {mean:.3f}±{std:.3f}  "
        print(line)

    wandb.log(summary)
    wandb.finish()
    print("\n5-fold CV complete → checkpoints saved under:", ckpt_root)
