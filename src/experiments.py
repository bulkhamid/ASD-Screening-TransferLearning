#!/usr/bin/env python
# coding: utf-8

import os, cv2, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
import wandb

# 1. Configuration
CONFIG = {
    "data_path": r"C:\Users\zhams\HWs\Autsim",
    "target_size": (112, 112),
    "max_frames": 60,
    "batch_size": 8,
    "epochs_head": 100,
    "epochs_finetune": 100,
    "lr_head": 1e-3,
    "lr_finetune": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10,
    "n_splits": 5,
    "random_seed": 42
}

# 2. Initialize W&B
wandb.init(project="ASD-video-screening", config=CONFIG)
config = wandb.config

# 3. Dataset
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(path)
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
            pad = frames[-1] if frames else np.zeros((*config.target_size, 3), np.uint8)
            frames += [pad] * (config.max_frames - len(frames))
        video_np = np.array(frames, dtype=np.float32) / 255.0
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # (C, T, H, W)
        return video, torch.tensor(label, dtype=torch.float32)

# 4a. Transfer model (3D-ResNet)
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)       # (B,1)
        return torch.sigmoid(logits).view(-1)

# 4b. Simple 3D-CNN + LSTM
class Simple3DLSTM(nn.Module):
    def __init__(self, max_frames, target_size):
        super().__init__()
        C, T, H, W = 3, max_frames, *target_size
        self.cnn = nn.Sequential(
            nn.Conv3d(3,32,(3,3,3),padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(32),
            nn.Conv3d(32,64,(3,3,3),padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(64),
            nn.Conv3d(64,128,(3,3,3),padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(128),
        )
        spatial_h, spatial_w = H//8, W//8
        feat_dim = 128 * spatial_h * spatial_w
        self.lstm = nn.LSTM(feat_dim, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32,1), nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        f = self.cnn(x)                  # (B,128,T,H/8,W/8)
        f = f.permute(0,2,1,3,4)         # (B,T,128,H/8,W/8)
        f = f.reshape(B, f.size(1), -1)  # (B,T,feat_dim)
        out,_ = self.lstm(f)             # (B,T,64)
        last = out[:,-1,:]               # (B,64)
        return self.head(last).view(-1)  # (B,)

# 5. Metrics
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs, all_preds = [],[],[]
    total_loss = 0.0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            probs = model(vids)
            loss = criterion(probs, labs)
            total_loss += loss.item() * labs.size(0)
            preds = (probs>=0.5).int()
            all_labels.extend(labs.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    n = len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels))>1 else float("nan")
    return {
        "loss": total_loss/n,
        "auc": auc,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
    }

# 6. Training w/ early stopping
def train_with_earlystop(model, tr_loader, vl_loader, criterion,
                         optimizer, scheduler, max_epochs, stage, device):
    best_auc, no_imp = -np.inf, 0
    for epoch in range(1, max_epochs+1):
        start = time.time()
        model.train()
        for vids,labs in tr_loader:
            vids, labs = vids.to(device), labs.to(device)
            optimizer.zero_grad()
            probs = model(vids)
            loss = criterion(probs, labs)
            loss.backward()
            optimizer.step()

        tr_met = evaluate_metrics(model, tr_loader, criterion, device)
        vl_met = evaluate_metrics(model, vl_loader, criterion, device)
        scheduler.step(vl_met["loss"])

        wandb.log({
            f"{stage}/epoch":      epoch,
            f"{stage}/train_loss": tr_met["loss"],
            f"{stage}/val_loss":   vl_met["loss"],
            f"{stage}/train_auc":  tr_met["auc"],
            f"{stage}/val_auc":    vl_met["auc"],
        })

        elapsed = time.time() - start
        print(f"[{stage}] Epoch {epoch}/{max_epochs} — "
              f"train_loss {tr_met['loss']:.4f}, val_loss {vl_met['loss']:.4f}, "
              f"val_auc {vl_met['auc']:.4f} ({elapsed:.1f}s)")

        auc = vl_met["auc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc, no_imp = auc, 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("checkpoints", f"best_{stage}.pth"))
        else:
            if not np.isnan(auc):
                no_imp += 1
            if no_imp >= config.patience:
                print(f"→ Early stopping {stage} at epoch {epoch}")
                break

# 7. 5-fold CV
if __name__=="__main__":
    # gather all video paths + labels
    pos_dir = os.path.join(config.data_path, "Positive_Trimmed")
    neg_dir = os.path.join(config.data_path, "Negative_Trimmed")
    paths, labels = [], []
    for d, lab in [(pos_dir,1),(neg_dir,0)]:
        for fn in os.listdir(d):
            if fn.lower().endswith((".mp4",".webm",".avi")):
                paths.append(os.path.join(d, fn))
                labels.append(lab)

    skf = StratifiedKFold(n_splits=config.n_splits,
                          shuffle=True, random_state=config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    fold_results = []

    for fold, (train_idx, temp_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n=== Fold {fold} ===")
        # split temp into val/test
        t_paths  = [paths[i] for i in train_idx]
        t_lbls   = [labels[i] for i in train_idx]
        tmp_paths= [paths[i] for i in temp_idx]
        tmp_lbls = [labels[i] for i in temp_idx]
        val_p, test_p, val_l, test_l = train_test_split(
            tmp_paths, tmp_lbls, test_size=0.5,
            stratify=tmp_lbls, random_state=config.random_seed
        )

        # dataloaders
        tr_loader = DataLoader(VideoDataset(t_paths, t_lbls),
                               batch_size=config.batch_size, shuffle=True)
        vl_loader = DataLoader(VideoDataset(val_p, val_l),
                               batch_size=config.batch_size, shuffle=False)
        te_loader = DataLoader(VideoDataset(test_p, test_l),
                               batch_size=config.batch_size, shuffle=False)

        # --- Transfer model pipeline ---
        model = AutismDetectionModel().to(device)
        wandb.watch(model, log="all", log_freq=10)

        # Stage 1: head-only
        opt1 = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                           lr=config.lr_head, weight_decay=config.weight_decay)
        sch1 = ReduceLROnPlateau(opt1, mode="min", factor=0.5, patience=config.patience)
        train_with_earlystop(model, tr_loader, vl_loader,
                             criterion, opt1, sch1,
                             config.epochs_head, "transfer_head", device)
        model.load_state_dict(torch.load("best_transfer_head.pth", map_location=device))
        th_val = evaluate_metrics(model, vl_loader, criterion, device)
        th_test= evaluate_metrics(model, te_loader, criterion, device)
        print(f" Transfer head  → val AUC {th_val['auc']:.4f}, test AUC {th_test['auc']:.4f}")

        # Stage 2: fine-tune last block
        for name,p in model.backbone.named_parameters():
            p.requires_grad = ("layer4" in name) or ("fc" in name)
        opt2 = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                           lr=config.lr_finetune, weight_decay=config.weight_decay)
        sch2 = ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=config.patience)
        train_with_earlystop(model, tr_loader, vl_loader,
                             criterion, opt2, sch2,
                             config.epochs_finetune, "transfer_ft", device)
        model.load_state_dict(torch.load("best_transfer_ft.pth", map_location=device))
        tf_val = evaluate_metrics(model, vl_loader, criterion, device)
        tf_test= evaluate_metrics(model, te_loader, criterion, device)
        print(f" Transfer FT    → val AUC {tf_val['auc']:.4f}, test AUC {tf_test['auc']:.4f}")

        # --- Simple3DLSTM pipeline ---
        simple = Simple3DLSTM(config.max_frames, config.target_size).to(device)
        wandb.watch(simple, log="all", log_freq=10)
        opt_s = optim.AdamW(simple.parameters(),
                            lr=config.lr_finetune, weight_decay=config.weight_decay)
        sch_s = ReduceLROnPlateau(opt_s, mode="min", factor=0.5, patience=config.patience)
        train_with_earlystop(simple, tr_loader, vl_loader,
                             criterion, opt_s, sch_s,
                             config.epochs_finetune, "simple", device)
        simple.load_state_dict(torch.load("best_simple.pth", map_location=device))
        sm_val = evaluate_metrics(simple, vl_loader, criterion, device)
        sm_test= evaluate_metrics(simple, te_loader, criterion, device)
        print(f" Simple3DLSTM  → val AUC {sm_val['auc']:.4f}, test AUC {sm_test['auc']:.4f}")

        # record
        fold_results.append({
            "fold": fold,
            "transfer_head_val": th_val,   "transfer_head_test": th_test,
            "transfer_ft_val":   tf_val,   "transfer_ft_test":   tf_test,
            "simple_val":        sm_val,   "simple_test":        sm_test
        })

    # aggregate
    summary = {}
    phases = [
        "transfer_head_val","transfer_head_test",
        "transfer_ft_val",  "transfer_ft_test",
        "simple_val",       "simple_test"
    ]
    metrics_list = ["loss","auc","accuracy","precision","recall","f1_score"]
    for ph in phases:
        for m in metrics_list:
            vals = [fr[ph][m] for fr in fold_results]
            summary[f"{ph}_{m}_mean"] = float(np.mean(vals))
            summary[f"{ph}_{m}_std"]  = float(np.std(vals))

    # print overall
    print("\n=== CV Summary (mean ± std) ===")
    for ph in phases:
        line = f"{ph:20s}: "
        for m in ["auc","accuracy","f1_score"]:
            mean,std = summary[f"{ph}_{m}_mean"], summary[f"{ph}_{m}_std"]
            line += f"{m} {mean:.3f}±{std:.3f}  "
        print(line)

    wandb.log(summary)
    wandb.finish()
    print("\n5-fold CV complete")
