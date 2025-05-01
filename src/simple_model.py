#!/usr/bin/env python
# coding: utf-8

import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score,
    recall_score, f1_score
)
import wandb

# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------
CONFIG = {
    "data_path": r"C:\Users\zhams\HWs\Autsim",
    "target_size": (112, 112),
    "max_frames": 60,
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 10,
    "n_splits": 5,
    "random_seed": 42
}

# -------------------------------------------------
# 2. Initialize W&B
# -------------------------------------------------
wandb.init(project="ASD-video-simple3dlstm", config=CONFIG)
config = wandb.config

# -------------------------------------------------
# 3. Dataset
# -------------------------------------------------
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

        # even-temporal sampling or zero padding
        if len(frames) > config.max_frames:
            indices = np.linspace(0, len(frames)-1, config.max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < config.max_frames:
            pad_frame = np.zeros((*config.target_size,3), np.uint8)
            frames += [pad_frame] * (config.max_frames - len(frames))

        # normalize + channel-wise standardization
        video_np = np.array(frames, dtype=np.float32) / 255.0
        means = np.array([0.485,0.456,0.406], dtype=np.float32)
        stds  = np.array([0.229,0.224,0.225], dtype=np.float32)
        video_np = (video_np - means) / stds

        # to tensor (C,T,H,W)
        video = torch.from_numpy(video_np).permute(3,0,1,2).float()
        return video, torch.tensor(label, dtype=torch.float32)

# -------------------------------------------------
# 4. Simple3DLSTM Model
# -------------------------------------------------
class Simple3DLSTM(nn.Module):
    def __init__(self, max_frames, target_size):
        super().__init__()
        C, T, H, W = 3, max_frames, *target_size
        self.cnn = nn.Sequential(
            nn.Conv3d(3,32,(3,3,3),padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(32),
            nn.Conv3d(32,64,(3,3,3),padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(64),
            nn.Conv3d(64,128,(3,3,3),padding=1),nn.ReLU(), nn.MaxPool3d((1,2,2)), nn.BatchNorm3d(128),
        )
        spatial_h, spatial_w = H//8, W//8
        feat_dim = 128 * spatial_h * spatial_w
        self.lstm = nn.LSTM(feat_dim, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32,1),  nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        f = self.cnn(x)                  # (B,128,T,H/8,W/8)
        f = f.permute(0,2,1,3,4)         # (B,T,128,H/8,W/8)
        f = f.reshape(B, f.size(1), -1)  # (B,T,feat_dim)
        out,_ = self.lstm(f)             # (B,T,64)
        last = out[:,-1,:]               # (B,64)
        return self.head(last).view(-1)  # (B,)

# -------------------------------------------------
# 5. Metrics + ROC logging
# -------------------------------------------------
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            probs = model(vids)
            total_loss += criterion(probs, labs).item() * labs.size(0)
            all_labels.extend(labs.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    n = len(loader.dataset)
    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds = (probs_arr >= 0.5).astype(int)

    if len(np.unique(labels_arr)) > 1:
        auc = roc_auc_score(labels_arr, probs_arr)
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
    else:
        auc, fpr, tpr = float("nan"), [0,1], [0,1]

    return {
        "loss": total_loss / n,
        "auc": auc,
        "fpr": fpr, "tpr": tpr,
        "labels": labels_arr, "probs": probs_arr,
        "accuracy": accuracy_score(labels_arr, preds),
        "precision": precision_score(labels_arr, preds, zero_division=0),
        "recall": recall_score(labels_arr, preds, zero_division=0),
        "f1_score": f1_score(labels_arr, preds, zero_division=0),
    }

# -------------------------------------------------
# 6. Training loop with early stopping
# -------------------------------------------------
def train_with_earlystop(model, tr_loader, vl_loader, criterion,
                         optimizer, scheduler, max_epochs,
                         device, fold):
    best_auc = -np.inf
    no_imp = 0
    for epoch in range(1, max_epochs+1):
        start = time.time()
        model.train()
        for vids, labs in tr_loader:
            vids, labs = vids.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(vids), labs)
            loss.backward()
            optimizer.step()

        trm = evaluate_metrics(model, tr_loader, criterion, device)
        vlm = evaluate_metrics(model, vl_loader, criterion, device)
        scheduler.step(vlm["loss"])

        # build 2-column prob array for W&B ROC
        if len(vlm["labels"]) > 1:
            probas = np.stack([1-vlm["probs"], vlm["probs"]], axis=1)
        else:
            # dummy
            probas = np.zeros((len(vlm["labels"]),2))

        wandb.log({
            f"simple/fold{fold}/epoch":      epoch,
            f"simple/fold{fold}/train_loss": trm["loss"],
            f"simple/fold{fold}/val_loss":   vlm["loss"],
            f"simple/fold{fold}/train_auc":  trm["auc"],
            f"simple/fold{fold}/val_auc":    vlm["auc"],
            f"simple/fold{fold}/val_roc":    wandb.plot.roc_curve(
                                                vlm["labels"],
                                                probas
                                            )
        })

        elapsed = time.time() - start
        print(f"[simple][fold {fold}] Epoch {epoch}/{max_epochs} — "
              f"train_loss {trm['loss']:.4f}, val_auc {vlm['auc']:.4f} ({elapsed:.1f}s)")

        if not np.isnan(vlm["auc"]) and vlm["auc"] > best_auc:
            best_auc, no_imp = vlm["auc"], 0
            torch.save(model.state_dict(),
                       f"best_simple_fold{fold}.pth")
        else:
            no_imp += 1
            if no_imp >= config.patience:
                print(f"→ Early stopping simple at epoch {epoch}")
                break

# -------------------------------------------------
# 7. 5-Fold CV main
# -------------------------------------------------
if __name__ == "__main__":
    pos_dir = os.path.join(config.data_path, "Positive_Trimmed")
    neg_dir = os.path.join(config.data_path, "Negative_Trimmed")
    paths, labels = [], []
    for d,l in [(pos_dir,1),(neg_dir,0)]:
        for fn in os.listdir(d):
            if fn.lower().endswith((".mp4",".webm",".avi")):
                paths.append(os.path.join(d,fn))
                labels.append(l)

    skf     = StratifiedKFold(n_splits=config.n_splits,
                              shuffle=True, random_state=config.random_seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    fold_results = []

    for fold, (train_idx, temp_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n=== Fold {fold} ===")
        t_paths  = [paths[i] for i in train_idx]
        t_lbls   = [labels[i] for i in train_idx]
        tmp_p    = [paths[i] for i in temp_idx]
        tmp_l    = [labels[i] for i in temp_idx]
        val_p, test_p, val_l, test_l = train_test_split(
            tmp_p, tmp_l, test_size=0.5,
            stratify=tmp_l, random_state=config.random_seed
        )

        tr_loader = DataLoader(VideoDataset(t_paths, t_lbls),
                               batch_size=config.batch_size, shuffle=True)
        vl_loader = DataLoader(VideoDataset(val_p, val_l),
                               batch_size=config.batch_size, shuffle=False)
        te_loader = DataLoader(VideoDataset(test_p, test_l),
                               batch_size=config.batch_size, shuffle=False)

        model = Simple3DLSTM(config.max_frames, config.target_size).to(device)
        wandb.watch(model, log="all", log_freq=10)

        opt = optim.AdamW(model.parameters(),
                          lr=config.lr,
                          weight_decay=config.weight_decay)
        sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

        train_with_earlystop(model, tr_loader, vl_loader,
                             criterion, opt, sch,
                             config.epochs, device, fold)

        # load best checkpoint
        model.load_state_dict(torch.load(f"best_simple_fold{fold}.pth",
                                         map_location=device))
        val_met  = evaluate_metrics(model, vl_loader, criterion, device)
        test_met = evaluate_metrics(model, te_loader, criterion, device)

        print(f" Simple → val AUC {val_met['auc']:.4f}, test AUC {test_met['auc']:.4f}")
        fold_results.append({"val": val_met, "test": test_met})

    # aggregate
    import numpy as _np
    summary = {}
    for phase in ["val","test"]:
        for m in ["loss","auc","accuracy","precision","recall","f1_score"]:
            vals = [fr[phase][m] for fr in fold_results]
            summary[f"{phase}_{m}_mean"] = float(_np.mean(vals))
            summary[f"{phase}_{m}_std"]  = float(_np.std(vals))

    print("\n=== CV Summary (mean ± std) ===")
    for phase in ["val","test"]:
        line = f"{phase:6s}: "
        for m in ["auc","accuracy","f1_score"]:
            mn = summary[f"{phase}_{m}_mean"]
            sd = summary[f"{phase}_{m}_std"]
            line += f"{m} {mn:.3f}±{sd:.3f}  "
        print(line)

    wandb.log(summary)
    wandb.finish()
    print("\n5-fold CV complete for Simple3DLSTM")
