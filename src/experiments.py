#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
import wandb

# 1. Configuration
CONFIG = {
    "data_path": "C:\\Users\\zhams\\HWs\\Autsim",  # Base path with 'Positive_Trimmed' & 'Negative_Trimmed'
    "target_size": (112, 112),
    "max_frames": 60,
    "batch_size": 8,
    "epochs_head": 10,
    "epochs_finetune": 10,
    "lr_head": 1e-3,
    "lr_finetune": 1e-4,
    "weight_decay": 1e-5,
    "val_size": 0.1,    # 10% of total for validation
    "test_size": 0.2,   # 20% of total for testing
    "random_seed": 42
}

# 2. Initialize W&B
wandb.init(project="ASD-video-screening", config=CONFIG)
config = wandb.config

# 3. Dataset
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

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
        if self.transform:
            video = self.transform(video)
        return video, torch.tensor(label, dtype=torch.float32)

# 4. Model
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x):
        return torch.sigmoid(self.backbone(x).squeeze())

# 5. Metrics helper
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * videos.size(0)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {
        "loss": avg_loss,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# 6. Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * videos.size(0)
    return running_loss / len(loader.dataset)

# 7. Main
if __name__ == "__main__":
    # Gather files
    pos_dir = os.path.join(config.data_path, "Positive_Trimmed")
    neg_dir = os.path.join(config.data_path, "Negative_Trimmed")
    paths, labels = [], []
    for d, lab in [(pos_dir,1),(neg_dir,0)]:
        for fn in os.listdir(d):
            if fn.lower().endswith(('.mp4','.webm','.avi')):
                paths.append(os.path.join(d,fn))
                labels.append(lab)

    # 70/10/20 split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(config.val_size+config.test_size),
        stratify=labels,
        random_state=config.random_seed
    )
    # From temp 30%, allocate 10% val (i.e. 1/3 of temp) and 20% test (2/3 of temp)
    val_ratio = config.val_size / (config.val_size + config.test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=config.random_seed
    )

    # DataLoaders
    train_ds = VideoDataset(train_paths, train_labels, transform=None)
    val_ds   = VideoDataset(val_paths,   val_labels,   transform=None)
    test_ds  = VideoDataset(test_paths,  test_labels,  transform=None)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutismDetectionModel().to(device)
    criterion = nn.BCELoss()

    wandb.watch(model, log="all", log_freq=10)

    # Stage 1: head only
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr_head, weight_decay=config.weight_decay
    )
    for epoch in range(config.epochs_head):
        train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate_metrics(model, train_loader, criterion, device)
        val_metrics   = evaluate_metrics(model, val_loader,   criterion, device)
        wandb.log({f"head/train_{k}":v for k,v in train_metrics.items()})
        wandb.log({f"head/val_{k}":v   for k,v in val_metrics.items()})
        print(f"[Head] Epoch {epoch+1}: Train Loss {train_metrics['loss']:.4f}, Val AUC {val_metrics['auc']:.4f}")

    # Stage 2: unfreeze last block
    for name, param in model.backbone.named_parameters():
        param.requires_grad = ("layer4" in name) or ("fc" in name)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr_finetune, weight_decay=config.weight_decay
    )
    for epoch in range(config.epochs_finetune):
        train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate_metrics(model, train_loader, criterion, device)
        val_metrics   = evaluate_metrics(model, val_loader,   criterion, device)
        wandb.log({f"finetune/train_{k}":v for k,v in train_metrics.items()})
        wandb.log({f"finetune/val_{k}":v   for k,v in val_metrics.items()})
        print(f"[Finetune] Epoch {epoch+1}: Train Loss {train_metrics['loss']:.4f}, Val AUC {val_metrics['auc']:.4f}")

    # Final test eval
    test_metrics = evaluate_metrics(model, test_loader, criterion, device)
    wandb.log({f"test_{k}":v for k,v in test_metrics.items()})
    print("Test metrics:", test_metrics)

    # Save
    torch.save(model.state_dict(), "best_model.pth")
    wandb.save("best_model.pth")

    wandb.finish()
    print("Training complete. Model saved and W&B run finished.")