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
from sklearn.metrics import roc_auc_score
import wandb

# 1. Configuration
CONFIG = {
    "data_path": "C:\\Users\\zhams\\HWs\\Autsim",             # Base path containing 'Positive_Trimmed' & 'Negative_Trimmed'
    "target_size": (112, 112),
    "max_frames": 60,
    "batch_size": 8,
    "epochs_head": 10,
    "epochs_finetune": 10,
    "lr_head": 1e-3,
    "lr_finetune": 1e-4,
    "weight_decay": 1e-5,
    "test_size": 0.4,              # 60% train, 40% temp -> split temp into 20% val / 20% test
    "random_seed": 42
}

# 2. Initialize Weights & Biases
wandb.init(project="ASD-video-screening", config=CONFIG)
config = wandb.config

# 3. Video Dataset Definition
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
        
        # Pad if fewer frames
        if len(frames) < config.max_frames:
            pad_frame = frames[-1] if frames else np.zeros((*config.target_size, 3), np.uint8)
            frames += [pad_frame] * (config.max_frames - len(frames))
        
        video_np = np.array(frames, dtype=np.float32) / 255.0
        video = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        if self.transform:
            video = self.transform(video)
        
        return video, torch.tensor(label, dtype=torch.float32)

# 4. Model Definition: Pretrained 3D ResNet
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(pretrained=True)
        # Freeze all layers initially
        for param in backbone.parameters():
            param.requires_grad = False
        # Replace final layer
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return torch.sigmoid(self.backbone(x).squeeze())

# 5. Utility: train one epoch
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

# 6. Utility: evaluate
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            preds = model(videos).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    return auc

# 7. Main Routine
if __name__ == "__main__":
    # Gather video paths and labels
    pos_dir = os.path.join(config.data_path, 'Positive_Trimmed')
    neg_dir = os.path.join(config.data_path, 'Negative_Trimmed')
    video_paths, labels = [], []
    for p in os.listdir(pos_dir):
        if p.endswith(('.mp4','.webm','.avi')):
            video_paths.append(os.path.join(pos_dir, p)); labels.append(1)
    for n in os.listdir(neg_dir):
        if n.endswith(('.mp4','.webm','.avi')):
            video_paths.append(os.path.join(neg_dir, n)); labels.append(0)

    # Split 60/20/20
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels,
        test_size=config.test_size,
        stratify=labels,
        random_state=config.random_seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=config.random_seed
    )

    # Data transforms
    frame_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    video_transform = lambda vid: vid  # no-op; augment inside dataset if desired

    # DataLoaders
    train_ds = VideoDataset(train_paths, train_labels, transform=video_transform)
    val_ds   = VideoDataset(val_paths,   val_labels,   transform=None)
    test_ds  = VideoDataset(test_paths,  test_labels,  transform=None)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Model, criterion, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutismDetectionModel().to(device)
    criterion = nn.BCELoss()

    # Stage 1: Train head only
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.lr_head, weight_decay=config.weight_decay)
    wandb.watch(model, log="all", log_freq=10)
    for epoch in range(config.epochs_head):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = evaluate(model, val_loader, device)
        wandb.log({"stage": "head", "epoch": epoch+1,
                   "train_loss": train_loss, "val_auc": val_auc})
        print(f"[Head] Epoch {epoch+1}/{config.epochs_head} — Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Stage 2: Unfreeze last block & fine-tune
    for name, param in model.backbone.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.lr_finetune, weight_decay=config.weight_decay)
    for epoch in range(config.epochs_finetune):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = evaluate(model, val_loader, device)
        wandb.log({"stage": "finetune", "epoch": epoch+1,
                   "train_loss": train_loss, "val_auc": val_auc})
        print(f"[Finetune] Epoch {epoch+1}/{config.epochs_finetune} — Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Final evaluation on test
    test_auc = evaluate(model, test_loader, device)
    print(f"Test AUC: {test_auc:.4f}")
    wandb.log({"test_auc": test_auc})

    # Save model
    torch.save(model.state_dict(), "best_model.pth")
    wandb.save("best_model.pth")

