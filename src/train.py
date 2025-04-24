import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
import wandb

# -----------------------------
# 1. Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="ASD Video Screening Training Script")
parser.add_argument("--data_path", type=str, required=True,
                    help="Base path containing 'Positive_Trimmed' & 'Negative_Trimmed'")
parser.add_argument("--model_type", type=str, choices=["pretrained", "simple"], default="pretrained",
                    help="Which model to use: pretrained R3D-18 or simple 3D CNN")
parser.add_argument("--target_size", type=int, nargs=2, default=(112,112),
                    help="Frame resize dimensions H W")
parser.add_argument("--max_frames", type=int, default=60, help="Number of frames per video")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# -----------------------------
# 2. WandB Initialization
# -----------------------------
wandb.init(project="ASD-video-screening", config=vars(args))
config = wandb.config

# -----------------------------
# 3. Dataset Definition
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, tuple(config.target_size))
            frames.append(frame)
        cap.release()
        if len(frames) < config.max_frames:
            pad = frames[-1] if frames else np.zeros((*config.target_size,3), np.uint8)
            frames += [pad] * (config.max_frames - len(frames))
        video = np.array(frames, dtype=np.float32) / 255.0
        video = torch.from_numpy(video).permute(3,0,1,2)
        return video, torch.tensor(label, dtype=torch.float32)

# -----------------------------
# 4. Model Definitions
# -----------------------------
class AutismDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.video.r3d_18(pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone
    def forward(self, x):
        out = self.backbone(x)
        return torch.sigmoid(out).view(-1)

class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(64,1)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return torch.sigmoid(self.fc(x)).view(-1)

# -----------------------------
# 5. Metrics and Training Utils
# -----------------------------
def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs, all_preds = [],[],[]
    total_loss = 0.0
    with torch.no_grad():
        for vids, labs in loader:
            vids, labs = vids.to(device), labs.to(device)
            probs = model(vids)
            loss = criterion(probs, labs)
            total_loss += loss.item()*labs.size(0)
            preds = (probs>=0.5).int()
            all_labels.extend(labs.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    n = len(loader.dataset)
    return {
        "loss": total_loss/n,
        "auc": roc_auc_score(all_labels, all_probs),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0)
    }

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for vids, labs in loader:
        vids, labs = vids.to(device), labs.to(device)
        optimizer.zero_grad()
        probs = model(vids)
        loss = criterion(probs, labs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*labs.size(0)
    return running_loss/len(loader.dataset)

# -----------------------------
# 6. Data Split (70/10/20)
# -----------------------------
paths, labels = [], []
for d, lab in [(os.path.join(args.data_path,"Positive_Trimmed"),1),
               (os.path.join(args.data_path,"Negative_Trimmed"),0)]:
    for fn in os.listdir(d):
        if fn.lower().endswith(('.mp4','.avi','.webm')):
            paths.append(os.path.join(d,fn))
            labels.append(lab)

# 70% train, 30% holdout
train_p, hold_p, train_l, hold_l = train_test_split(
    paths, labels, train_size=0.7, stratify=labels, random_state=args.seed
)
# split holdout into 10% val (1/3 of 30%) and 20% test (2/3 of 30%)
val_p, test_p, val_l, test_l = train_test_split(
    hold_p, hold_l, test_size=2/3, stratify=hold_l, random_state=args.seed
)

train_loader = DataLoader(VideoDataset(train_p,train_l),
                          batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(VideoDataset(val_p,val_l),
                          batch_size=config.batch_size)
test_loader  = DataLoader(VideoDataset(test_p,test_l),
                          batch_size=config.batch_size)

# -----------------------------
# 7. Instantiate Model
# -----------------------------
device = torch.device(config.device)
model = (AutismDetectionModel() if config.model_type=="pretrained"
         else Simple3DCNN()).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr,
                       weight_decay=config.weight_decay)
wandb.watch(model, log="all", log_freq=10)

# -----------------------------
# 8. Training Loop
# -----------------------------
for epoch in range(config.epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_metrics = evaluate_metrics(model, train_loader, criterion, device)
    val_metrics   = evaluate_metrics(model, val_loader,   criterion, device)
    wandb.log({"train_loss": train_loss, **{f"train_{k}":v for k,v in train_metrics.items()},
               **{f"val_{k}":v for k,v in val_metrics.items()}})
    print(f"Epoch {epoch+1}/{config.epochs} — "
          f"Train loss {train_loss:.4f}, Val AUC {val_metrics['auc']:.4f}")

# -----------------------------
# 9. Final Test Evaluation
# -----------------------------
test_metrics = evaluate_metrics(model, test_loader, criterion, device)
wandb.log({f"test_{k}":v for k,v in test_metrics.items()})
print("Test metrics:", test_metrics)

# -----------------------------
# 10. Save Model
# -----------------------------
out_name = f"{config.model_type}_model.pth"
torch.save(model.state_dict(), out_name)
wandb.save(out_name)
wandb.finish()
print(f"Saved model as {out_name}")
