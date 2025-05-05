import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random

# ───────────────────────────── Frame-level augment ────────────────────────────
def build_frame_aug(target_size):
    """Random spatial/colour aug + correct normalisation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])

class VideoFrameTransform:
    """Applies frame_aug to every frame in a 4-D video tensor (C, T, H, W)."""
    def __init__(self, frame_aug): self.frame_aug = frame_aug
    def __call__(self, video):
        frames = []
        for t in range(video.shape[1]):
            frame = transforms.ToPILImage()(video[:, t])
            frames.append(self.frame_aug(frame))
        return torch.stack(frames, dim=1)

# ───────────────────────────── Video dataset ──────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, paths, labels, max_frames, target_size, transform=None):
        self.paths, self.labels = paths, labels
        self.max_frames, self.target_size = max_frames, target_size
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path, label = self.paths[i], self.labels[i]
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Randomly pick frame indices
        if total_frames <= self.max_frames:
            frame_idxs = list(range(total_frames))
        else:
            frame_idxs = sorted(random.sample(range(total_frames), self.max_frames))

        frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if not ret:
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, self.target_size)
            frames.append(fr)
        cap.release()

        # Pad if needed
        if len(frames) < self.max_frames:
            pad = np.zeros((*self.target_size, 3), np.uint8)
            frames += [pad] * (self.max_frames - len(frames))

        vid = np.stack(frames).astype(np.float32) / 255.0
        vid = torch.from_numpy(vid).permute(3, 0, 1, 2)  # (C, T, H, W)

        if self.transform:
            vid = self.transform(vid)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
            vid = (vid - mean) / std

        return vid, torch.tensor(label, dtype=torch.float32)

# ───────────────────────────── Factory ────────────────────────────────────────
def make_dataloaders(paths, labels, max_frames, target_size, batch_size, num_workers,
                     augment=True, val_frac=0.10, test_frac=0.20, random_seed=42):
    """70/10/20 hold-out split ➜ three DataLoaders."""
    tr_val_p, test_p, tr_val_l, test_l = train_test_split(
        paths, labels, test_size=test_frac, stratify=labels, random_state=random_seed)

    val_ratio = val_frac / (1. - test_frac)
    train_p, val_p, train_l, val_l = train_test_split(
        tr_val_p, tr_val_l, test_size=val_ratio, stratify=tr_val_l, random_state=random_seed)

    frame_aug = build_frame_aug(target_size) if augment else None

    tr_ds = VideoDataset(train_p, train_l, max_frames, target_size,
                         transform=VideoFrameTransform(frame_aug) if augment else None)
    vl_ds = VideoDataset(val_p, val_l, max_frames, target_size)
    te_ds = VideoDataset(test_p, test_l, max_frames, target_size)

    return (
        DataLoader(tr_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(vl_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(te_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )