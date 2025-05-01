# data/dataloader.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VideoFrameTransform:
    """
    Apply a torchvision.transforms pipeline _to each frame_ of a video Tensor.
    Expects video shape: (C, T, H, W)
    """
    def __init__(self, frame_transforms):
        self.frame_transforms = frame_transforms

    def __call__(self, video):
        # video: torch.Tensor[3, T, H, W]
        frames = []
        for t in range(video.shape[1]):
            # grab frame (C,H,W) → PIL → augment → back to Tensor
            frame = video[:, t]                          # (3,H,W)
            frame = transforms.ToPILImage()(frame)       # PIL
            frame = self.frame_transforms(frame)         # PIL→Tensor
            frames.append(frame)
        # stack back to (3,T,H,W)
        return torch.stack(frames, dim=1)


class VideoDataset(Dataset):
    def __init__(self, paths, labels, max_frames, target_size, transform=None):
        self.paths       = paths
        self.labels      = labels
        self.max_frames  = max_frames
        self.target_size = target_size
        self.transform   = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.max_frames:
            ret, fr = cap.read()
            if not ret: break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, self.target_size)
            frames.append(fr)
        cap.release()

        # pad / truncate
        if len(frames) < self.max_frames:
            pad = frames[-1] if frames else np.zeros((*self.target_size,3),np.uint8)
            frames += [pad] * (self.max_frames - len(frames))
        else:
            frames = frames[:self.max_frames]

        vid = np.stack(frames).astype(np.float32) / 255.0  # (T,H,W,3)
        vid = torch.from_numpy(vid).permute(3,0,1,2)       # (3,T,H,W)

        # normalize
        means = torch.tensor([0.485,0.456,0.406])[:,None,None,None]
        stds  = torch.tensor([0.229,0.224,0.225])[:,None,None,None]
        vid = (vid - means) / stds

        # per-frame augment
        if self.transform:
            vid = self.transform(vid)

        return vid, torch.tensor(label, dtype=torch.float32)


def make_dataloaders(train_paths, train_labels,
                     val_paths,   val_labels,
                     max_frames,  target_size,
                     batch_size,  num_workers,
                     augment=True):
    # build augmentation pipeline
    frame_augs = transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5)
    ])
    video_transform = VideoFrameTransform(frame_augs) if augment else None

    train_ds = VideoDataset(train_paths, train_labels,
                            max_frames, target_size,
                            transform=video_transform)
    val_ds   = VideoDataset(val_paths, val_labels,
                            max_frames, target_size,
                            transform=None)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader
