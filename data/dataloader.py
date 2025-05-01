# data/dataloader.py  (updated)

import os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class VideoFrameTransform:
    def __init__(self, frame_transforms): self.frame_transforms = frame_transforms
    def __call__(self, video):
        # video: Tensor[3, T, H, W]
        frames = []
        for t in range(video.shape[1]):
            frame = video[:, t]
            frame = transforms.ToPILImage()(frame)
            frame = self.frame_transforms(frame)
            frames.append(frame)
        return torch.stack(frames, dim=1)

class VideoDataset(Dataset):
    def __init__(self, paths, labels, max_frames, target_size, transform=None):
        self.paths, self.labels = paths, labels
        self.max_frames, self.target_size = max_frames, target_size
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path, label = self.paths[i], self.labels[i]
        cap, frames = cv2.VideoCapture(path), []
        while len(frames) < self.max_frames:
            ret, fr = cap.read()
            if not ret: break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, self.target_size)
            frames.append(fr)
        cap.release()

        if len(frames) < self.max_frames:
            pad = frames[-1] if frames else np.zeros((*self.target_size,3),np.uint8)
            frames += [pad] * (self.max_frames - len(frames))
        else:
            frames = frames[:self.max_frames]

        vid = np.stack(frames).astype(np.float32)/255.0        # (T,H,W,3)
        vid = torch.from_numpy(vid).permute(3,0,1,2)           # (3,T,H,W)

        # channel‐wise normalize
        means = torch.tensor([0.485,0.456,0.406])[:,None,None,None]
        stds  = torch.tensor([0.229,0.224,0.225])[:,None,None,None]
        vid = (vid - means)/stds

        # per‐frame augment
        if self.transform:
            vid = self.transform(vid)

        return vid, torch.tensor(label, dtype=torch.float32)


def make_dataloaders(paths, labels,
                     max_frames, target_size,
                     batch_size, num_workers,
                     augment=True,
                     val_frac=0.10, test_frac=0.20,
                     random_seed=42):
    """
    Splits `paths, labels` into train/val/test in 70/10/20 proportions
    and returns three DataLoaders.
    """

    # 1) first split off test
    train_val_p, test_p, train_val_l, test_l = train_test_split(
        paths, labels,
        test_size=test_frac,
        stratify=labels,
        random_state=random_seed
    )
    # 2) then split train_val into train / val
    val_ratio = val_frac / (1.0 - test_frac)  # relative to train_val
    train_p, val_p, train_l, val_l = train_test_split(
        train_val_p, train_val_l,
        test_size=val_ratio,
        stratify=train_val_l,
        random_state=random_seed
    )

    # 3) build augment pipeline (train only)
    frame_aug = transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5)
    ]) if augment else None

    train_ds = VideoDataset(train_p, train_l, max_frames, target_size,
                            transform=VideoFrameTransform(frame_aug) if augment else None)
    val_ds   = VideoDataset(val_p,   val_l,   max_frames, target_size, transform=None)
    test_ds  = VideoDataset(test_p,  test_l,  max_frames, target_size, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
