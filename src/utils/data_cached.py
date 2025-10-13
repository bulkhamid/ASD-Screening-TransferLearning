"""
Cached data loading utilities for pre-processed video tensors.

This module provides efficient data loading from pre-processed .npz files
with optional augmentation for training. Videos are stored as memory-mapped
arrays and loaded on-the-fly to minimize memory usage.
"""
import numpy as np
import torch
import random
import functools
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_split(split):
    """
    Load a pre-processed split from .npz file.
    
    The .npz files are created by data/save_numpy_dataset.py with shape
    (N, T, H, W, C) in uint8 format. This function loads them as memory-mapped
    arrays and converts to PyTorch tensors with shape (N, C, T, H, W) in float32.
    
    Args:
        split: Split name ('train', 'val', or 'test')
    
    Returns:
        TensorDataset containing video tensors and labels
    """
    npz = np.load(f"data/{split}.npz", mmap_mode="r")
    # Convert from (N, T, H, W, C) to (N, C, T, H, W) and normalize to [0, 1]
    vids = torch.from_numpy(npz["X"]).permute(0, 4, 1, 2, 3).float() / 255.0
    labels = torch.from_numpy(npz["y"]).float()
    return TensorDataset(vids, labels)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_frame_aug(target_size=(224, 224)):
    """
    Build frame-level augmentation pipeline for training.
    
    Applies random spatial and color augmentations to individual frames
    including random cropping, color jittering, horizontal flipping,
    rotation, and random erasing. All frames are normalized using
    ImageNet statistics.
    
    Args:
        target_size: Target size (height, width) for frames
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # Convert to (C, H, W) in [0, 1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5)
    ])


def _normalize_tensor(video_5d):
    """
    Apply ImageNet normalization to video tensor.
    
    Args:
        video_5d: Video tensor of shape (C, T, H, W) with values in [0, 1]
    
    Returns:
        Normalized tensor with same shape
    """
    m = torch.tensor(IMAGENET_MEAN, device=video_5d.device)[:, None, None, None]
    s = torch.tensor(IMAGENET_STD, device=video_5d.device)[:, None, None, None]
    return (video_5d - m) / s


# ---------------------------------------------------------------------------
# DataLoader Builders
# ---------------------------------------------------------------------------
def _train_collate(batch, frame_aug=None, aug_p=0.5):
    """
    Custom collate function for training data with augmentation.
    
    Args:
        batch: List of (video, label) tuples
        frame_aug: Frame augmentation pipeline (optional)
        aug_p: Probability of applying augmentation
    
    Returns:
        Tuple of (batched videos, batched labels)
    """
    vids, lbls = zip(*batch)
    out = []
    to_pil = ToPILImage()
    
    for v in vids:
        if frame_aug is not None and random.random() < aug_p:
            # Apply augmentation to each frame
            frs = [frame_aug(to_pil(v[:, t])) for t in range(v.shape[1])]
            v = torch.stack(frs, dim=1)
        else:
            v = _normalize_tensor(v)
        out.append(v)
    
    return torch.stack(out), torch.stack(lbls)


def make_train_loader(batch_size,
                      aug_p=0.5,
                      target_size=(224, 224),
                      shuffle=True,
                      num_workers=4):
    """
    Create a training data loader with augmentation.
    
    Args:
        batch_size: Batch size
        aug_p: Probability of applying augmentation per sample (default: 0.5)
        target_size: Target frame size (height, width) (default: (224, 224))
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of data loading workers (default: 4)
    
    Returns:
        DataLoader for training set
    """
    ds = load_split("train")
    frame_aug = build_frame_aug(target_size)
    collate = functools.partial(
        _train_collate,
        frame_aug=frame_aug,
        aug_p=aug_p
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True
    )


def _eval_collate(batch):
    """
    Custom collate function for evaluation data (no augmentation).
    
    Args:
        batch: List of (video, label) tuples
    
    Returns:
        Tuple of (batched videos, batched labels)
    """
    vids, lbls = zip(*batch)
    vids = torch.stack([_normalize_tensor(v) for v in vids])
    return vids, torch.stack(lbls)


def make_eval_loader(split,
                     batch_size,
                     num_workers=4):
    """
    Create an evaluation data loader (validation or test).
    
    Args:
        split: Split name ('val' or 'test')
        batch_size: Batch size
        num_workers: Number of data loading workers (default: 4)
    
    Returns:
        DataLoader for evaluation set
    """
    ds = load_split(split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
        num_workers=num_workers,
        pin_memory=True
    )
