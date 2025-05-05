# ── src/data_cached.py ───────────────────────────────────────────────────────
import numpy as np, torch, random
import functools
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------#
# 0.  Helper to load the memory‑mapped .npz split that save_numpy_dataset.py
#     produced.  Shape on disk = (N,T,H,W,C) uint8.  Returned tensors are
#     (N,C,T,H,W) float32 in [0,1].
# ---------------------------------------------------------------------------#
def load_split(split):
    npz   = np.load(f"data/{split}.npz", mmap_mode="r")
    vids  = torch.from_numpy(npz["X"]).permute(0,4,1,2,3).float() / 255.
    labels= torch.from_numpy(npz["y"]).float()
    return TensorDataset(vids, labels)

# ---------------------------------------------------------------------------#
# 1.  Stochastic spatial + colour augmentation *for training only*
# ---------------------------------------------------------------------------#
def build_frame_aug(target_size=(224,224)):
    """Return a torchvision `Compose` to be applied to every single frame."""
    return transforms.Compose([
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),                 # → (C,H,W) in [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5)
    ])

def _normalize_tensor(video_5d):
    """
    video_5d : (C,T,H,W) float tensor in [0,1]
    Apply ImageNet mean/std **without** changing dtype.
    """
    m = torch.tensor(IMAGENET_MEAN, device=video_5d.device)[:, None, None, None]
    s = torch.tensor(IMAGENET_STD,  device=video_5d.device)[:, None, None, None]
    return (video_5d - m) / s

# ---------------------------------------------------------------------------#
# 2.  Ready‑made DataLoader builders
# ---------------------------------------------------------------------------#
def _train_collate(batch, frame_aug=None, aug_p=0.5):
    vids, lbls = zip(*batch)
    out = []
    to_pil = ToPILImage()  
    for v in vids:
        if frame_aug is not None and random.random() < aug_p:
            frs = [frame_aug(to_pil(v[:, t]))       # convert → PIL → aug
                   for t in range(v.shape[1])]
            v   = torch.stack(frs, dim=1)
        else:
            v = _normalize_tensor(v)
        out.append(v)
    return torch.stack(out), torch.stack(lbls)


def make_train_loader(batch_size,
                      aug_p=0.5,
                      target_size=(224, 224),
                      shuffle=True,
                      num_workers=4):
    ds        = load_split("train")
    frame_aug = build_frame_aug(target_size)
    collate   = functools.partial(_train_collate,
                                  frame_aug=frame_aug,
                                  aug_p=aug_p)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate,
                      num_workers=num_workers,
                      pin_memory=True)

def _eval_collate(batch):
    vids, lbls = zip(*batch)
    vids = torch.stack([_normalize_tensor(v) for v in vids])
    return vids, torch.stack(lbls)


def make_eval_loader(split,
                     batch_size,
                     num_workers=4):
    ds = load_split(split)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                     collate_fn=_eval_collate,
                      num_workers=num_workers,
                      pin_memory=True)
