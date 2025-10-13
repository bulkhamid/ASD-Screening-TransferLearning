#!/usr/bin/env python
# coding: utf-8
"""
Create train/val/test .npz files with no duplicate videos across splits.

This script processes video files from 'positive' and 'negative' class folders,
removes duplicates using hash-based deduplication, and creates stratified
train/validation/test splits saved as compressed .npz files.

The hash-level guarantee prevents data leakage across splits.
"""
import os
import cv2
import random
import hashlib
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Constants
TARGET_SIZE = (112, 112)  # Target frame size (height, width)
MAX_FRAMES = 60           # Maximum number of frames to sample per video
SEED = 42                 # Random seed for reproducibility
HASH_NBYTES = 1_000_000   # Number of bytes to read for hashing (1 MB)

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def quick_hash(path, n=HASH_NBYTES, algo="sha1"):
    """
    Compute a hash of the first n bytes of a file for duplicate detection.
    
    Args:
        path: Path to the file to hash
        n: Number of bytes to read (default: HASH_NBYTES)
        algo: Hash algorithm to use (default: "sha1")
    
    Returns:
        Hexadecimal digest string
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def load_video(path):
    """
    Load and preprocess a video file into a fixed-size array.
    
    Samples up to MAX_FRAMES frames from the video, resizes them to TARGET_SIZE,
    and converts color space from BGR to RGB. Videos shorter than MAX_FRAMES
    are padded by repeating the last frame.
    
    Args:
        path: Path to the video file
    
    Returns:
        numpy array of shape (MAX_FRAMES, HEIGHT, WIDTH, 3) with dtype uint8
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Pick frame indices (may end up empty if total==0)
    if total == 0:
        idxs = []
    elif total <= MAX_FRAMES:
        idxs = list(range(total))
    else:
        idxs = sorted(random.sample(range(total), MAX_FRAMES))

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            # Fallback: create a black frame
            fr = np.zeros((*TARGET_SIZE, 3), np.uint8)
        fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        frames.append(fr)

    cap.release()

    # Handle completely unreadable videos
    if len(frames) == 0:
        return np.zeros((MAX_FRAMES, *TARGET_SIZE, 3), dtype="uint8")

    # Pad short clips by repeating the last frame
    if len(frames) < MAX_FRAMES:
        pad = frames[-1]
        frames += [pad] * (MAX_FRAMES - len(frames))

    return np.stack(frames).astype("uint8")  # (T, H, W, C)


# ----------------------------------------------------------------------------
# Main Processing
# ----------------------------------------------------------------------------
def main(root):
    """
    Process video dataset and create train/val/test splits.
    
    Steps:
    1. Crawl 'positive' and 'negative' class folders
    2. Deduplicate videos based on hash
    3. Create stratified train/val/test splits (70/10/20)
    4. Save as compressed .npz files
    
    Args:
        root: Root directory containing 'positive' and 'negative' subdirectories
    """
    # 1) Crawl both class folders
    raw = []  # List of tuples: (hash, path, label)
    for lab, folder in [(1, "positive"), (0, "negative")]:
        full = os.path.join(root, folder)
        if not os.path.exists(full):
            print(f"Warning: Directory not found: {full}")
            continue
            
        for fn in os.listdir(full):
            if fn.lower().endswith((".mp4", ".webm", ".avi")):
                path = os.path.join(full, fn)
                h = quick_hash(path)
                raw.append((h, path, lab))

    if not raw:
        raise ValueError(f"No video files found in {root}")

    # 2) Deduplicate based on hash
    chosen_paths, chosen_labels = [], []
    seen = {}  # hash -> (path, label)
    clashes = defaultdict(list)

    for h, path, lab in raw:
        if h not in seen:
            seen[h] = (path, lab)
            chosen_paths.append(path)
            chosen_labels.append(lab)
        else:
            prev_path, prev_lab = seen[h]
            if prev_lab != lab:
                # Same video with different labels - conflict!
                clashes[h].append((prev_path, prev_lab))
                clashes[h].append((path, lab))
            # If same label, silently keep the first occurrence

    if clashes:
        print("Warning: Conflicting duplicates found (same video, different labels)")
        print("Keeping the FIRST copy only:\n")
        for h, items in clashes.items():
            for p, l in items:
                print(f"  {os.path.basename(p)}  label={l}")
        print()

    print(f"Found {len(chosen_paths)} unique videos "
          f"(removed {len(raw) - len(chosen_paths)} duplicates)")

    # 3) Create train/val/test splits (70/10/20 with stratification)
    tr, te, y_tr, y_te = train_test_split(
        chosen_paths, chosen_labels, test_size=0.20,
        stratify=chosen_labels, random_state=SEED
    )

    tr, vl, y_tr, y_vl = train_test_split(
        tr, y_tr, test_size=0.125,  # 0.125 * 0.80 = 0.10 of total
        stratify=y_tr, random_state=SEED
    )

    splits = [("train", tr, y_tr), ("val", vl, y_vl), ("test", te, y_te)]

    # 4) Save as .npz files
    os.makedirs("data", exist_ok=True)
    for split, plist, lab in splits:
        print(f"\nProcessing {split} split ({len(plist)} videos)...")
        X = np.stack([load_video(p) for p in plist])
        y = np.array(lab, dtype=np.uint8)
        
        output_path = f"data/{split}.npz"
        np.savez_compressed(output_path, X=X, y=y)
        
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        print(f"Saved {split}: {X.shape[0]} videos "
              f"(positive={pos_count}, negative={neg_count})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create train/val/test .npz files from video dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the root directory containing 'positive' and 'negative' folders"
    )
    args = parser.parse_args()
    
    main(args.data_root)
