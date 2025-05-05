# ── save_numpy_dataset.py  (with leakage check) ──────────────────────────────
#!/usr/bin/env python
# coding: utf-8
"""
Create train / val / test .npz files with *no* duplicate videos
across splits (hash‑level guaranteed).
"""
import os, cv2, random, argparse, hashlib, numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

TARGET_SIZE = (112, 112)
MAX_FRAMES  = 60
SEED        = 42
HASH_NBYTES = 1_000_000      # hash first 1 MB

random.seed(SEED)
np.random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────────
def quick_hash(path, n=HASH_NBYTES, algo="sha1"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()

def load_video(path):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # pick indices (may end up empty if total==0)
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
            fr = np.zeros((*TARGET_SIZE, 3), np.uint8)          # ← fallback
        fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        frames.append(fr)

    cap.release()

    # ── NEW: deal with *completely unreadable* video ────────────────────
    if len(frames) == 0:                                      # no frame decoded
        return np.zeros((MAX_FRAMES, *TARGET_SIZE, 3), dtype="uint8")

    # pad short clips
    if len(frames) < MAX_FRAMES:
        pad = frames[-1]
        frames += [pad] * (MAX_FRAMES - len(frames))

    return np.stack(frames).astype("uint8")          # (T,H,W,C)

# ────────────────────────────────────────────────────────────────────────────
def main(root):
    # 1) crawl both class folders ------------------------------------------------
    raw = []            # tuples: (hash, path, label)
    for lab, folder in [(1, "positive"), (0, "negative")]:
        full = os.path.join(root, folder)
        for fn in os.listdir(full):
            if fn.lower().endswith((".mp4", ".webm", ".avi")):
                path  = os.path.join(full, fn)
                h     = quick_hash(path)
                raw.append((h, path, lab))

    # 2) de‑duplicate ------------------------------------------------------------
    chosen_paths, chosen_labels = [], []
    seen = {}           # hash ➜ (path, label)
    clashes = defaultdict(list)

    for h, path, lab in raw:
        if h not in seen:
            seen[h] = (path, lab)
            chosen_paths.append(path)
            chosen_labels.append(lab)
        else:
            prev_path, prev_lab = seen[h]
            if prev_lab != lab:
                clashes[h].append((prev_path, prev_lab))
                clashes[h].append((path, lab))
            # else (same label) silently keep the first one

    if clashes:
        print("⚠️  Conflicting duplicates found (same video, different labels) – keeping the FIRST copy only:\n")
        for h, items in clashes.items():
            for p, l in items:
                print(f"  {os.path.basename(p)}  label={l}")
        print()

    # 3) train / val / test split ------------------------------------------------
    tr, te, y_tr, y_te = train_test_split(
        chosen_paths, chosen_labels, test_size=.20,
        stratify=chosen_labels, random_state=SEED)

    tr, vl, y_tr, y_vl = train_test_split(
        tr, y_tr, test_size=.125, stratify=y_tr, random_state=SEED)

    splits = [("train", tr, y_tr), ("val", vl, y_vl), ("test", te, y_te)]

    # 4) save .npz ---------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    for split, plist, lab in splits:
        X = np.stack([load_video(p) for p in plist])
        y = np.array(lab, dtype=np.uint8)
        np.savez_compressed(f"data/{split}.npz", X=X, y=y)
        print(f"{split}: {X.shape[0]} videos  (pos={int(y.sum())}, neg={len(y)-int(y.sum())})")

if __name__ == "__main__":
    main(r"C:\Users\zhams\HWs\Autsim\dataset")
