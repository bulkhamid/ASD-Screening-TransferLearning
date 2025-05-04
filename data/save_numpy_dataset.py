# bin/save_numpy_dataset.py  (patch)
import os, cv2, numpy as np, argparse, random
from sklearn.model_selection import train_test_split

TARGET_SIZE = (224, 224)
MAX_FRAMES  = 60
SEED        = 42          # <─ reproducible!

# ----------------------------------------------------------
# 0.  Seed everything once
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------
def load_video(path):
    """Return a (T,H,W,C) uint8 tensor with ≤ MAX_FRAMES chronologically‑sorted
    frames randomly sampled from the whole clip."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES

    # choose which frame indices to grab
    if total <= MAX_FRAMES:
        idxs = list(range(total))                         # take all
    else:
        idxs = sorted(random.sample(range(total), MAX_FRAMES))  # reproducible
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            # fallback: repeat last good frame or insert zero frame
            fr = frames[-1] if frames else np.zeros((*TARGET_SIZE,3), np.uint8)
        fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        frames.append(fr)

    cap.release()

    # pad if the video was shorter than MAX_FRAMES
    if len(frames) < MAX_FRAMES:
        pad = np.zeros_like(frames[-1])
        frames += [pad] * (MAX_FRAMES - len(frames))

    return np.stack(frames).astype("uint8")               # (T,H,W,C)

# ----------------------------------------------------------
def main(root):
    paths, labels = [], []
    for lab, folder in [(1,"Positive_Trimmed"), (0,"Negative_Trimmed")]:
        full = os.path.join(root, folder)
        for fn in os.listdir(full):
            if fn.lower().endswith((".mp4",".webm",".avi")):
                paths.append(os.path.join(full, fn))
                labels.append(lab)

    tr, te, y_tr, y_te = train_test_split(
        paths, labels, test_size=.20, stratify=labels, random_state=SEED)
    tr, vl, y_tr, y_vl = train_test_split(
        tr,   y_tr, test_size=.125, stratify=y_tr, random_state=SEED)

    for split, plist, lab in [("train", tr, y_tr),
                              ("val",   vl, y_vl),
                              ("test",  te, y_te)]:
        X = np.stack([load_video(p) for p in plist])      # (N,T,H,W,C)
        y = np.array(lab, dtype=np.uint8)
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(f"data/{split}.npz", X=X, y=y)
        print(f"{split}: {X.shape[0]} videos saved.")

if __name__ == "__main__":
    main(r"C:\Users\zhams\HWs\Autsim")
