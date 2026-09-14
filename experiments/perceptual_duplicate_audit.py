#!/usr/bin/env python
"""Find visually near-duplicate cached frames across subject boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.models import ResNet18_Weights, resnet18

from src.utils.subject_data import unpack_masks


def nearest_cross_subject_pairs(embeddings, metadata, *, top_k=50):
    normalized = functional.normalize(torch.as_tensor(embeddings).float(), dim=1)
    similarities = (normalized @ normalized.T).cpu().numpy()
    pairs = []
    for left in range(len(metadata)):
        for right in range(left + 1, len(metadata)):
            if metadata[left]["subject_id"] == metadata[right]["subject_id"]:
                continue
            pairs.append(
                {
                    "similarity": float(similarities[left, right]),
                    "left": metadata[left],
                    "right": metadata[right],
                }
            )
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)[:top_k]


def _content_crop(frame, mask):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Content mask is empty")
    return frame[y.min(): y.max() + 1, x.min(): x.max() + 1]


def _load_items(cache_root: Path, rows, frame_index: int):
    metadata, frames = [], []
    for row in rows:
        clips = np.load(cache_root / row["cache_path"], mmap_mode="r")
        packed = np.load(cache_root / row["mask_path"], mmap_mode="r")
        masks = unpack_masks(packed, clips.shape[3])
        index = min(max(frame_index, 0), clips.shape[1] - 1)
        for clip_index in range(len(clips)):
            frame = np.asarray(clips[clip_index, index]).copy()
            mask = np.asarray(masks[clip_index, index]).copy()
            frames.append(_content_crop(frame, mask))
            metadata.append(
                {
                    "video_id": row["video_id"],
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "label": int(row["label"]),
                    "clip_index": clip_index,
                    "frame_index": index,
                    "item_index": len(metadata),
                }
            )
    return frames, metadata


def _encode(frames, *, batch_size: int, device: torch.device):
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.to(device).eval()
    transform = weights.transforms()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = []
            for frame in frames[start: start + batch_size]:
                rgb = torch.from_numpy(frame.copy()).permute(2, 0, 1)
                batch.append(transform(rgb))
            encoded = encoder(torch.stack(batch).to(device))
            embeddings.append(encoded.float().cpu())
    return torch.cat(embeddings).numpy()


def _save_review(output_dir: Path, pair, frames, rank: int):
    left = frames[pair["left"]["item_index"]]
    right = frames[pair["right"]["item_index"]]
    height = 256
    panels = []
    for frame in (left, right):
        scale = height / frame.shape[0]
        width = max(1, int(round(frame.shape[1] * scale)))
        panels.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    canvas = np.full((height, panels[0].shape[1] + panels[1].shape[1] + 8, 3), 128, np.uint8)
    canvas[:, : panels[0].shape[1]] = panels[0]
    canvas[:, panels[0].shape[1] + 8:] = panels[1]
    path = output_dir / "review" / f"{rank:03d}_{pair['similarity']:.4f}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return path.relative_to(output_dir).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", default="runs/perceptual_duplicate_audit")
    parser.add_argument("--frame_index", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--review_threshold", type=float, default=0.98)
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    manifest_path = Path(args.manifest or cache_root / "cache_manifest.csv")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(not row["subject_id"] for row in rows):
        raise ValueError("Perceptual duplicate checks require completed subject IDs")
    frames, metadata = _load_items(cache_root, rows, args.frame_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = _encode(frames, batch_size=args.batch_size, device=device)
    pairs = nearest_cross_subject_pairs(embeddings, metadata, top_k=args.top_k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, pair in enumerate(pairs, 1):
        if pair["similarity"] >= args.review_threshold:
            pair["review_image"] = _save_review(output_dir, pair, frames, rank)
    output = {
        "encoder": "torchvision ResNet18 IMAGENET1K_V1 penultimate embedding",
        "items": len(metadata),
        "cross_subject_pairs_review_threshold": args.review_threshold,
        "pairs": pairs,
        "interpretation": "High similarity is a review candidate, not automatic proof of duplication.",
    }
    (output_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
