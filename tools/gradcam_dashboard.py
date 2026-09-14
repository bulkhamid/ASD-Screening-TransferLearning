#!/usr/bin/env python
"""Generate traceable Grad-CAM views from exact cached clips."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from data.video_utils import resize_with_letterbox
from src.models.pretrained_r3d import PretrainedR3D
from src.models.simple2dcnn import Simple2DCNN
from src.models.simple3dlstm import Simple3DLSTM
from src.models.simple3dlstm_v2 import Simple3DLSTMV2
from src.utils.subject_data import KINETICS_MEAN, KINETICS_STD, unpack_masks


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_module(root, dotted_name):
    module = root
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def build_model(name, *, pretrained=False):
    if name == "r3d":
        return PretrainedR3D(pretrained=pretrained)
    if name in {"simple3dlstm", "simple3dlstm_legacy"}:
        return Simple3DLSTM(60, (112, 112), p_drop=0.0, p_head=0.0)
    if name == "simple3dlstm_v2":
        return Simple3DLSTMV2(p_drop=0.0)
    if name == "simple2dcnn":
        return Simple2DCNN(p_drop=0.0)
    raise ValueError(f"Unknown model: {name}")


def default_layer(name):
    return "net.layer4" if name == "r3d" else "cnn.10"


def sample_video(
    path,
    max_frames=60,
    target_size=(112, 112),
    pad_value=128,
    return_masks=False,
):
    """Legacy raw-video sampler retained only for existing local inspection apps."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise RuntimeError(f"Could not read frames from {path}")
    indices = (
        list(range(total))
        if total <= max_frames
        else np.linspace(0, total - 1, max_frames).round().astype(int).tolist()
    )
    frames, masks = [], []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, mask = resize_with_letterbox(
            frame, target_size, pad_value=pad_value, return_mask=True
        )
        frames.append(frame)
        masks.append(mask)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    while len(frames) < max_frames:
        frames.append(frames[-1])
        masks.append(masks[-1])
    frame_array = np.stack(frames[:max_frames]).astype(np.uint8)
    if not return_masks:
        return frame_array
    return frame_array, np.stack(masks[:max_frames]).astype(bool)


def tensor_from_frames(frames, mean=IMAGENET_MEAN, std=IMAGENET_STD, masks=None):
    """Normalize T,H,W,C frames and zero neutral letterbox padding."""
    video = torch.from_numpy(np.asarray(frames).copy()).permute(0, 3, 1, 2).float() / 255.0
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    video = (video - mean_tensor) / std_tensor
    if masks is not None:
        mask = torch.from_numpy(np.asarray(masks).copy()).bool().view(len(video), 1, *video.shape[-2:])
        video = torch.where(mask.expand_as(video), video, torch.zeros_like(video))
    return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def close(self):
        self.forward_handle.remove()

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradients = gradient

    def __call__(self, x, target_class=1):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        if target_class in (1, "positive"):
            resolved_target = 1
        elif target_class in (0, "negative"):
            resolved_target = 0
        elif target_class == "predicted":
            resolved_target = int(float(logit.detach()[0]) >= 0.0)
        else:
            raise ValueError(f"Unknown Grad-CAM target: {target_class}")
        score = logit if resolved_target == 1 else -logit
        score.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        if activations.ndim == 5:  # B,C,T,H,W from 3D convolution
            acts = activations[0]
            grads = gradients[0]
            weights = grads.mean(dim=(1, 2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=0))
        elif activations.ndim == 4:  # B*T,C,H,W from frame-level 2D convolution
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1))
        else:
            raise RuntimeError(f"Unsupported target activation shape: {activations.shape}")
        cam = cam.float().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return logit.detach(), cam


def overlay_cam(frame, cam_frame, alpha=0.45, content_mask=None):
    heat = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]))
    if content_mask is not None:
        heat = np.where(content_mask, heat, 0)
        if heat.max() > 0:
            heat = heat / heat.max()
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * frame + alpha * heat)
    if content_mask is None:
        return overlay
    result = frame.copy()
    result[content_mask] = overlay[content_mask]
    return result


def cam_frame_for_index(cam, frame_index, max_frames):
    if cam.shape[0] == max_frames:
        index = frame_index
    else:
        index = int(round(frame_index * (cam.shape[0] - 1) / max(1, max_frames - 1)))
    return cam[min(max(index, 0), cam.shape[0] - 1)]


def save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _load_checkpoint(model, path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
        return payload
    model.load_state_dict(payload)
    return {}


def _read_cache_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _filter_by_fold(rows, assignments_path, fold, role):
    if not assignments_path:
        return rows
    if fold is None or role is None:
        raise ValueError("--fold_assignments requires both --fold and --role")
    with Path(assignments_path).open(newline="", encoding="utf-8") as handle:
        subjects = {
            row["subject_id"]
            for row in csv.DictReader(handle)
            if int(row["fold"]) == fold and row["role"] == role
        }
    return [row for row in rows if row["subject_id"] in subjects]


def _load_cached_clip(cache_root: Path, row, clip_index):
    clips = np.load(cache_root / row["cache_path"], mmap_mode="r")
    packed_masks = np.load(cache_root / row["mask_path"], mmap_mode="r")
    masks = unpack_masks(packed_masks, clips.shape[3])
    if not 0 <= clip_index < len(clips):
        raise IndexError(f"clip index {clip_index} outside 0..{len(clips) - 1}")
    return np.asarray(clips[clip_index]).copy(), np.asarray(masks[clip_index]).copy()


def _resolve_target(requested, label, logit):
    if requested == "true":
        return int(label)
    if requested == "predicted":
        return int(logit >= 0)
    return 1 if requested == "positive" else 0


def render_html(output_dir, model_name, layer_name, target_name, cards):
    articles = []
    for card in cards:
        badge = "correct" if card["correct"] else "wrong"
        frames = "".join(
            f"""
            <figure>
              <div class="pair">
                <div><span>Original</span><img src="{html.escape(frame['original'])}" alt="Original frame"></div>
                <div><span>Target evidence</span><img src="{html.escape(frame['overlay'])}" alt="Grad-CAM overlay"></div>
              </div>
              <figcaption>frame {frame['frame_index']}</figcaption>
            </figure>
            """
            for frame in card["frames"]
        )
        articles.append(
            f"""
            <article>
              <div class="row"><strong>{html.escape(card['video'])}</strong><span class="badge {badge}">{'correct' if card['correct'] else 'wrong'}</span></div>
              <div class="meta"><span>subject {html.escape(card['subject'])}</span><span>session {html.escape(card['session'])}</span><span>clip {card['clip']}</span><span>target class {card['target']}</span><span>label {card['label']}</span><span>prediction {card['prediction']}</span><span>p(positive) {card['probability']:.3f}</span></div>
              <div class="frames">{frames}</div>
            </article>
            """
        )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traceable Grad-CAM Audit</title><style>
:root{{--ink:#162027;--muted:#60707b;--line:#d8dee3;--bg:#f5f7f8;--panel:#fff;--good:#176b38;--bad:#9f1d20}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}}header{{background:var(--panel);border-bottom:1px solid var(--line);padding:24px clamp(16px,4vw,44px)}}h1{{font-size:24px;margin:0 0 8px;letter-spacing:0}}header p{{margin:4px 0;color:var(--muted)}}main{{max-width:1500px;margin:auto;padding:20px clamp(12px,3vw,36px) 44px}}article{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}}.row,.meta{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}.row{{justify-content:space-between;font-size:16px}}.meta{{color:var(--muted);margin:8px 0 14px}}.badge{{padding:2px 8px;border-radius:999px;font-size:12px}}.correct{{color:var(--good);background:#e7f6ec}}.wrong{{color:var(--bad);background:#fdecec}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}figure{{margin:0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}.pair div{{position:relative;background:#20272b;aspect-ratio:1}}.pair span{{position:absolute;z-index:1;top:6px;left:6px;background:rgba(255,255,255,.9);padding:2px 5px;font-size:11px}}img{{width:100%;height:100%;object-fit:contain;display:block}}figcaption{{font-size:12px;color:var(--muted);padding-top:4px}}@media(max-width:560px){{article{{padding:10px}}.frames{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Traceable Grad-CAM Audit</h1><p>Model <b>{html.escape(model_name)}</b> | layer <b>{html.escape(layer_name)}</b> | target <b>{html.escape(target_name)}</b></p><p>Warm colors indicate spatial-temporal locations whose gradients support the selected target. They are explanations of local model sensitivity, not proof of a clinical cue.</p></header><main>{''.join(articles)}</main></body></html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["r3d", "simple2dcnn", "simple3dlstm", "simple3dlstm_legacy", "simple3dlstm_v2"],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--fold_assignments", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--role", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--clip_index", type=int, default=2)
    parser.add_argument("--frame_indices", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--target", choices=["positive", "negative", "predicted", "true"], default="predicted")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--normalization_json", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_name = args.layer or default_layer(args.model)
    output_dir = Path(args.output_dir or f"gradcam_outputs/{args.model}_grouped")
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=False)
    checkpoint_meta = _load_checkpoint(model, Path(args.checkpoint), device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.to(device).eval()
    if hasattr(model, "lstm"):
        model.lstm.train()
    cam_runner = GradCAM(model, get_module(model, layer_name))

    if args.model == "r3d":
        mean, std = KINETICS_MEAN, KINETICS_STD
    elif checkpoint_meta.get("normalization"):
        mean = checkpoint_meta["normalization"]["mean"]
        std = checkpoint_meta["normalization"]["std"]
    elif args.normalization_json:
        normalization = json.loads(Path(args.normalization_json).read_text(encoding="utf-8"))
        mean, std = normalization["mean"], normalization["std"]
    else:
        raise ValueError("Simple grouped models require normalization metadata from checkpoint or --normalization_json")

    cache_root = Path(args.cache_dir)
    manifest = Path(args.manifest or cache_root / "cache_manifest.csv")
    rows = _filter_by_fold(
        _read_cache_rows(manifest), args.fold_assignments, args.fold, args.role
    )[: args.limit]
    cards = []
    for row_index, row in enumerate(rows, 1):
        frames, masks = _load_cached_clip(cache_root, row, args.clip_index)
        x = tensor_from_frames(frames, mean, std, masks).to(device)
        requested_target = int(row["label"]) if args.target == "true" else args.target
        logit, cam = cam_runner(x, requested_target)
        logit_value = float(logit.cpu()[0])
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit_value))))
        prediction = int(logit_value >= 0)
        target = _resolve_target(args.target, int(row["label"]), logit_value)
        frame_cards = []
        for frame_index in args.frame_indices:
            frame_index = min(max(frame_index, 0), len(frames) - 1)
            stem = f"{row_index:03d}_clip{args.clip_index}_frame{frame_index}"
            original_path = image_dir / f"{stem}_original.jpg"
            overlay_path = image_dir / f"{stem}_target{target}.jpg"
            overlay = overlay_cam(
                frames[frame_index],
                cam_frame_for_index(cam, frame_index, len(frames)),
                content_mask=masks[frame_index],
            )
            save_image(original_path, frames[frame_index])
            save_image(overlay_path, overlay)
            frame_cards.append(
                {
                    "frame_index": frame_index,
                    "original": original_path.relative_to(output_dir).as_posix(),
                    "overlay": overlay_path.relative_to(output_dir).as_posix(),
                }
            )
        cards.append(
            {
                "video": row["video_id"][:12],
                "subject": row["subject_id"],
                "session": row["session_id"],
                "clip": args.clip_index,
                "target": target,
                "label": int(row["label"]),
                "prediction": prediction,
                "probability": probability,
                "correct": prediction == int(row["label"]),
                "frames": frame_cards,
            }
        )
    cam_runner.close()
    render_html(output_dir, args.model, layer_name, args.target, cards)
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
