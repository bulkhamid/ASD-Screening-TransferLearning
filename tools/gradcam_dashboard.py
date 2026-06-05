#!/usr/bin/env python
import argparse
import csv
import html
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from data.video_utils import resize_with_letterbox
from src.models.pretrained_r3d import PretrainedR3D
from src.models.simple3dlstm import Simple3DLSTM

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]


def get_module(root, dotted_name):
    module = root
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def build_model(name):
    if name == "r3d":
        return PretrainedR3D()
    if name == "simple3dlstm":
        return Simple3DLSTM(60, (112, 112), p_drop=0.0, p_head=0.0)
    raise ValueError(f"Unknown model: {name}")


def default_layer(name):
    return "net.layer4" if name == "r3d" else "cnn.10"


def sample_video(path, max_frames=60, target_size=(112, 112), pad_value=128,
                 return_masks=False):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise RuntimeError(f"Could not read frames from {path}")
    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, max_frames).round().astype(int).tolist()

    frames = []
    masks = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
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
    if len(frames) < max_frames:
        frames += [frames[-1]] * (max_frames - len(frames))
        masks += [masks[-1]] * (max_frames - len(masks))
    frames = np.stack(frames[:max_frames]).astype(np.uint8)
    if not return_masks:
        return frames
    return frames, np.stack(masks[:max_frames]).astype(bool)


def tensor_from_frames(frames):
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    video = (video - IMAGENET_MEAN) / IMAGENET_STD
    return video.unsqueeze(0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd = target_layer.register_forward_hook(self._forward_hook)

    def close(self):
        self.fwd.remove()

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        acts = self.activations.detach()[0]
        grads = self.gradients.detach()[0]
        if acts.ndim != 4:
            raise RuntimeError(f"Expected 3D feature map (C,T,H,W), got {acts.shape}")
        weights = grads.mean(dim=(1, 2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return logit.detach(), cam


def overlay_cam(frame, cam_frame, alpha=0.45, content_mask=None):
    heat = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]))
    if content_mask is not None:
        heat = np.where(content_mask, heat, 0)
        if heat.max() > 0:
            heat = heat / heat.max()
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * frame + alpha * heat)
    if content_mask is None:
        return overlay
    result = frame.copy()
    result[content_mask] = overlay[content_mask]
    return result


def cam_frame_for_index(cam, frame_index, max_frames):
    if cam.shape[0] == max_frames:
        idx = frame_index
    else:
        idx = int(round(frame_index * (cam.shape[0] - 1) / max(1, max_frames - 1)))
    return cam[min(max(idx, 0), cam.shape[0] - 1)]


def read_manifest(path, split, limit):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split:
                rows.append(row)
    return rows[:limit]


def save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def render_html(output_dir, model_name, layer_name, cards):
    rows = []
    for card in cards:
        badge_class = "correct" if card["correct"] else "wrong"
        rows.append(f"""
        <article class="card">
          <img src="{html.escape(card['image'])}" alt="Grad-CAM overlay">
          <div class="meta">
            <span class="badge {badge_class}">{'correct' if card['correct'] else 'wrong'}</span>
            <span>probability {card['prob']:.3f}</span>
            <span>label {card['label']}</span>
          </div>
          <h2>{html.escape(card['name'])}</h2>
          <p>{html.escape(card['path'])}</p>
        </article>
        """)

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASD Grad-CAM Dashboard</title>
  <style>
    :root {{ color-scheme: light; --ink:#172026; --muted:#62707c; --line:#d8dee4; --bg:#f6f7f9; --panel:#ffffff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:28px 32px 18px; border-bottom:1px solid var(--line); background:var(--panel); }}
    h1 {{ margin:0 0 8px; font-size:28px; font-weight:650; letter-spacing:0; }}
    .sub {{ color:var(--muted); font-size:14px; display:flex; gap:18px; flex-wrap:wrap; }}
    main {{ padding:24px 32px 40px; display:grid; grid-template-columns:repeat(auto-fill, minmax(290px, 1fr)); gap:18px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .card img {{ width:100%; aspect-ratio:1 / 1; object-fit:cover; display:block; background:#111; }}
    .card h2 {{ margin:12px 14px 6px; font-size:15px; line-height:1.3; font-weight:650; }}
    .card p {{ margin:0 14px 14px; color:var(--muted); font-size:12px; line-height:1.4; overflow-wrap:anywhere; }}
    .meta {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:12px 14px 0; color:var(--muted); font-size:12px; }}
    .badge {{ border-radius:999px; padding:3px 8px; font-weight:650; }}
    .correct {{ background:#e7f6ec; color:#176b38; }}
    .wrong {{ background:#fdecec; color:#9f1d20; }}
  </style>
</head>
<body>
  <header>
    <h1>ASD Grad-CAM Dashboard</h1>
    <div class="sub"><span>model {html.escape(model_name)}</span><span>layer {html.escape(layer_name)}</span><span>{len(cards)} samples</span></div>
  </header>
  <main>
    {''.join(rows)}
  </main>
</body>
</html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["r3d", "simple3dlstm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/split_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frame_index", type=int, default=30)
    parser.add_argument("--pad_value", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_name = args.layer or default_layer(args.model)
    output_dir = Path(args.output_dir or f"gradcam_outputs/{args.model}_{args.split}")
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    for param in model.parameters():
        param.requires_grad_(True)
    model.to(device).eval()
    if args.model == "simple3dlstm":
        model.lstm.train()
    cam_runner = GradCAM(model, get_module(model, layer_name))

    cards = []
    for i, row in enumerate(read_manifest(args.manifest, args.split, args.limit), 1):
        path = Path(row["path"])
        label = int(row["label"])
        frames, masks = sample_video(path, pad_value=args.pad_value, return_masks=True)
        x = tensor_from_frames(frames).to(device)
        logit, cam = cam_runner(x)
        prob = float(torch.sigmoid(logit).cpu().item())
        pred = int(prob >= args.threshold)
        frame_index = min(max(args.frame_index, 0), frames.shape[0] - 1)
        overlay = overlay_cam(
            frames[frame_index],
            cam_frame_for_index(cam, frame_index, frames.shape[0]),
            content_mask=masks[frame_index],
        )
        image_name = f"{i:03d}_{args.model}_{path.stem}.jpg"
        image_path = image_dir / image_name
        save_image(image_path, overlay)
        cards.append({
            "image": str(Path("images") / image_name).replace("\\", "/"),
            "prob": prob,
            "label": label,
            "correct": pred == label,
            "name": path.name,
            "path": str(path),
        })

    cam_runner.close()
    render_html(output_dir, args.model, layer_name, cards)
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
