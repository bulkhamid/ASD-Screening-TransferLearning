#!/usr/bin/env python
import argparse
import csv
import json
import random
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.gradcam_dashboard import (
    GradCAM,
    build_model,
    cam_frame_for_index,
    default_layer,
    get_module,
    overlay_cam,
    sample_video,
    save_image,
    tensor_from_frames,
)


def read_split_rows(manifest_path, split):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("split") == split]


def choose_samples(rows, videos, frames_per_video, max_frames, seed):
    rng = random.Random(seed)
    selected = rows[:]
    rng.shuffle(selected)
    selected = selected[: min(videos, len(selected))]

    samples = []
    for row in selected:
        frame_indices = sorted(rng.sample(range(max_frames), min(frames_per_video, max_frames)))
        samples.append({"row": row, "frames": frame_indices})
    return samples


def load_checkpoint(model_name, checkpoint, device):
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    for param in model.parameters():
        param.requires_grad_(True)
    model.to(device).eval()
    if model_name == "simple3dlstm":
        model.lstm.train()
    return model


def model_result(model, cam_runner, frames, masks, frame_index, threshold,
                 temperature, device):
    x = tensor_from_frames(frames).to(device)
    logit, cam = cam_runner(x)
    prob = float(torch.sigmoid(logit / temperature).detach().cpu().item())
    pred = int(prob >= threshold)
    cam_frame = cam_frame_for_index(cam, frame_index, frames.shape[0])
    overlay = overlay_cam(frames[frame_index], cam_frame,
                          content_mask=masks[frame_index])
    return prob, pred, overlay


def write_dashboard(output_dir, payload):
    payload_json = json.dumps(payload).replace("</", "<\\/")
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASD Model Inspector</title>
  <style>
    :root {{ color-scheme: light; --ink:#172026; --muted:#62707c; --line:#d8dee4; --bg:#f7f8fa; --panel:#fff; --green:#176b38; --red:#9f1d20; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, Segoe UI, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ position:sticky; top:0; z-index:5; padding:18px 24px; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
    h1 {{ margin:0 0 10px; font-size:24px; font-weight:700; letter-spacing:0; }}
    .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    button {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:8px 10px; border-radius:8px; font:inherit; font-size:13px; cursor:pointer; }}
    button.active {{ border-color:#172026; background:#172026; color:#fff; }}
    select {{ border:1px solid var(--line); background:#fff; padding:8px 10px; border-radius:8px; font:inherit; font-size:13px; }}
    main {{ max-width:1440px; margin:0 auto; padding:20px 24px 42px; }}
    .sample {{ margin:0 0 20px; border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }}
    .sample-head {{ display:flex; gap:12px; justify-content:space-between; align-items:flex-start; padding:14px 16px; border-bottom:1px solid var(--line); }}
    .sample-title {{ min-width:0; }}
    h2 {{ margin:0 0 6px; font-size:16px; line-height:1.25; overflow-wrap:anywhere; }}
    .meta {{ display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
    .badge {{ display:inline-flex; align-items:center; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:700; }}
    .ok {{ background:#e7f6ec; color:var(--green); }}
    .bad {{ background:#fdecec; color:var(--red); }}
    .frames {{ display:grid; gap:16px; padding:16px; }}
    .frame-block {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .frame-label {{ padding:10px 12px; color:var(--muted); font-size:12px; border-bottom:1px solid var(--line); }}
    .compare {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:1px; background:var(--line); }}
    .pane {{ background:#fff; min-width:0; }}
    .pane img {{ display:block; width:100%; aspect-ratio:1 / 1; object-fit:cover; background:#111; }}
    .pane figcaption {{ margin:0; padding:10px 12px; font-size:12px; color:var(--muted); display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .pane strong {{ color:var(--ink); }}
    .hidden {{ display:none !important; }}
    @media (max-width: 860px) {{
      header {{ padding:16px; }}
      main {{ padding:14px; }}
      .sample-head {{ display:block; }}
      .compare {{ grid-template-columns:1fr; }}
      h1 {{ font-size:22px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ASD Model Inspector</h1>
    <div class="toolbar">
      <button class="active" data-filter="all">All</button>
      <button data-filter="wrong">Wrong</button>
      <button data-filter="correct">Correct</button>
      <button data-filter="positive">ASD positive</button>
      <button data-filter="negative">ASD negative</button>
      <button class="active" data-toggle="original">Original</button>
      <button class="active" data-toggle="r3d">R3D</button>
      <button class="active" data-toggle="simple3dlstm">Simple3DLSTM</button>
      <select id="sort">
        <option value="order">Random order</option>
        <option value="gap">Largest model gap</option>
        <option value="name">Filename</option>
      </select>
    </div>
  </header>
  <main id="app"></main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const app = document.getElementById('app');
    const visible = {{ original:true, r3d:true, simple3dlstm:true }};
    let filter = 'all';
    let sortMode = 'order';

    function badge(correct) {{
      return `<span class="badge ${{correct ? 'ok' : 'bad'}}">${{correct ? 'correct' : 'wrong'}}</span>`;
    }}
    function sampleVisible(sample) {{
      if (filter === 'all') return true;
      if (filter === 'positive') return sample.label === 1;
      if (filter === 'negative') return sample.label === 0;
      return sample.frames.some(frame => frame.r3d.correct === (filter === 'correct') || frame.simple3dlstm.correct === (filter === 'correct'));
    }}
    function sortedSamples() {{
      const samples = payload.samples.filter(sampleVisible).slice();
      if (sortMode === 'name') samples.sort((a, b) => a.name.localeCompare(b.name));
      if (sortMode === 'gap') samples.sort((a, b) => Math.max(...b.frames.map(f => f.gap)) - Math.max(...a.frames.map(f => f.gap)));
      return samples;
    }}
    function pane(kind, frame, title, data) {{
      const cls = visible[kind] ? 'pane' : 'pane hidden';
      if (kind === 'original') {{
        return `<figure class="${{cls}}" data-kind="${{kind}}"><img src="${{frame.original}}" alt="Original frame"><figcaption><strong>${{title}}</strong></figcaption></figure>`;
      }}
      return `<figure class="${{cls}}" data-kind="${{kind}}"><img src="${{data.image}}" alt="${{title}} Grad-CAM"><figcaption><strong>${{title}}</strong>${{badge(data.correct)}}<span>p=${{data.prob.toFixed(3)}}</span><span>pred=${{data.pred}}</span></figcaption></figure>`;
    }}
    function render() {{
      app.innerHTML = sortedSamples().map(sample => `
        <section class="sample">
          <div class="sample-head">
            <div class="sample-title">
              <h2>${{sample.name}}</h2>
              <div class="meta"><span>label ${{sample.label}}</span><span>${{sample.split}}</span><span>${{sample.path}}</span></div>
            </div>
            <span class="badge ${{sample.label ? 'ok' : 'bad'}}">${{sample.label ? 'ASD positive' : 'ASD negative'}}</span>
          </div>
          <div class="frames">
            ${{sample.frames.map(frame => `
              <div class="frame-block">
                <div class="frame-label">frame ${{frame.frame_index}} · model probability gap ${{frame.gap.toFixed(3)}}</div>
                <div class="compare">
                  ${{pane('original', frame, 'Original')}}
                  ${{pane('r3d', frame, 'R3D', frame.r3d)}}
                  ${{pane('simple3dlstm', frame, 'Simple3DLSTM', frame.simple3dlstm)}}
                </div>
              </div>`).join('')}}
          </div>
        </section>`).join('');
    }}
    document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
      button.classList.add('active');
      filter = button.dataset.filter;
      render();
    }}));
    document.querySelectorAll('[data-toggle]').forEach(button => button.addEventListener('click', () => {{
      const key = button.dataset.toggle;
      visible[key] = !visible[key];
      button.classList.toggle('active', visible[key]);
      render();
    }}));
    document.getElementById('sort').addEventListener('change', event => {{
      sortMode = event.target.value;
      render();
    }});
    render();
  </script>
</body>
</html>"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(f"Port {port} is already in use")


def serve(output_dir, host, port):
    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Serving http://{host}:{port}/index.html")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3d_checkpoint", required=True)
    parser.add_argument("--simple_checkpoint", required=True)
    parser.add_argument("--manifest", default="data/split_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--videos", type=int, default=12)
    parser.add_argument("--frames_per_video", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--r3d_temperature", type=float, default=1.0)
    parser.add_argument("--simple_temperature", type=float, default=1.0)
    parser.add_argument("--pad_value", type=int, default=128)
    parser.add_argument("--output_dir", default="gradcam_outputs/inspection_app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no_server", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = read_split_rows(args.manifest, args.split)
    samples = choose_samples(rows, args.videos, args.frames_per_video, args.max_frames, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r3d = load_checkpoint("r3d", args.r3d_checkpoint, device)
    simple = load_checkpoint("simple3dlstm", args.simple_checkpoint, device)
    r3d_cam = GradCAM(r3d, get_module(r3d, default_layer("r3d")))
    simple_cam = GradCAM(simple, get_module(simple, default_layer("simple3dlstm")))

    payload = {
        "split": args.split,
        "seed": args.seed,
        "threshold": args.threshold,
        "models": {
            "r3d": {
                "checkpoint": args.r3d_checkpoint,
                "layer": default_layer("r3d"),
                "temperature": args.r3d_temperature,
            },
            "simple3dlstm": {
                "checkpoint": args.simple_checkpoint,
                "layer": default_layer("simple3dlstm"),
                "temperature": args.simple_temperature,
            },
        },
        "samples": [],
    }

    try:
        for sample_idx, sample in enumerate(samples, 1):
            row = sample["row"]
            path = Path(row["path"])
            label = int(row["label"])
            frames, masks = sample_video(
                path, max_frames=args.max_frames, pad_value=args.pad_value,
                return_masks=True
            )
            sample_payload = {
                "name": path.name,
                "path": str(path),
                "split": args.split,
                "label": label,
                "frames": [],
            }

            for frame_index in sample["frames"]:
                frame_index = min(frame_index, frames.shape[0] - 1)
                base_name = f"{sample_idx:03d}_{frame_index:02d}_{path.stem}"
                original_path = image_dir / f"{base_name}_original.jpg"
                save_image(original_path, frames[frame_index])

                r3d_prob, r3d_pred, r3d_overlay = model_result(
                    r3d, r3d_cam, frames, masks, frame_index, args.threshold,
                    args.r3d_temperature, device
                )
                simple_prob, simple_pred, simple_overlay = model_result(
                    simple, simple_cam, frames, masks, frame_index, args.threshold,
                    args.simple_temperature, device
                )
                r3d_path = image_dir / f"{base_name}_r3d.jpg"
                simple_path = image_dir / f"{base_name}_simple3dlstm.jpg"
                save_image(r3d_path, r3d_overlay)
                save_image(simple_path, simple_overlay)

                sample_payload["frames"].append({
                    "frame_index": frame_index,
                    "gap": abs(r3d_prob - simple_prob),
                    "original": str(Path("images") / original_path.name).replace("\\", "/"),
                    "r3d": {
                        "image": str(Path("images") / r3d_path.name).replace("\\", "/"),
                        "prob": r3d_prob,
                        "pred": r3d_pred,
                        "correct": r3d_pred == label,
                    },
                    "simple3dlstm": {
                        "image": str(Path("images") / simple_path.name).replace("\\", "/"),
                        "prob": simple_prob,
                        "pred": simple_pred,
                        "correct": simple_pred == label,
                    },
                })

            payload["samples"].append(sample_payload)
    finally:
        r3d_cam.close()
        simple_cam.close()

    write_dashboard(output_dir, payload)
    print(f"Wrote {output_dir / 'index.html'}")
    print(f"Wrote {output_dir / 'metadata.json'}")

    if not args.no_server:
        ensure_port_available(args.host, args.port)
        serve(output_dir, args.host, args.port)


if __name__ == "__main__":
    main()
