#!/usr/bin/env python
import argparse
import csv
import gc
import json
import random
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from tools.gradcam_dashboard import (
    GradCAM,
    build_model,
    cam_frame_for_index,
    get_module,
    overlay_cam,
    sample_video,
    save_image,
    tensor_from_frames,
)


R3D_STAGES = [
    ("layer1", "net.layer1", "Early 3D block"),
    ("layer2", "net.layer2", "Mid-low 3D block"),
    ("layer3", "net.layer3", "Mid-high 3D block"),
    ("layer4", "net.layer4", "Final 3D block"),
]

SIMPLE_STAGES = [
    ("conv1", "cnn.0", "First Conv3D"),
    ("conv2", "cnn.5", "Second Conv3D"),
    ("conv3", "cnn.10", "Final Conv3D"),
]


def read_rows(manifest_path, splits):
    split_set = set(splits)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("split") in split_set]


def select_rows(rows, splits, videos_per_split, start_index=0):
    selected = []
    for split in splits:
        split_rows = [row for row in rows if row.get("split") == split]
        if start_index > 0:
            split_rows = split_rows[start_index:]
        if videos_per_split > 0:
            split_rows = split_rows[:videos_per_split]
        selected.extend(split_rows)
    return selected


def frame_indices(max_frames, frames_per_video, seed, row_index):
    rng = random.Random(seed + row_index)
    count = min(frames_per_video, max_frames)
    if count <= 1:
        return [max_frames // 2]
    return sorted(rng.sample(range(max_frames), count))


def load_checkpoint(model_name, checkpoint, device):
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    for param in model.parameters():
        param.requires_grad_(True)
    model.to(device).eval()
    if model_name == "simple3dlstm":
        model.lstm.train()
    return model


def predict(model, frames, threshold, temperature, device):
    with torch.no_grad():
        logit = model(tensor_from_frames(frames).to(device))
        prob = float(torch.sigmoid(logit / temperature).detach().cpu().item())
    return prob, int(prob >= threshold)


def stage_overlay(model, stage, frames, masks, frame_index, device):
    cam_runner = GradCAM(model, get_module(model, stage[1]))
    try:
        x = tensor_from_frames(frames).to(device)
        _logit, cam = cam_runner(x)
    finally:
        cam_runner.close()
    del x
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cam_frame = cam_frame_for_index(cam, frame_index, frames.shape[0])
    return overlay_cam(frames[frame_index], cam_frame, content_mask=masks[frame_index])


def image_ref(path):
    return str(Path("images") / path.name).replace("\\", "/")


def write_dashboard(output_dir, payload):
    payload_json = json.dumps(payload).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASD Layer Inspector</title>
  <style>
    :root {{ color-scheme:light; --ink:#172026; --muted:#62707c; --line:#d9dfe5; --bg:#f6f7f9; --panel:#fff; --ok:#176b38; --bad:#9f1d20; --blue:#195cc8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter, Segoe UI, Arial, sans-serif; }}
    header {{ position:sticky; top:0; z-index:4; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); padding:16px 22px; backdrop-filter:blur(8px); }}
    h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
    .note {{ color:var(--muted); font-size:13px; margin:0 0 12px; max-width:980px; line-height:1.45; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    button, select {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:8px 10px; font:inherit; font-size:13px; }}
    button {{ cursor:pointer; }}
    button.active {{ background:#172026; color:#fff; border-color:#172026; }}
    main {{ max-width:1500px; margin:0 auto; padding:18px 22px 40px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px; margin-bottom:16px; }}
    .stat {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:12px; }}
    .stat strong {{ display:block; font-size:22px; }}
    .stat span {{ color:var(--muted); font-size:12px; }}
    .sample {{ border:1px solid var(--line); background:#fff; border-radius:8px; margin-bottom:18px; overflow:hidden; }}
    .sample-head {{ display:flex; justify-content:space-between; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0 0 6px; font-size:16px; overflow-wrap:anywhere; }}
    .meta, .badges {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:12px; }}
    .badge {{ border-radius:999px; padding:3px 8px; font-size:12px; font-weight:700; display:inline-flex; }}
    .ok {{ background:#e7f6ec; color:var(--ok); }}
    .bad {{ background:#fdecec; color:var(--bad); }}
    .split {{ background:#eaf1ff; color:var(--blue); }}
    .label {{ background:#f0f2f4; color:#334; }}
    .frames {{ display:grid; gap:18px; padding:16px; }}
    .frame {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .frame-title {{ padding:10px 12px; border-bottom:1px solid var(--line); color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:220px 1fr 1fr; gap:1px; background:var(--line); }}
    figure {{ margin:0; background:#fff; min-width:0; }}
    img {{ display:block; width:100%; aspect-ratio:1 / 1; object-fit:cover; background:#888; }}
    figcaption {{ padding:9px 10px; font-size:12px; color:var(--muted); line-height:1.35; }}
    figcaption strong {{ color:var(--ink); display:block; margin-bottom:3px; }}
    .stage-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:1px; background:var(--line); }}
    .hidden {{ display:none !important; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns:1fr; }} .sample-head {{ display:block; }} }}
    @media (max-width: 700px) {{ header, main {{ padding-left:14px; padding-right:14px; }} h1 {{ font-size:22px; }} .stage-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>ASD Layer Inspector</h1>
    <p class="note">Use this page to separate correctness from explanation. Correct/wrong badges come from the model prediction. Heatmaps show selected layer influence and are masked so gray padding is not visualized as important.</p>
    <div class="toolbar">
      <button class="active" data-filter="all">All</button>
      <button data-filter="wrong">Any wrong</button>
      <button data-filter="train">Train</button>
      <button data-filter="val">Validation</button>
      <button data-filter="test">Test</button>
      <button data-filter="positive">ASD positive</button>
      <button data-filter="negative">ASD negative</button>
      <button class="active" data-model="r3d">R3D stages</button>
      <button class="active" data-model="simple3dlstm">Simple3DLSTM stages</button>
      <select id="sort"><option value="split">Split order</option><option value="wrong">Wrong first</option><option value="name">Filename</option></select>
    </div>
  </header>
  <main>
    <section class="stats" id="stats"></section>
    <section id="samples"></section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const root = document.getElementById('samples');
    const stats = document.getElementById('stats');
    const visibleModels = {{ r3d:true, simple3dlstm:true }};
    let filter = 'all';
    let sortMode = 'split';
    const esc = value => String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    const badge = (text, cls) => `<span class="badge ${{cls}}">${{esc(text)}}</span>`;
    function visible(sample) {{
      if (filter === 'all') return true;
      if (['train','val','test'].includes(filter)) return sample.split === filter;
      if (filter === 'positive') return sample.label === 1;
      if (filter === 'negative') return sample.label === 0;
      if (filter === 'wrong') return !sample.r3d.correct || !sample.simple3dlstm.correct;
      return true;
    }}
    function ordered() {{
      const out = payload.samples.filter(visible).slice();
      if (sortMode === 'name') out.sort((a,b) => a.name.localeCompare(b.name));
      if (sortMode === 'wrong') out.sort((a,b) => Number(a.r3d.correct && a.simple3dlstm.correct) - Number(b.r3d.correct && b.simple3dlstm.correct));
      return out;
    }}
    function prediction(model) {{
      return `${{badge(model.correct ? 'correct' : 'wrong', model.correct ? 'ok' : 'bad')}} <span>p=${{model.prob.toFixed(3)}}</span> <span>pred=${{model.pred}}</span>`;
    }}
    function stageFigure(stage) {{
      return `<figure><img src="${{stage.image}}" alt="${{esc(stage.model)}} ${{esc(stage.name)}} Grad-CAM"><figcaption><strong>${{esc(stage.name)}}</strong>${{esc(stage.description)}}</figcaption></figure>`;
    }}
    function modelBlock(frame, modelKey, title) {{
      const cls = visibleModels[modelKey] ? '' : ' hidden';
      return `<div class="${{cls}}" data-model-block="${{modelKey}}"><div class="stage-grid">${{frame[modelKey].stages.map(stageFigure).join('')}}</div></div>`;
    }}
    function sampleCard(sample) {{
      return `<article class="sample">
        <div class="sample-head">
          <div>
            <h2>${{esc(sample.name)}}</h2>
            <div class="meta">${{badge(sample.split, 'split')}}${{badge(sample.label ? 'ASD positive' : 'ASD negative', 'label')}}<span>${{esc(sample.path)}}</span></div>
          </div>
          <div class="badges"><span>R3D ${{prediction(sample.r3d)}}</span><span>Simple ${{prediction(sample.simple3dlstm)}}</span></div>
        </div>
        <div class="frames">${{sample.frames.map(frame => `<section class="frame">
          <div class="frame-title">frame ${{frame.frame_index}} · original plus layer-stage Grad-CAM overlays</div>
          <div class="grid">
            <figure><img src="${{frame.original}}" alt="Original frame"><figcaption><strong>Original</strong>letterboxed video frame</figcaption></figure>
            ${{modelBlock(frame, 'r3d', 'R3D')}}
            ${{modelBlock(frame, 'simple3dlstm', 'Simple3DLSTM')}}
          </div>
        </section>`).join('')}}</div>
      </article>`;
    }}
    function renderStats(list) {{
      const count = list.length;
      const wrongR3D = list.filter(s => !s.r3d.correct).length;
      const wrongSimple = list.filter(s => !s.simple3dlstm.correct).length;
      const splitCounts = payload.splits.map(s => `${{s}} ${{list.filter(x => x.split === s).length}}`).join(' · ');
      stats.innerHTML = [
        ['Videos shown', count],
        ['R3D wrong', wrongR3D],
        ['Simple wrong', wrongSimple],
        ['Splits', splitCounts]
      ].map(([label, value]) => `<div class="stat"><strong>${{value}}</strong><span>${{label}}</span></div>`).join('');
    }}
    function render() {{
      const list = ordered();
      renderStats(list);
      root.innerHTML = list.map(sampleCard).join('');
    }}
    document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
      button.classList.add('active'); filter = button.dataset.filter; render();
    }}));
    document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => {{
      const key = button.dataset.model; visibleModels[key] = !visibleModels[key]; button.classList.toggle('active', visibleModels[key]); render();
    }}));
    document.getElementById('sort').addEventListener('change', event => {{ sortMode = event.target.value; render(); }});
    render();
  </script>
</body>
</html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")
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
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--videos_per_split", type=int, default=0,
                        help="0 means include every video in each selected split")
    parser.add_argument("--start_index", type=int, default=0,
                        help="zero-based offset within each split for chunked dashboards")
    parser.add_argument("--frames_per_video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--r3d_temperature", type=float, default=1.0)
    parser.add_argument("--simple_temperature", type=float, default=1.0)
    parser.add_argument("--pad_value", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output_dir", default="gradcam_outputs/layer_inspection_app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no_server", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = select_rows(
        read_rows(args.manifest, args.splits),
        args.splits,
        args.videos_per_split,
        start_index=args.start_index,
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    r3d = load_checkpoint("r3d", args.r3d_checkpoint, device)
    simple = load_checkpoint("simple3dlstm", args.simple_checkpoint, device)

    payload = {
        "splits": args.splits,
        "threshold": args.threshold,
        "models": {
            "r3d": {"temperature": args.r3d_temperature, "stages": R3D_STAGES},
            "simple3dlstm": {"temperature": args.simple_temperature, "stages": SIMPLE_STAGES},
        },
        "samples": [],
    }

    for row_index, row in enumerate(rows):
        path = Path(row["path"])
        label = int(row["label"])
        frames, masks = sample_video(
            path, max_frames=args.max_frames, pad_value=args.pad_value,
            return_masks=True
        )
        r3d_prob, r3d_pred = predict(
            r3d, frames, args.threshold, args.r3d_temperature, device
        )
        simple_prob, simple_pred = predict(
            simple, frames, args.threshold, args.simple_temperature, device
        )
        sample = {
            "name": path.name,
            "path": str(path),
            "split": row["split"],
            "label": label,
            "r3d": {"prob": r3d_prob, "pred": r3d_pred, "correct": r3d_pred == label},
            "simple3dlstm": {
                "prob": simple_prob, "pred": simple_pred,
                "correct": simple_pred == label,
            },
            "frames": [],
        }

        for frame_index in frame_indices(args.max_frames, args.frames_per_video, args.seed, row_index):
            frame_index = min(frame_index, frames.shape[0] - 1)
            stem = f"{row_index + 1:03d}_{row['split']}_{frame_index:02d}_{path.stem}"
            original_path = image_dir / f"{stem}_original.jpg"
            save_image(original_path, frames[frame_index])
            frame_payload = {
                "frame_index": frame_index,
                "original": image_ref(original_path),
                "r3d": {"stages": []},
                "simple3dlstm": {"stages": []},
            }
            for stage in R3D_STAGES:
                out_path = image_dir / f"{stem}_r3d_{stage[0]}.jpg"
                save_image(out_path, stage_overlay(r3d, stage, frames, masks, frame_index, device))
                frame_payload["r3d"]["stages"].append({
                    "model": "R3D",
                    "name": stage[0],
                    "description": stage[2],
                    "image": image_ref(out_path),
                })
            for stage in SIMPLE_STAGES:
                out_path = image_dir / f"{stem}_simple_{stage[0]}.jpg"
                save_image(out_path, stage_overlay(simple, stage, frames, masks, frame_index, device))
                frame_payload["simple3dlstm"]["stages"].append({
                    "model": "Simple3DLSTM",
                    "name": stage[0],
                    "description": stage[2],
                    "image": image_ref(out_path),
                })
            sample["frames"].append(frame_payload)
        payload["samples"].append(sample)

    write_dashboard(output_dir, payload)
    print(f"Wrote {output_dir / 'index.html'}")
    print(f"Wrote {output_dir / 'metadata.json'}")
    print(f"Videos: {len(payload['samples'])}")

    if not args.no_server:
        ensure_port_available(args.host, args.port)
        serve(output_dir, args.host, args.port)


if __name__ == "__main__":
    main()
