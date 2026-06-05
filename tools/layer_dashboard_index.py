#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="gradcam_outputs")
    parser.add_argument("--output_dir", default="gradcam_outputs/layer_inspection_index")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboards = [
        "layer_inspection_val_test",
        "layer_inspection_train_001_020",
        "layer_inspection_train_021_040",
        "layer_inspection_train_041_060",
        "layer_inspection_train_061_074",
    ]
    cards = []
    for name in dashboards:
        meta_path = root / name / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        samples = meta["samples"]
        splits = ", ".join(sorted({sample["split"] for sample in samples}))
        cards.append({
            "name": name.replace("layer_inspection_", "").replace("_", " "),
            "href": f"../{name}/index.html",
            "videos": len(samples),
            "splits": splits,
            "r3d_wrong": sum(not sample["r3d"]["correct"] for sample in samples),
            "simple_wrong": sum(not sample["simple3dlstm"]["correct"] for sample in samples),
        })

    items = "\n".join(
        f"""<a class="card" href="{card['href']}">
          <strong>{card['name']}</strong>
          <span>{card['videos']} videos · {card['splits']}</span>
          <span>R3D wrong {card['r3d_wrong']} · Simple wrong {card['simple_wrong']}</span>
        </a>"""
        for card in cards
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASD Layer Dashboard Index</title>
  <style>
    body {{ margin:0; font-family:Inter, Segoe UI, Arial, sans-serif; background:#f6f7f9; color:#172026; }}
    header {{ padding:28px 24px 14px; background:#fff; border-bottom:1px solid #d9dfe5; }}
    h1 {{ margin:0 0 8px; font-size:26px; }}
    p {{ margin:0; color:#62707c; }}
    main {{ max-width:920px; margin:0 auto; padding:22px; display:grid; gap:12px; }}
    .card {{ display:grid; gap:6px; padding:16px; border:1px solid #d9dfe5; border-radius:8px; background:#fff; color:inherit; text-decoration:none; }}
    .card:hover {{ border-color:#172026; }}
    .card strong {{ font-size:17px; text-transform:capitalize; }}
    .card span {{ color:#62707c; font-size:13px; }}
  </style>
</head>
<body>
  <header><h1>ASD Layer Dashboard Index</h1><p>Open validation/test or train chunks. Each dashboard shows original frames plus R3D and Simple3DLSTM layer-stage Grad-CAM.</p></header>
  <main>{items}</main>
</body>
</html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
