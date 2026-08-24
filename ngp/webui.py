"""Shared pieces of the two browser UIs: image encoding and the flat stylesheet.

Both `scripts/gui.py` (registration) and `scripts/gui_image.py` (image fitting)
serve a single self-contained page and poll a JSON endpoint, so the only things
worth sharing are how a tensor becomes a PNG data URI and what the page looks
like.
"""

from __future__ import annotations

import base64
import io

import matplotlib
import numpy as np
import torch
from PIL import Image

DISPLAY_H = 460                      # panel height in px; images are sent downsampled


def png_data_uri(rgb: np.ndarray, max_h: int = DISPLAY_H) -> str:
    im = Image.fromarray(rgb)
    if im.height > max_h:
        im = im.resize((max(1, round(im.width * max_h / im.height)), max_h),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_png(t: torch.Tensor, max_h: int = DISPLAY_H) -> str:
    a = np.clip(t.detach().cpu().numpy(), 0, 1)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return png_data_uri((a * 255).astype(np.uint8), max_h)


def cmap_png(a: np.ndarray, vmax: float, name="inferno", max_h: int = DISPLAY_H) -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    return png_data_uri((matplotlib.colormaps[name](x)[..., :3] * 255).astype(np.uint8),
                        max_h)


CSS = """
  :root { --fg:#fff; --bg:#000; --dim:#9a9a9a; --red:#e5484d; --blue:#4da3ff;
          --amber:#e5a23c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1400px; margin:0 auto; padding:26px 22px 48px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 6px; }
  .sub { font-size:12px; color:var(--dim); margin:0 0 22px; max-width:960px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:18px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .seg button:disabled { opacity:.35; cursor:default; }
  .knobs { display:flex; flex-wrap:wrap; gap:26px; margin:0 0 14px;
           padding:14px 16px; border:1px solid #333; }
  .knobs .title { width:100%; font-size:10px; letter-spacing:.16em;
                  text-transform:uppercase; color:var(--dim); margin-bottom:2px; }
  .knob { display:flex; flex-direction:column; gap:5px; min-width:250px; flex:1; }
  .knob .kl { font-size:11px; color:var(--dim); display:flex;
              justify-content:space-between; gap:12px; }
  .knob .kl b { color:var(--fg); font-weight:600;
                font-variant-numeric:tabular-nums; }
  .knob .ends { display:flex; justify-content:space-between; font-size:9px;
                color:#666; font-variant-numeric:tabular-nums; }
  input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                      height:1px; background:var(--fg); outline:none; margin:6px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
    appearance:none; width:13px; height:13px; background:var(--fg);
    border:1px solid var(--fg); cursor:pointer; border-radius:0; }
  input[type=range]::-moz-range-thumb { width:13px; height:13px;
    background:var(--fg); border:1px solid var(--fg); cursor:pointer;
    border-radius:0; }
  .row { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid #333; }
  .cap { font-size:10px; letter-spacing:.14em; text-transform:uppercase;
         color:var(--dim); }
  .cap i { color:var(--red); font-style:normal; }
  .cap b { color:var(--blue); font-weight:600; }
  .stats { font-size:12px; color:var(--dim); margin-top:18px;
           font-variant-numeric:tabular-nums; line-height:1.9; }
  .stats b { color:var(--fg); font-weight:600; }
  .stats .bad { color:var(--red); }
  .bar { height:2px; background:#222; margin:14px 0 0; }
  .bar i { display:block; height:2px; background:var(--fg); width:0; }
  .note { font-size:11px; color:#7a7a7a; margin-top:6px; }
  table.ladder { border-collapse:collapse; font-size:11px;
                 font-variant-numeric:tabular-nums; }
  table.ladder th { text-align:right; font-weight:600; color:var(--dim);
                    padding:2px 10px; font-size:10px; letter-spacing:.1em;
                    text-transform:uppercase; }
  table.ladder td { text-align:right; padding:2px 10px; color:#d8d8d8; }
  table.ladder tr.hashed td { color:var(--amber); }
"""
