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
  /* A row of equal panels that must not wrap: the canvases keep their pixel
     backing store and are scaled by CSS, so four of them share the width
     evenly however narrow the window is. */
  .row.equal { flex-wrap:nowrap; }
  .row.equal .panel { flex:1 1 0; min-width:0; }
  .row.equal canvas { width:100%; height:auto; }
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
  .setup { font-size:12px; color:#fff; margin:12px 0 2px;
           font-variant-numeric:tabular-nums; }
  .setup b { font-weight:600; }
  .setup span.dim { color:var(--dim); }
  table.ladder { border-collapse:collapse; font-size:11px;
                 font-variant-numeric:tabular-nums; }
  table.ladder th { text-align:right; font-weight:600; color:var(--dim);
                    padding:2px 10px; font-size:10px; letter-spacing:.1em;
                    text-transform:uppercase; }
  table.ladder td { text-align:right; padding:2px 10px; color:#d8d8d8; }
  table.ladder tr.hashed td { color:var(--amber); }
  .modal { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none;
           z-index:50; overflow:auto; }
  .modal.open { display:block; }
  .modal .sheet { max-width:820px; margin:6vh auto; background:#000;
                  border:1px solid var(--fg); padding:28px 30px 34px; }
  .modal h2 { font-size:13px; letter-spacing:.14em; text-transform:uppercase;
              margin:22px 0 8px; font-weight:600; }
  .modal h2:first-of-type { margin-top:0; }
  .modal p { font-size:13px; line-height:1.6; color:#d0d0d0; margin:0 0 10px; }
  .modal code { color:var(--amber); font-family:ui-monospace,Menlo,monospace;
                font-size:12px; }
  .modal a { color:var(--blue); }
  .modal .close { float:right; background:#000; color:var(--fg);
                  border:1px solid var(--fg); padding:5px 12px; cursor:pointer;
                  font:inherit; font-size:12px; }
  .modal ul { margin:0 0 10px; padding-left:20px; color:#d0d0d0; font-size:13px;
              line-height:1.6; }
"""

# One explainer, shared by both pages: what the encoding is, what each control
# changes, and which claims here were measured rather than assumed.
ABOUT_HTML = """
<button class="close" onclick="closeAbout()">close</button>
<h2>The idea</h2>
<p>Muller, Evans, Schied and Keller, <i>Instant Neural Graphics Primitives with a
Multiresolution Hash Encoding</i>, SIGGRAPH 2022
(<a href="https://arxiv.org/abs/2201.05989" target="_blank">arXiv:2201.05989</a>,
<a href="https://github.com/NVlabs/instant-ngp" target="_blank">NVlabs/instant-ngp</a>).</p>
<p>A coordinate goes in; a colour, a density or a displacement comes out. Rather
than ask one large network to carry every scale, the coordinate is first
<i>encoded</i>: <code>L</code> grids are laid over the domain at geometrically
growing resolutions <code>N_l = N_min * b^l</code>, each node carries <code>F</code>
learnable numbers, and a query interpolates the <code>2^D</code> corners of its cell
at every level. The levels are concatenated into an <code>L*F</code> vector that a
small MLP decodes. Both the table and the network are trained by ordinary
gradients.</p>

<h2>The hash table is the entire data structure</h2>
<p>Each level owns one flat array of <code>T = 2^log2_hashmap_size</code> feature
vectors. Nothing else exists: no tree, no nodes, no pointers, no traversal, no
refinement bookkeeping. A lookup is an index computation and a gather, O(1),
identical for every level and every point.</p>
<p>When a level's grid fits inside its table it is indexed <b>directly</b> and there
are no collisions at all -- the coarse levels of any sane configuration are plain
dense grids. When it does not fit, node coordinates are folded through a spatial
hash,</p>
<p><code>h(x) = ( XOR_d  x_d * pi_d )  mod  T</code>, &nbsp; pi = (1, 2654435761,
805459861, 3674653429)</p>
<p>and distinct nodes start <i>sharing</i> entries. This is the step that decouples
cost from resolution: a 512-cell-per-axis level in 3D has ~10^8 nodes, which nobody
stores, but its table is still <code>T</code> entries. Refining a level costs
nothing extra. In this page the ladder marks which levels are dense and which are
hashed -- move <code>log2 table size T</code> and watch the boundary move.</p>

<h2>Why collisions are the mechanism, not a defect</h2>
<p>Two far-apart fine nodes sharing an entry receive the <i>sum</i> of their
gradients. Where one of them sits in empty or unconstrained space it contributes
almost nothing, so the entry is won, automatically, by whichever node the data
actually constrains. No importance heuristic decides this and no pass detects it:
the optimiser does it as a side effect of training. The failure case is honest and
worth knowing -- two <i>equally</i> well-constrained distant regions colliding do
fight over one entry.</p>

<h2>The hierarchy is built for free</h2>
<p>This is the part worth dwelling on. An adaptive octree or quadtree has to be
<i>constructed</i>: decide where to subdivide, maintain the topology as the fit
changes, keep gradients consistent across a structure that is itself moving, and
serialise all of it. A hash encoding skips the entire problem. Every level is
evaluated at every query point, always. There is no decision about where to refine,
so there is nothing to build, nothing to update mid-training, and nothing that
serialises a GPU.</p>
<p>Adaptivity still happens -- it just falls out of the gradients rather than out of
bookkeeping. Entries only move where samples touch them; entries whose region
carries no signal keep their initialisation. The result is a fit that spends its
fine levels exactly where the data has detail, with no mechanism anywhere in the
code that aimed for that. The panel on this page showing cells drawn at the scale
of the level dominating each block is that effect, measured: on the painting, the
dark surround is carried by a level with ~33 px cells while the face and turban are
carried by one with ~3 px cells.</p>
<p>Free of <i>structure</i>, not free of memory: every level is allocated whether or
not it earns its keep. What is avoided is the <code>O(N^D)</code> growth of the fine
levels, and that is the whole game in 3D.</p>

<h2>The controls</h2>
<ul>
<li><code>levels L</code>, <code>growth b</code> -- the ladder of scales. The
coarsest should span the domain; the finest should stop at the spacing of your
data. Two levels landing on the same lattice are wasted: they are separate feature
sets on identical nodes, which <code>F</code> buys more directly.</li>
<li><code>features per level F</code> -- width per scale; doubling it doubles the
table.</li>
<li><code>log2 table size T</code> -- where collisions begin.</li>
<li><code>finest cells per axis</code> -- the misleading one. Refining past the
pixel spacing barely moves PSNR while costing parameters, and the structure it
invents below the sampling scale is invisible in the image and fatal in any
derivative taken through the fit.</li>
<li><code>interpolation</code> -- linear is the paper's default and only C0.
Smoothstep replaces the weight w by 3w^2-2w^3, making the encoding C1, which is
required for second derivatives.</li>
</ul>

<h2>Measured here, not assumed</h2>
<ul>
<li>Autograd through this pure-PyTorch encoder matches float64 finite differences
to 5.6e-10 median relative error.</li>
<li>With linear interpolation the Laplacian of a fit scores relative L2 of 1.011
against analytic truth -- exactly what predicting zero scores.</li>
<li>Capping the finest level at the image's pixel count gave <i>both</i> fewer
parameters and higher PSNR than leaving it uncapped: 5.92M / 40.28 dB against
7.66M / 39.54 dB.</li>
</ul>
"""
