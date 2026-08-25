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
from matplotlib.colors import LinearSegmentedColormap
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


# The level ramp, shared with the pages' levColor(): bright at both ends, so the
# coarse levels are visible against a dark image and the legend means the same
# thing in the overlay and in the map.
LEVEL_RAMP = LinearSegmentedColormap.from_list(
    "levels", [(0.30, 0.64, 1.00), (0.25, 0.88, 0.82), (0.49, 1.00, 0.35),
               (1.00, 0.82, 0.30), (1.00, 0.42, 0.42)])


def cmap_png(a: np.ndarray, vmax: float, name="inferno", max_h: int = DISPLAY_H) -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    cmap = LEVEL_RAMP if name == "levels" else matplotlib.colormaps[name]
    return png_data_uri((cmap(x)[..., :3] * 255).astype(np.uint8), max_h)


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


# What every control on each page does. Separate from ABOUT_HTML, which explains
# the method: a reader who knows what a hash grid is still cannot guess what
# "displacement scale" or "level window" is wired to here.
INTERFACE_COMMON = """
<h2>Reading the run</h2>
<ul>
<li>The terminal prints <code>[run]</code> with the configuration, <code>[images]</code>
when the first frames go out, and <code>[done]</code> or <code>[stopped]</code> with the
final numbers. If the page looks empty and those lines are there, the problem is in the
browser, not the fit.</li>
<li><b>run / stop</b> -- stop leaves the fit where it is and reports it, rather than
discarding it.</li>
</ul>

<h2>Training</h2>
<ul>
<li><b>learning rate</b> -- Adam's step, log-spaced. The hash table tolerates 1e-2
comfortably; a dense control grid wants roughly 5x that, because each of its parameters
sees far more of the image.</li>
<li><b>iterations</b> -- how long, with a cosine decay to 3% of the starting rate. The
schedules that depend on it (level window, image pyramid) are expressed as fractions of
this, so changing it rescales them rather than truncating them.</li>
<li><b>batch size</b> -- sample points per step, drawn 90% inside the foreground mask and
10% uniformly. This is the knob that makes an empty background cost nothing: the points
are where the compute goes.</li>
</ul>
"""

INTERFACE_IMAGE = INTERFACE_COMMON + """
<h2>The encoding</h2>
<ul>
<li><b>levels L</b> and <b>growth b</b> -- the ladder of resolutions,
<code>N_l = N_min * b^l</code>. Watch the ladder table: two levels that land on the same
lattice are wasted, since they are separate feature sets on identical nodes.</li>
<li><b>features per level F</b> -- width per scale. Doubling it doubles the table.</li>
<li><b>log2 table size T</b> -- where collisions begin. A level whose nodes fit in T is
stored densely and never collides; the ladder marks which are which.</li>
<li><b>coarsest / finest cells per axis</b> -- the ends of the ladder. Finest at 0 means
the image's own pixel count. Going finer than the pixel spacing buys almost no PSNR,
costs parameters, and shows up as noise in any derivative of the fit.</li>
<li><b>decoder width / hidden layers</b> -- the MLP that turns the concatenated features
into RGB. Small on purpose: the table is meant to carry the signal.</li>
<li><b>interpolation</b> -- linear is the paper's default and only C0. Smoothstep
(3w^2-2w^3) makes the encoding C1, which you need if anything downstream takes a second
derivative.</li>
<li><b>decoder activation</b> -- relu matches the paper; gelu and softplus are smooth,
which matters for the same reason as smoothstep.</li>
<li><b>loss</b> -- relative L2 is instant-NGP's, and weights dark pixels up by dividing
by the prediction's own magnitude. Plain l2 does not.</li>
<li><b>downsample</b> -- 1, 2 or 4. It changes the reference the compression figure is
measured against, so the same encoder reads four times larger at downsample 2.</li>
</ul>

<h2>The panels</h2>
<ul>
<li><b>reference / fit / absolute error</b> -- error is on a fixed 0-0.1 scale, so it
darkens as the fit improves rather than rescaling itself.</li>
<li><b>finest level contributing</b> -- per 64 px block, the finest level whose
contribution clears 8% of that block's strongest. It separates regions that carry signal
from regions that do not. It is <i>not</i> a map of local spatial scale: the levels do
not specialise by frequency, which is measured in
<code>tests/level_specialisation.py</code>.</li>
<li><b>psnr against training time</b> -- finished runs stay, colour-keyed to the table
below, so settings compare on quality against time and parameters.</li>
<li><b>magnifier</b> -- hover to magnify; with <i>reference fixed</i> the first panel
stays whole and marks the region the others show. Scroll changes the factor.</li>
</ul>
"""

INTERFACE_REG = INTERFACE_COMMON + """
<h2>The problem</h2>
<ul>
<li><b>deformation</b> -- the analytic warp applied to the source to make the target, so
the error of a fit is measured against a known field. <code>global smooth</code> is a few
low Fourier modes; <code>local bending</code> is compact Gaussian bumps;
<code>multiscale</code> runs four vertical bands from 132 px to 23 px features at equal
amplitude; <code>slip band</code> is a 12 px shear the smooth parameterisations cannot
represent.</li>
<li><b>mismatch</b> -- how unlike the two "modalities" are. <code>matched</code> is
identical intensities and an L2 loss. <code>gamma noise</code> applies a gamma remap plus
Poisson-Gaussian noise and switches the loss to patch LNCC, because matching intensities
no longer means matching tissue.</li>
<li><b>parameterisation</b> -- what represents the displacement. <code>ngp</code> is the
hash grid capped at the deformation's own scale; <code>ngp fine</code> is the uncapped
version kept for comparison; <code>tensor N</code> is a dense NxN control grid,
bilinearly interpolated, which is the classical parameterisation.</li>
<li><b>image pyramid</b> -- blurs both images from sigma 16 down to 0 over the first 60%
of the run. Registration by intensity is a local search, and without this the loss has a
capture radius of a few pixels against displacements of 12-42: neither model converges.
The dashed lines on the curve mark the switches.</li>
</ul>

<h2>The model</h2>
<ul>
<li><b>levels</b> and <b>finest cells per axis</b> -- the ladder. The cap is the important
one: set it at the finest structure the deformation contains, not at the image
resolution. At 128 cells here that is 7 px per cell against a 12-23 px finest feature;
uncapped at 512 the fit is no better, uses 34x the parameters, and the field is 64x
rougher.</li>
<li><b>interpolation</b> -- smoothstep, because the folding penalty differentiates the
Jacobian and a linear interpolant's second derivative is identically zero.</li>
<li><b>coarse to fine (level window)</b> -- <i>on</i> starts with 4 of the levels live
and ramps to all of them by half-way, multiplying each level's features by
<code>clamp(alpha - l, 0, 1)</code>; the stats line shows the count as it climbs. It stops
the optimiser fitting fine detail into a misaligned pose. Without an image pyramid it is
worth 43x in endpoint error (0.115 px against 4.950, which also folds); with one it is
redundant, because the pyramid supplies the same thing through the data.</li>
<li><b>control points per axis</b> -- the control grid's only knob, for the
<code>tensor</code> models.</li>
<li><b>smoothness weight</b> -- penalises the field's first derivative.
<b>folding penalty weight</b> -- penalises a Jacobian determinant heading through zero,
which is a warp turning itself inside out. Both are set per loss kind, because the L2 and
LNCC data terms differ by ~1000x and one absolute weight would leave the cross-modal arm
effectively unregularised.</li>
<li><b>displacement scale</b> -- the pixel scale the model's raw output is multiplied by.
Set it above the largest displacement you expect.</li>
<li><b>overlay grid spacing</b> -- cosmetic; the spacing of the warped grid drawn in the
ground-truth-vs-fit panel.</li>
</ul>

<h2>The panels</h2>
<ul>
<li><b>endpoint error</b> -- per pixel, the distance between the displacement recovered
and the true one, in pixels, on a fixed 0-10 scale.</li>
<li><b>grid</b> -- a regular grid carried through the warp: ground truth in red, the fit
in blue dashes. Where they coincide, the field is right.</li>
<li><b>finest level contributing</b> -- as on the image page, and with the same caveat:
signal versus none, not local scale.</li>
<li><b>the curve</b> -- endpoint error by region only. The training loss is deliberately
not plotted: it is measured against whichever pyramid level is current, so it jumps two
orders of magnitude at a switch while the fit is improving.</li>
<li>The <b>background</b> curve sits near the ground-truth displacement magnitude and
stays there. Nothing constrains the warp in a region with no image content, so that line
reports how much warp exists where no data can reach it.</li>
</ul>
"""
