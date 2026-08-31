"""Shared pieces of the two browser UIs: image encoding and the flat stylesheet.

Both `scripts/gui_field.py` (registration) and `scripts/gui_image.py` (image fitting)
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


# Diverging, through black rather than white: these panels sit on a black page,
# so zero should read as "nothing here" and not as the brightest thing in the
# frame.  Blue negative, red positive, symmetric about zero.
SIGNED_RAMP = LinearSegmentedColormap.from_list(
    "signed", [(0.25, 0.55, 1.00), (0.10, 0.20, 0.45), (0.00, 0.00, 0.00),
               (0.45, 0.12, 0.12), (1.00, 0.30, 0.28)])


def signed_rgb(a: np.ndarray, vmax: float) -> np.ndarray:
    """Signed data on a symmetric +-vmax scale -> uint8 RGB."""
    x = np.clip(a / max(vmax, 1e-6), -1.0, 1.0) * 0.5 + 0.5
    return (SIGNED_RAMP(x)[..., :3] * 255).astype(np.uint8)


def signed_png(a: np.ndarray, vmax: float, max_h: int = DISPLAY_H) -> str:
    return png_data_uri(signed_rgb(a, vmax), max_h)


def cmap_png(a: np.ndarray, vmax: float, name="inferno", max_h: int = DISPLAY_H) -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    cmap = LEVEL_RAMP if name == "levels" else matplotlib.colormaps[name]
    return png_data_uri((cmap(x)[..., :3] * 255).astype(np.uint8), max_h)


def field_png(a: np.ndarray, vmax: float, name="viridis",
              max_h: int = DISPLAY_H) -> str:
    """A SIGNED scalar field on a symmetric [-vmax, vmax], through viridis.

    Symmetric because zero is a meaningful value in a wave and should land in the
    same colour whatever the frame's own extremes are, and fixed because a panel
    that rescales itself each refresh cannot be compared with the one beside it.
    """
    x = np.clip(a / max(vmax, 1e-6), -1.0, 1.0) * 0.5 + 0.5
    return png_data_uri((matplotlib.colormaps[name](x)[..., :3] * 255).astype(np.uint8),
                        max_h)


def flow_png(u: np.ndarray, vmax: float, max_h: int = DISPLAY_H) -> str:
    """A displacement field as the optical-flow colour wheel: hue is direction,
    brightness is magnitude.

    A magnitude map alone is blind to the most interesting cases. A shear band
    displaces two half-planes by equal and opposite amounts, so |u| is uniform
    and the map is flat -- the thing that makes it a shear is entirely in the
    direction, which this shows as two opposed hues meeting at a line.

    u: (H, W, 2) in pixels.
    """
    import colorsys
    ang = (np.arctan2(u[..., 1], u[..., 0]) / (2 * np.pi)) % 1.0
    mag = np.clip(np.linalg.norm(u, axis=-1) / max(vmax, 1e-6), 0, 1)
    hsv = np.stack([ang, np.ones_like(mag), mag], -1).reshape(-1, 3)
    rgb = np.array([colorsys.hsv_to_rgb(*c) for c in hsv]).reshape(*u.shape[:2], 3)
    return png_data_uri((rgb * 255).astype(np.uint8), max_h)


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
  /* Four columns on the same track as a four-panel .row.equal above it: same
     gap, same 1fr columns, so a panel here lines up with the panel above it
     however wide the window is.  A flex row cannot do this -- its free space is
     split after removing ITS OWN gaps, so a row of three lands ~9 px off. */
  .row.grid4 { display:grid; grid-template-columns:repeat(4, 1fr); gap:18px;
               align-items:start; }
  .row.grid4 .panel { min-width:0; }
  .row.grid4 .span2 { grid-column:span 2; }
  .row.grid4 canvas { width:100%; height:auto; }
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


# Both explainers run coarse to specific: the headline, then the mechanism, then
# the detail. A reader who stops after the first paragraph should still have the
# true shape of it.

ABOUT_HTML = """
<button class="close" onclick="closeAbout()">close</button>

<h2>In one sentence</h2>
<p>A coordinate goes in and a value comes out &mdash; a colour, a density, a
displacement &mdash; and the whole trick is that the coordinate is looked up in a
stack of learnable grids before a small network ever sees it.</p>

<h2>The mechanism</h2>
<p><code>L</code> grids are laid over the domain at geometrically growing
resolutions, <code>N_l = N_min * b^l</code>. Every grid node carries <code>F</code>
learnable numbers. A query finds its cell at each level, interpolates that cell's
<code>2^D</code> corners, and the <code>L</code> results are concatenated into one
<code>L*F</code> vector which a small MLP decodes. Gradients reach the grids and the
network alike, so the features are learned rather than designed.</p>
<p>That is the entire method. What makes it fast is that almost all the capacity
sits in the lookup, so the network can be tiny; what makes it fit in memory is the
next section.</p>
<p>M&uuml;ller, Evans, Schied and Keller, <i>Instant Neural Graphics Primitives with a
Multiresolution Hash Encoding</i>, SIGGRAPH 2022
(<a href="https://arxiv.org/abs/2201.05989" target="_blank">arXiv:2201.05989</a>,
<a href="https://github.com/NVlabs/instant-ngp" target="_blank">NVlabs/instant-ngp</a>).</p>

<h2>The table is smaller than the grid, so nodes share entries</h2>
<p>That one sentence is the whole of it. A level of resolution <code>r</code> has
<code>(r+1)^D</code> nodes and each node wants its own feature vector, but the level
is only given a table of <code>T</code> entries. When there are more nodes than
entries, many nodes must share one.</p>
<p>Take the finest level of this page's default encoder. It is 657 cells per axis, so
it has <b>658 x 658 = 432,964 nodes</b>, and its table holds <b>1,024</b> entries.
So on average <b>423 distinct nodes share each entry</b>: they read the same two
numbers, and any update one of them makes lands on all of them.</p>
<p>Which nodes share is decided by a spatial hash of the node's integer
coordinates,</p>
<p><code>h(x) = ( XOR_d  x_d * pi_d )  mod  T</code>, &nbsp;
pi = (1, 2654435761, 805459861, 3674653429)</p>
<p>and the point of hashing is that the sharers come out <i>scattered</i>. The 421
nodes that land on entry 0 of that level have a median separation of 330 px in a
904 px frame &mdash; they are spread across the picture, not clustered where the
damage would be visible as a block.</p>
<p>When a level's nodes <i>do</i> fit in <code>T</code> there is no hash and no
sharing: the level is indexed directly and stores every node separately. That is the
normal state of the coarse levels, and it is why the table size does nothing until a
level outgrows it.</p>
<p>What this buys is that cost stops following resolution. A 512-cell-per-axis level
in 3D has ~10^8 nodes, which nobody stores; its table is still <code>T</code>
entries, so refining a level is free.</p>

<h2>Why sharing is survivable</h2>
<p>Because the entry is not really shared equally &mdash; it is won.</p>
<p>In the backward pass an entry <code>e</code> receives the <i>sum</i> of everything
asked of it:</p>
<p><code>dL/de = SUM over every sampled point whose cell has a corner mapping to e,
of  w * dL/df</code></p>
<p>where <code>w</code> is that corner's interpolation weight and <code>dL/df</code>
is the gradient arriving from the decoder. Three things scale each term:</p>
<ul>
<li><b>how many points land there</b> &mdash; sampling is masked to the foreground, so
an empty region contributes few terms or none.</li>
<li><b>the interpolation weight</b> &mdash; a corner far from a query gets a small
<code>w</code> and barely moves the entry.</li>
<li><b>the residual</b> &mdash; a region the fit already explains sends back almost no
gradient, whatever its sample count.</li>
</ul>
<p>If most of the 423 sharers sit where there is no data, no residual, or no weight,
their terms are ~0 and the sum is dominated by the one node the data actually
constrains. The entry converges to what that node needs and the rest ride along on
it. No importance heuristic decides this and no pass detects it: the optimiser does
it as a side effect of ordinary training.</p>
<p><b>The limit is when that is not true.</b> If every sharer is equally well
constrained &mdash; a fully textured image with a small table, which is exactly this
page's default &mdash; no term dominates, the entry settles on a gradient-weighted
compromise, and the fit pays for it. That is not a bug to be fixed with a cleverer
rule; it is the table being too small for the data, and the fix is a bigger table or
a coarser finest level. It is also why the method works so well on a NeRF or on a
mostly-empty volume: there, most sharers really are in empty space.</p>

<h2>What is free, and what is not</h2>
<p>An adaptive octree has to be built: decide where to subdivide, maintain the
topology as the fit changes, keep gradients consistent across a structure that is
itself moving. A hash encoding skips all of it. Every level is evaluated at every
query point, always, so there is nothing to build and nothing to update
mid-training.</p>
<p>What that buys is narrower than it is usually described. Entries only move where
samples touch them, so a region with no data costs nothing to fit. That is the whole
of the adaptivity.</p>
<p><b>The levels do not specialise by spatial frequency.</b> Fit this encoder by
plain regression to a field that is smooth on its left half and ten times finer on
its right at equal amplitude, and every level contributes about equally to both: the
level with 344 cells per axis scores a fine/smooth ratio of 1.14. Nothing in the
architecture would do otherwise, since all levels are queried everywhere, summed
into one vector, and a fine level represents a smooth function perfectly well by
varying its entries slowly. So a level map separates regions that carry signal from
regions that do not; it does not report local scale. If capacity belongs at a
particular scale, cap the finest level there rather than expecting the hierarchy to
discover it &mdash; on a deformation field that gave 34x fewer parameters, lower
endpoint error and 64x less bending energy.
(<code>tests/level_specialisation.py</code>)</p>

<h2>Measured here, not assumed</h2>
<ul>
<li>Autograd through this pure-PyTorch encoder matches float64 finite differences to
5.6e-10 median relative error.</li>
<li>With linear interpolation the Laplacian of a fit scores relative L2 of 1.011
against analytic truth &mdash; exactly what predicting zero scores, because a
multilinear interpolant's second derivative is identically zero.</li>
<li>Capping the finest level at the image's pixel count gave <i>both</i> fewer
parameters and higher PSNR than leaving it uncapped: 5.92M / 40.28 dB against
7.66M / 39.54 dB.</li>
</ul>
"""


_TRAINING = """
<h2>Training</h2>
<p>Three knobs, and they behave as they do anywhere else.</p>
<ul>
<li><b>learning rate</b> &mdash; Adam's step, log-spaced, cosine-decayed to 3% of its
starting value. The hash table takes 1e-2 comfortably; a dense control grid wants
roughly 5x that, because each of its parameters sees far more of the image.</li>
<li><b>iterations</b> &mdash; how long. Every schedule on the page is expressed as a
fraction of this, so changing it rescales them rather than truncating them.</li>
<li><b>batch size</b> &mdash; sample points per step, drawn 90% inside the foreground
mask and 10% uniformly. This is the knob that makes empty space free: the points are
where the compute goes, so a black surround costs nothing.</li>
</ul>
"""

_TERMINAL = """
<h2>If the page looks wrong</h2>
<p>The terminal prints <code>[run]</code> with the configuration,
<code>[images]</code> when the first frames go out, and <code>[done]</code> or
<code>[stopped]</code> with the final numbers. If those lines are there and the page
is blank, the fault is in the browser and not in the fit; the page also reports its
own exceptions back to the terminal as <code>[client]</code>.</p>
"""

INTERFACE_IMAGE = """
<button class="close" onclick="closeHelp()">close</button>

<h2>What this page does</h2>
<p>It fits the painting &mdash; random pixel coordinates in, RGB out &mdash; and shows
what your encoder settings <i>cost</i> before you spend a minute training them. The
number to watch is the compression figure in the line above the panels: parameters
against the image's own value count, green under 50%, amber under 100%, red once
your "compression" is an expansion.</p>

<h2>The shape of the encoder</h2>
<p>Table 1 of the paper lists five encoding parameters and then says that only two
of them need tuning: the table size <code>T</code> and the finest resolution
<code>N_max</code>. Those are the knobs here, plus <code>L</code>. The rest &mdash;
<code>F</code>&nbsp;=&nbsp;2 features per level, <code>N_min</code>&nbsp;=&nbsp;4
cells, a 64-wide 2-layer decoder, linear interpolation, L2 loss &mdash; are fixed in
the source, and the growth factor <code>b</code> is not a setting at all: equation 3
derives it from the two ends and <code>L</code>, so the ladder always lands exactly
on <code>N_max</code>. The ladder table below the panels shows what they built.</p>
<ul>
<li><b>levels L</b> &mdash; how many rungs between <code>N_min</code> and
<code>N_max</code>. More rungs means a gentler <code>b</code>, and two rungs that
land on the same lattice are wasted: they are separate feature sets on identical
nodes.</li>
<li><b>max entries per level T</b> &mdash; shown as the entry count rather than the
exponent. A level with fewer nodes than this stores every node separately and the
knob does nothing; a level with more folds its nodes through the hash and starts
sharing entries. The line under the ladder gives the comparison directly: how many
nodes the finest level wants against how many it may hold, and how many levels
collide as a result.</li>
<li><b>px per finest cell</b> &mdash; where the ladder stops, said in the unit that
means something on a picture: how many pixels one cell of the finest level covers.
1 is the pixel grid itself and <code>N_max</code> follows from it and the image
size. Going below 1 buys almost no PSNR, costs parameters, and shows up as noise in
any derivative taken through the fit; going up coarsens the fit and is how you see
real cells in the level panel.</li>
</ul>

<h2>The one knob that is not in the paper's table</h2>
<ul>
<li><b>hash index</b> &mdash; <i>xor primes</i> is instant-NGP's spatial hash: the
node's integer coordinates are multiplied by large primes and XORed together, so two
nodes that share a row are in unrelated places, and the paper's whole argument rests
on that ("collisions are pseudo-randomly scattered across space, and statistically
unlikely to occur simultaneously at every level"). <i>raster mod T</i> throws the
shuffle away and indexes by the node's plain raster number modulo T. The collisions
then repeat on a fixed stride instead of scattering, and they line up level after
level. Switch it and watch the error panel.</li>
<li><b>downsample</b> &mdash; 1, 2 or 4. It changes the reference the compression
figure is measured against, so the same encoder reads four times larger at
downsample 2.</li>
</ul>
""" + _TRAINING + """
<h2>Reading the panels</h2>
<ul>
<li><b>reference / fit / absolute error</b> &mdash; error on a fixed 0-0.1 scale, so
it darkens as the fit improves rather than rescaling itself.</li>
<li><b>finest level contributing</b> &mdash; the image is divided into fixed 64 px
<i>analysis blocks</i>; each is coloured by the finest level clearing 8% of that
block's strongest contribution, and drawn with that level's cells <i>if</i> they are
at least 3 screen pixels. At 1 px per finest cell nothing is drawable, so every
block tints and only the colours carry information. Raise <b>px per finest
cell</b> to see real cells. It shows signal versus none, not local scale.</li>
<li><b>the 16 finest levels</b> &mdash; the encoder taken apart while it trains. By
default each tile is what that level <i>adds</i>, signed: blue negative, black zero,
red positive, on a fixed &plusmn;0.1. The first tile is the baseline the differences
start from, and it is not black &mdash; fed an all-zero feature vector the decoder
returns a mid grey, which is why the dark background is corrected downwards by level
after level with nothing to compensate. Baseline plus the differences is the fit.
Each tile is sampled on <i>its own</i> level's grid, one sample per node, and blown
up with nearest-neighbour, so a level with ten cells across the picture reads as ten
blocks rather than as a smooth blur the display invented. <b>decompose</b> opens the
same thing full size, with a view that puts one level through the decoder alone.</li>
<li><b>psnr against training time</b> &mdash; finished runs stay, colour-keyed to the
table beside them, so settings compare on quality against time and parameters.</li>
<li><b>magnifier</b> &mdash; hover to magnify; with <i>reference fixed</i> the first
panel stays whole and marks the region the others show. Scroll changes the
factor.</li>
</ul>
""" + _TERMINAL

INTERFACE_REG = """
<button class="close" onclick="closeHelp()">close</button>

<h2>What this page does</h2>
<p>It warps the painting by a <i>known</i> analytic field to make a target, then asks
a parameterisation to recover that warp from the two images. Because the true
displacement is known everywhere, the score is the <b>endpoint error</b> &mdash; the
distance in pixels between the displacement recovered and the true one &mdash; and
not how well the pixels line up. Those two come apart badly, which is the point.</p>

<h2>The problem you are posing</h2>
<ul>
<li><b>deformation</b> &mdash; which warp to recover. <code>global smooth</code> is a
few low Fourier modes; <code>local bending</code> is compact Gaussian bumps;
<code>multiscale</code> runs four vertical bands from 132 px down to 23 px features
at equal amplitude; <code>slip band</code> is a 12 px shear that a smooth
parameterisation structurally cannot represent.</li>
<li><b>mismatch</b> &mdash; how unlike the two "modalities" are. <code>matched</code>
is identical intensities with an L2 loss. <code>gamma noise</code> applies a gamma
remap plus Poisson-Gaussian noise and switches the loss to patch LNCC, because
matching intensities no longer means matching tissue.</li>
<li><b>parameterisation</b> &mdash; what represents the displacement.
<code>ngp</code> is the hash grid capped at the deformation's own scale;
<code>ngp fine</code> is the uncapped version, kept so the cap counts as a result
rather than an assumption; <code>tensor N</code> is a dense NxN control grid,
bilinearly interpolated &mdash; the classical choice.</li>
</ul>

<h2>The two schedules that make it converge</h2>
<p>Registration by intensity is a <i>local</i> search. The loss only knows how to
improve an alignment that is already close, so both schedules exist to get it
close before letting it be precise.</p>
<ul>
<li><b>image pyramid</b> &mdash; blurs <i>both</i> images and sharpens them over
time: sigma 16, 8, 4, 2, 0 at 0%, 15%, 30%, 45% and 60% of the run, marked as dashed
lines on the curve. Blurring widens every feature, so the capture range at the
coarsest level is set by sigma rather than by the texture. Without it a 9 px LNCC
window has a capture radius of about 4 px against displacements of 12-42, and
<i>neither</i> model converges &mdash; measured, 14.2 px endpoint error against 1.2
with it. Because the loss is computed against the currently blurred pair, it jumps at
every switch; that is the target changing, not the fit failing, which is why the
curve plots endpoint error only.</li>
<li><b>coarse to fine (level window)</b> &mdash; the same idea applied to the model
instead of the data. It starts with 4 of the encoder's levels live and ramps to all
of them by half-way, multiplying each level's features by
<code>clamp(alpha - l, 0, 1)</code>; the stats line shows the count climbing. It
stops the optimiser fitting fine detail into a misaligned pose. The two are
<b>substitutes</b>: without a pyramid the level window is worth 43x in endpoint error
(0.115 px against 4.950, which also folds); with one it is redundant and slightly
worse, since it only delays access to the fine levels.</li>
</ul>

<h2>The model, and what keeps the field sane</h2>
<ul>
<li><b>levels L</b>, <b>max entries per level T</b> and <b>px per finest cell</b>
&mdash; the same three knobs the fitting page uses, and b is derived from them by
equation 3 rather than set. The last one is what matters here: set it at the finest
structure the <i>deformation</i> contains, not at the image resolution. 8 px per cell
is already under a 12-23 px finest feature; at 1 px per cell the fit is no better,
uses 34x the parameters, and the field
is 64x rougher.</li>
<li><b>interpolation</b> &mdash; smoothstep, because the folding penalty
differentiates the Jacobian and a linear interpolant's second derivative is
identically zero.</li>
<li><b>control points per axis</b> &mdash; the control grid's only knob.</li>
<li><b>smoothness weight</b> &mdash; penalises the field's first derivative.
<b>folding penalty weight</b> &mdash; penalises a Jacobian determinant heading
through zero, which is a warp turning itself inside out. Both are set per loss kind,
because the L2 and LNCC data terms differ by ~1000x and one absolute weight left the
cross-modal arm effectively unregularised: 10-23% folded Jacobians that looked like a
property of the mismatch and were a missing weight.</li>
<li><b>displacement scale</b> &mdash; the pixel scale the model's raw output is
multiplied by. Set it above the largest displacement you expect.</li>
<li><b>overlay grid spacing</b> &mdash; cosmetic, the spacing of the warped grid in
the ground-truth-vs-fit panel.</li>
</ul>
""" + _TRAINING + """
<h2>Reading the panels</h2>
<ul>
<li><b>source / target / warped by the fit</b> &mdash; if the third matches the
second, the images agree. That is necessary and not sufficient.</li>
<li><b>finest level contributing</b> &mdash; 64 px analysis blocks coloured by the
finest level doing work in each. Signal versus none, not local scale.</li>
<li><b>grid</b> &mdash; a regular grid carried through the warp, ground truth in red
against the fit in blue dashes. Where they coincide the field is right, which is a
stronger statement than the images matching.</li>
<li><b>endpoint error</b> &mdash; fixed 0-10 px scale.</li>
<li><b>the curve</b> &mdash; endpoint error by region. The <b>background</b> line
sits near the ground-truth displacement magnitude and stays there: nothing constrains
the warp where there is no image content, so it reports how much warp exists out of
reach rather than how good the fit is.</li>
</ul>
""" + _TERMINAL
