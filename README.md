# ngp-demo — a differentiable multiresolution hash encoding, in plain PyTorch

An instant-NGP encoder (Müller et al. 2022) written with nothing but autograd ops:
no `tinycudann`, no CUDA extension, no build step. It fits a 3.9 Mpixel painting in
under a minute on one A6000, fits a time-evolving 2D field and is scored at times it
never saw, and then hands you the derivatives of the fit with respect to its own
input coordinates — verified against ground truth rather than asserted.

Everything below was measured by the scripts in this repo, on two RTX A6000s,
torch 2.9.0+cu128.

```
ngp/hashgrid.py   MultiResHashGrid  — the encoder
ngp/model.py      NGPField = encoder + MLP, plus jacobian() / laplacian() helpers
ngp/fields.py     AdvDiffField — analytic ground truth for the (x, y, t) stage
scripts/          fit_image.py, fit_field.py, compare_time.py, demo_gradients.py
tests/            9 checks, including gradcheck w.r.t. coordinates and table
```

## Try it in the browser

Two pages, each a single self-contained server. They are the fastest way to see
what any setting does, because the panels update while the fit runs.

```bash
python scripts/gui_image.py     # http://localhost:8022  -- fit the painting
python scripts/gui.py           # http://localhost:8021  -- recover a known warp
```

`gui_image.py` puts every knob of the encoding on a slider and shows what that
setting *builds* before you spend a minute training it: the resolution ladder
level by level, which levels have to hash, the parameter count against the
image's own value count. Finished runs stay on the curve, so settings compare
directly on quality against time and parameters.

`gui.py` warps the painting by a known analytic field and asks a hash grid or a
control grid to recover it, scoring the *field* rather than the pixels. It
starts fitting the default configuration as soon as you open it.

Both carry a **what is an ngp?** button explaining the method and a **what is
this interface?** button explaining every control on that page. Each prints
`[run]`, `[images]` and `[done]` to the terminal, so an empty page can be told
apart from a fit that never started.

## Run it

```bash
python scripts/fit_image.py                                   # stage 1, ~55 s
python scripts/fit_field.py --interpolation smoothstep_xy     # stage 2, ~80 s
python scripts/fit_field.py --isotropic                       # the control
python scripts/compare_time.py                                # stage 2 figure
python scripts/demo_gradients.py                              # derivatives of a fit
python tests/test_ngp.py                                      # 9 passed
python tests/level_specialisation.py                          # the negative result
python tests/check_pages.py                                   # both GUI pages, needs node
```

Outputs (images, mp4s, figures, `report.json`) land in `out/`.

## An application: the zapbench flow field

```bash
pip install tensorstore                  # only this script needs it
python scripts/zapbench_view.py          # http://localhost:8023
```

Reads `gs://zapbench-release` anonymously and shows the flow field of a
light-sheet zebrafish run in 3D: 3 x 36 x 83 x 128 x 7879 displacements, drawn
only where the segmentation says a cell is. Drag to rotate, slider to travel
through the run.

Two measurements from that volume, both made here:

* **21.6%** of the flow voxels contain a segmented cell. The other 78% is
  background where the flow is extrapolated, which is the case for sampling a
  representation rather than storing a dense grid.
* Inside the mask and across the whole 7,879-frame run, the time-varying field
  is **two spatial modes to 0.56 voxels RMS** (97.9% of the energy), three to
  0.43, and flat after that -- 40 modes only reach 0.158. Mean displacement in
  the mask is 11.7 voxels. So a dense 4D grid stores ~250,000 numbers per frame
  to carry a few numbers' worth of new information per frame.

## Install

```bash
conda env create -f envs/environment.linux.yaml     # or environment.mac.yaml
conda activate ngp-demo
python tests/test_ngp.py                            # 9 passed
```

The specs pin only what the repo imports: torch, numpy, pillow, matplotlib,
imageio and pyyaml, plus `imageio-ffmpeg`. That last one is the *Python package*
and not the ffmpeg binary — `imageio`'s mp4 writer falls back silently without
it and produces a TIFF carrying an `.mp4` name.

The Linux spec takes torch 2.9.0 from the CUDA 12.8 wheel index, which is the
build every number in this README was measured on. The macOS spec takes torch
from conda and has no CUDA: the demos run on `mps` or the CPU, so expect the
timings to be slower rather than comparable.

`tests/check_pages.py` additionally needs `node` on PATH; it loads each browser
page's script against a stub DOM, which is not something Python can check. The
other tests do not need it.

If you would rather not use conda:

```bash
pip install torch numpy pillow matplotlib imageio imageio-ffmpeg pyyaml
```

## Stage 1 — an image

`f(x, y) -> RGB`, trained on uniformly random coordinates against a bilinear lookup
into the reference. No hold-out: this is the compression/fit benchmark, not a
generalization test.

| finest level | parameters | PSNR after 2000 steps |
|---|---|---|
| capped at the pixel count, 1808 × 2138 | 5.92 M (51% of the 11.6 M RGB values) | **40.28 dB** |
| uncapped, 7006 cells per axis | 7.66 M | 39.54 dB |

Capping the refinement at the data's own sampling is both smaller *and* better: the
levels below the pixel spacing were spending parameters on structure no sample
constrains, and the cost of that shows up in the derivatives of the fit
rather than in its PSNR.

## Stage 2 — a time-evolving field, scored on times it never saw

`f(x, y, t) -> u`. Ground truth is the closed-form solution of the advection-diffusion
equation `u_t + c·∇u = ν∇²u` on the periodic unit square, so `u` is exact at *any*
`(x, y, t)` and the fit can be scored between the observed frames rather than against
the nearest stored one. Training sees only 17 equally spaced times; the 16 midpoints
between them are held out.

| grid over t | trained times | **unseen midpoints** | mean over dense t | worst t |
|---|---|---|---|---|
| isotropic — t refined like x, y (1557 cells) | 68.62 dB | **23.19 dB** | 32.61 dB | 21.07 dB |
| capped at 16 cells = the frame spacing, linear | 74.18 dB | **47.38 dB** | | |
| capped, smoothstep | 67.47 dB | **47.66 dB** | | |
| capped, smoothstep on x,y + linear on t | 67.26 dB | **48.04 dB** | 52.69 dB | 37.93 dB |

The isotropic grid gets ~92 time cells per frame gap, so each observed frame owns
private parameters and nothing ties them together: it memorises the 17 frames and
collapses to a flat wash in between. `out/time_comparison/psnr_vs_time.png` shows the
error strobing at the frame rate. Capping the time axis at the frame spacing lifts
the worst time from 21 dB to 38 dB. Note the ripple does not vanish — even the capped
fit is ~20 dB better on a trained frame than between two.

## Stage 3 — registration: a hash grid against a control grid

`config/registration_benchmark.yaml` warps the painting by a known analytic field
to make a target, then asks each parameterisation to recover it. Because `u_gt` is
analytic, every run is scored on the **field** it recovered, split three ways:
the textured foreground, the black background where nothing constrains the warp,
and the 24 px band between them. 35 runs; `scripts/summarise.py` merges them.

```bash
python scripts/run_registration.py            # the grid
python scripts/summarise.py                   # merge + figure
python scripts/gui.py                         # or drive it in a browser
```

Matched-intensity arm, endpoint error in pixels (foreground / band / background):

| deformation | ngp | tensor_16 | tensor_256 |
|---|---|---|---|
| global_smooth | **0.108** / 1.51 / 12.53 | 0.408 / **0.97** / **12.29** | 0.157 / 4.95 / 13.36 |
| local_bending | **0.062** / **0.20** / 0.42 | 0.538 / 0.32 / **0.20** | 0.072 / 1.22 / 4.10 |
| global_plus_local | **0.162** / 1.56 / 12.57 | 0.736 / **1.06** / **12.27** | 0.409 / 4.98 / 13.32 |
| slip_band | **0.324** / **0.82** / **3.60** | 1.693 / 3.61 / 5.17 | 0.424 / 7.51 / 16.78 |

* **Where there is texture the hash grid wins**, by 3.8-5.2x over a 16x16 control
  grid on every deformation -- including the shear band, where a C^0 encoding
  represents a near-discontinuity that a bilinear control grid structurally cannot.
* **Where there is no data the coarse grid wins.** `tensor_16` is better in the
  band and background on the smooth warps. Its structural smoothness *is* the
  prior, and in an unconstrained region the prior is the entire answer.
* **`tensor_256` is the worst of both**: competitive in the foreground, 4.9-13.0 px
  in the band. Capacity without locality extrapolates badly exactly at the mask edge.

### Resolve to the data, again -- and it is the Jacobian that pays

| local_bending / matched | params | EPE fg | EPE bg | bending energy |
|---|---|---|---|---|
| finest 512 cells (1.8 px) | 2,356,738 | 0.062 | 0.422 | 2.07e-03 |
| finest 512, hashed at 2^16 | 552,834 | 0.063 | 0.813 | 2.12e-03 |
| **finest 128 cells (7.1 px)** | **69,428** | **0.053** | **0.402** | **3.24e-05** |

34x fewer parameters, slightly better accuracy, and a **64x lower bending energy**.
The deformation's finest feature is ~30 px, so every level below ~7 px/cell was
inventing structure no sample constrains. As in stages 1-3 it costs almost nothing
in the loss and everything in the derivative.

### Coarse-to-fine and an image pyramid are substitutes

| global_plus_local / matched | EPE fg | min det J |
|---|---|---|
| pyramid, level window on | 0.162 | 0.336 |
| pyramid, level window off | **0.111** | 0.330 |
| no pyramid, level window on | **0.115** | 0.206 |
| no pyramid, level window off | **4.950** | **-4.831** |

Either mechanism supplies the coarse-to-fine an intensity loss needs; without
both, the fit lands 43x worse and folds. Running both is mildly counterproductive,
since the level window only delays access to the fine levels. The level window is
the cheaper of the two -- no blurred copies of the volume.

Two bugs this stage caught, both of which would have produced confident wrong
conclusions, are worth naming because they are easy to repeat:

1. A single absolute regulariser weight against two data terms of very different
   magnitude (L2 ~1e-4, 1-LNCC ~3e-1) leaves the cross-modal arm effectively
   unregularised. It looked like "modality mismatch causes folding" (10-23% folded
   Jacobians against 0%); it was a missing weight.
2. Ground-truth displacements of 12-42 px against a 9 px LNCC window give a
   capture radius of ~4 px, so neither model could converge. That looked like a
   model comparison; it was the objective. An image pyramid took the hash grid
   from 11.59 to 1.27 px endpoint error and the control grid from 8.68 to 1.47.

### The levels do not specialise by frequency

A hash grid is often described as putting its fine levels where the fine
structure is. It does not, and `tests/level_specialisation.py` is the standing
check. Fit the encoder by plain regression -- no registration loss, no
regulariser -- to a field that is smooth on its left half and 10x finer on its
right at equal amplitude, and measure each level's contribution on each half:

| cells per axis | left (smooth) | right (fine) | ratio |
|---|---|---|---|
| 8 | 0.0285 | 0.0165 | 0.58 |
| 52 | 0.0378 | 0.0349 | 0.92 |
| 134 | 0.0269 | 0.0286 | 1.06 |
| 344 | 0.0075 | 0.0086 | 1.14 |

The 344-cell level does as much work on the smooth side as on the fine side.
Nothing in the architecture would do otherwise: every level is queried
everywhere, the levels are summed into one feature vector, and a fine level
represents a smooth function perfectly well by varying its entries slowly.

What *is* true is narrower: entries only move where samples touch them, so a
region with no data costs nothing. A level map therefore separates signal from
no-signal -- on the painting, the black surround from the face -- and says
nothing about the local scale of the structure. If capacity belongs at a
particular scale, cap the finest level there. That is what stage 3 measured:
34x fewer parameters, lower endpoint error and 64x less bending energy.

## What this encoder does that tiny-cuda-nn's does not

* Runs anywhere torch runs — no `nvcc`, no build. It is slower than the fused CUDA
  kernel, but 55 s to 40 dB on a 3.9 Mpixel image is not the bottleneck for most uses.
* Gradients flow to the **inputs** as well as the table, and double backward works,
  so `∇f` and `∇²f` are available for PDE residuals, Jacobian penalties or
  derivative supervision.
* `base_resolution`, `per_level_scale`, `max_resolution` and `interpolation` are all
  settable **per axis**. That is what stage 2 needs (time capped at the frame
  spacing), and it is not expressible in a single isotropic config. The same is
  true of `interpolation`: smoothstep in space with linear in time is the
  configuration that both fits best and differentiates correctly.

## The one rule every stage keeps repeating

Resolve to the data, not past it. The image wanted its finest level at the pixel
count; the field wanted its time axis at the frame spacing and its spatial axes at
~20 cells per finest wavelength. Over-refining barely moves PSNR — which is why it is
easy to miss — and shows up instead as sub-scale wiggle in the fit, which is invisible
in the value and fatal in the derivative.
