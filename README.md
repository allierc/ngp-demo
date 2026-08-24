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

## Run it

```bash
python scripts/fit_image.py                                   # stage 1, ~55 s
python scripts/fit_field.py --interpolation smoothstep_xy     # stage 2, ~80 s
python scripts/fit_field.py --isotropic                       # the control
python scripts/compare_time.py                                # stage 2 figure
python scripts/demo_gradients.py                              # stage 3
python tests/test_ngp.py                                      # 9 passed
```

Needs `torch`, `numpy`, `pillow`, `matplotlib`, `imageio`, `imageio-ffmpeg`.
Outputs (images, mp4s, figures, `report.json`) land in `out/`.

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
constrains. Stage 3 shows where that structure was hiding.

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

## Stage 3 — the derivatives

All from `torch.autograd` through the frozen fit, no training.

**The autograd path is exact.** Against central differences of the same model in
float64 with a step 10⁻⁴ of a grid cell: **5.6e-10 median relative error**. The
remaining 0.2% of points sit on a kink — a cell edge or a ReLU boundary — where the
fit is only C⁰ and no finite difference can agree. Worth knowing: the float32
gradient you actually get at run time differs from the float64 one by 5.4e-3.

**Derivatives are only faithful above the scale the samples constrain.** The fitted
image's gradient against the reference image's gradient:

| compared at | rel L2 | cosine |
|---|---|---|
| the pixel scale | 1.247 | 0.367 |
| low-passed, σ = 1 px | 0.755 | 0.627 |
| low-passed, σ = 2 px | 0.572 | **0.675** |

**Second derivatives need care.** At a time never trained on, relative L2 against the
analytic field (0.009 on the value = a 47 dB fit):

| interpolation | u | ∇u (x,y) | ∂u/∂t | ∇²u | PDE residual |
|---|---|---|---|---|---|
| linear | 0.009 | 0.118 | 0.056 | **1.011** | 0.132 |
| smoothstep | 0.007 | 0.119 | 0.197 | 3.160 | 0.321 |
| smoothstep on x,y + linear on t | 0.009 | 0.130 | 0.065 | 3.230 | 0.226 |

Three things to read off that table.

1. **A multilinear interpolant has zero second derivative**, so with `linear` the
   encoding contributes no curvature at all and the Laplacian's relative L2 is 1.011
   — exactly what predicting zero would score. If you regularise with a Laplacian or
   a bending energy through a linear-interpolated hash grid, you are regularising
   almost nothing.
2. **Smoothstep restores real curvature but is not free.** Its weight derivative
   `6w(1-w)` vanishes at every cell boundary and peaks mid-cell, so on an axis whose
   cells line up with your samples it forces the first derivative to zero at the
   sample points and inflates it between them: `∂u/∂t` degrades from 0.056 to 0.197.
   Interpolating smoothstep in space and linear in time fixes it (0.065) and also
   gives the best held-out PSNR of the four configurations.
3. **Even with smoothstep the Laplacian is noise-dominated** (3.2× too large), because
   fit error at the finest cell scale is amplified by 1/Δ². Sweeping the finest
   spatial resolution over 48 / 96 / 192 / 256 cells moves it between 2.7 and 4.2
   without fixing it. First derivatives (12–13% relative error) are usable; second
   derivatives through a hash grid need explicit smoothing or supervision.

## What this encoder does that tiny-cuda-nn's does not

* Runs anywhere torch runs — no `nvcc`, no build. It is slower than the fused CUDA
  kernel, but 55 s to 40 dB on a 3.9 Mpixel image is not the bottleneck for most uses.
* Gradients flow to the **inputs** as well as the table, and double backward works,
  so `∇f` and `∇²f` are available for PDE residuals, Jacobian penalties or
  derivative supervision.
* `base_resolution`, `per_level_scale`, `max_resolution` and `interpolation` are all
  settable **per axis**. That is what stage 2 needs (time capped at the frame
  spacing) and what stage 3 needs (smoothstep in space, linear in time), and neither
  is expressible in a single isotropic config.

## The one rule the three stages keep repeating

Resolve to the data, not past it. The image wanted its finest level at the pixel
count; the field wanted its time axis at the frame spacing and its spatial axes at
~20 cells per finest wavelength. Over-refining barely moves PSNR — which is why it is
easy to miss — and shows up instead as sub-scale wiggle in the fit, which is invisible
in the value and fatal in the derivative.
