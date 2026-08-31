# ngp-demo — a differentiable multiresolution hash encoding, in plain PyTorch

An instant-NGP encoder (Müller et al. 2022) written with nothing but autograd ops:
no `tinycudann`, no CUDA extension, no build step. It fits a 3.9 Mpixel painting in
one to two minutes on one A6000, fits a time-evolving 2D field and is scored at times it
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

Three pages, each a single self-contained server. They are the fastest way to see
what any setting does, because the panels update while the fit runs.

```bash
python scripts/gui_image.py     # http://localhost:8022  -- fit the painting
python scripts/gui_field.py           # http://localhost:8021  -- recover a known warp
python scripts/gui_time.py      # http://localhost:8024  -- a warp that moves
python scripts/gui_scalar_time.py  # http://localhost:8025  -- a scalar field from a zarr
```

`gui_image.py` puts every knob of the encoding on a slider and shows what that
setting *builds* before you spend a minute training it: the resolution ladder
level by level, which levels have to hash, the parameter count against the
image's own value count. Finished runs stay on the curve, so settings compare
directly on quality against time and parameters. A panel beside the effective-level
map shows the encoder decomposed **while it trains**: a 4x4 montage of the 16 finest
levels, by default `|levels 0..l| - |levels 0..l-1|`, so you watch the coarse levels
lay down blobs and the fine ones sharpen edges. The **decompose** button opens the
same thing full size, with a view that puts one level through the decoder alone.

`gui_field.py` warps the painting by a known analytic field and asks a hash grid or a
control grid to recover it, scoring the *field* rather than the pixels. It
starts fitting the default configuration as soon as you open it.

`gui_time.py` is the same recovery with the warp in motion: one `(x, y, t)`
encoder against a slip band that translates or rotates across the run. Every
panel is triplicated at the first, middle and last frame, so a fit that only
works at one end of the run cannot hide.

`gui_scalar_time.py` fits `f(x, y, t)` straight from a zarr -- written for the
toy2d store in `Plexus/prototype/graphcast`, a coarse slow wave plus a fast
Kuramoto on four discs, which wants opposite settings from one encoder. Measured
on it: the coarse part has **0%** of its energy above 32 cycles and a lag-1
autocorrelation of **0.998**; the fine part has **73.6%** above 32 cycles, 15 px
per cycle inside a disc on 15.4% of the pixels, and a lag-1 autocorrelation of
**0.829** -- past 0.5 after one frame. So it takes 4 px per finest cell and a
time axis **at** the frame spacing, the opposite end of stage 2. The field
selector fits either component alone. Viridis, on a fixed symmetric scale.

All the pages carry a **what is an ngp?** button explaining the method and a **what
is this interface?** button explaining every control on that page. Each prints
`[run]`, `[params]`, `[images]` and `[done]` to the terminal, so an empty page
can be told apart from a fit that never started, and the parameter count is on
the setup line next to the settings that produced it.

## Run it

```bash
python scripts/fit_image.py                                   # stage 1, ~100 s
python scripts/fit_field.py --interpolation smoothstep_xy     # stage 2, ~80 s
python scripts/fit_field.py --isotropic                       # the control
python scripts/compare_time.py                                # stage 2 figure
python scripts/demo_gradients.py                              # derivatives of a fit
python tests/test_ngp.py                                      # 9 passed
python tests/level_specialisation.py                          # the negative result
python tests/check_pages.py                                   # 4 pages, needs node
python scripts/collision_audit.py                             # who wins a collision
```

Outputs (images, mp4s, figures, `report.json`) land in `out/`.

## An application: the zapbench flow field

```bash
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
  is **two spatial modes to 0.56 RMS** (97.9% of the energy), three to 0.43, and
  flat after that -- 40 modes only reach 0.158, against a mean displacement of
  11.7. A rank-3 model of the masked field is 1.0 M numbers against the stored
  9.0e9, or **8,887x smaller**. Reproduce with `python scripts/zapbench_modes.py`,
  and `--consecutive` for the aliasing control.

Those errors are in the flow array's own units, which are **undocumented**:
`raw` and `segmentation` declare 406 x 406 x 4000 nm and `flow_fields` declares
nothing, so a mean of 11.7 is either 4.75 um or 16x that. Calibrating against
the raw data does not work, because `raw` is already aligned -- the measured
shift between frames 7,000 apart is under 0.01 voxel, so no residual motion
remains for the flow field to explain.

## Install

```bash
conda env create -f envs/environment.linux.yaml     # or environment.mac.yaml
conda activate ngp-demo
python tests/test_ngp.py                            # 9 passed
```

The specs pin only what the repo imports: torch, numpy, pillow, matplotlib,
imageio and pyyaml, plus `imageio-ffmpeg` and `tensorstore`. `imageio-ffmpeg` is
the *Python package* and not the ffmpeg binary — `imageio`'s mp4 writer falls
back silently without it and produces a TIFF carrying an `.mp4` name.
`tensorstore` is only read by the two zapbench scripts; drop that line and every
stage still runs, minus the `gs://zapbench-release` reads.

The Linux spec takes torch 2.9.0 from the CUDA 12.8 wheel index, which is the
build every number in this README was measured on. The macOS spec takes torch
from conda and has no CUDA: the demos run on `mps` or the CPU, so expect the
timings to be slower rather than comparable.

`tests/check_pages.py` additionally needs `node` on PATH; it loads each browser
page's script against a stub DOM, which is not something Python can check. The
other tests do not need it.

If you would rather not use conda:

```bash
pip install torch numpy pillow matplotlib imageio imageio-ffmpeg pyyaml tensorstore
```

## Stage 1 — an image

`f(x, y) -> RGB`, trained on uniformly random coordinates against a bilinear lookup
into the reference. No hold-out: this is the compression/fit benchmark, not a
generalization test. 2000 steps, L = 16, F = 2, N_min = 16, T = 2^19:

| finest level | parameters | PSNR after 2000 steps |
|---|---|---|
| capped at the pixel count, 1808 × 2138 (b = 1.40) | 5.92 M (51% of the 11.6 M RGB values) | 33.06 dB |
| uncapped, 2489 cells per axis (b = 1.40) | 5.92 M | 33.03 dB |
| uncapped, 7006 cells per axis (b = 1.50) | 7.66 M | **33.96 dB** |

Capping the refinement at the data's own sampling costs **nothing** against an equally
fast ladder (rows 1 and 2 differ by 0.03 dB and not at all in size, because the table
caps both), and **0.90 dB** against a ladder that climbs faster and spends 29% more
parameters getting past the pixel spacing.

These numbers replace an earlier table that read 40.28 dB against 39.54 and concluded
that capping was smaller *and* better. That conclusion was an artefact: `BilinearImage`
looked the reference up half a pixel off, so the training target was a half-pixel box
blur of the painting and the PSNR was quoted against the blur. On the same fit that
reads 45.91 dB against the blurred target and 31.18 dB against the actual pixels. With
the lookup fixed, over-refining is *mildly useful* for PSNR, and the case for capping
rests on the parameter count and on the derivatives -- stage 3 measures 34x fewer
parameters and 64x less bending energy at equal endpoint error.

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
python scripts/gui_field.py                         # or drive it in a browser
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

## Stage 4 — a warp that moves

`scripts/gui_time.py` puts the stage-3 recovery in motion: one `(x, y, t)` encoder
against a slip band that translates or rotates over the run, with every panel
triplicated at the first, middle and last frame.

```bash
python scripts/gui_time.py                    # http://localhost:8024
```

An 18 px slip band translating 300 px over the run, 400 steps, the time axis
capped at 16 cells:

| frames | px per frame | parameters | EPE mean | worst frame |
|---|---|---|---|---|
| 100 | 3.030 | 1,036,930 | 0.119 | 0.141 |
| 200 | 1.508 | 1,036,930 | 0.116 | 0.162 |
| 500 | 0.601 | 1,036,930 | 0.121 | 0.159 |
| 800 | 0.376 | 1,036,930 | 0.121 | 0.183 |

**8x the frames costs nothing.** The parameter count is set by the cap on the
time axis and not by how many frames were sampled, and the accuracy moves by
0.005 px across the sweep. What the frame count changes is the sampling density
along `t`, which this deformation does not need: the band moves smoothly, so 100
samples of that motion already pin it.

Speed and kind of motion, at 200 frames:

| motion over the run | EPE mean | worst frame |
|---|---|---|
| 30 px translation | 0.078 | 0.102 |
| 300 px translation | 0.116 | 0.162 |
| 1200 px translation | 0.152 | 0.292 |
| **90 deg rotation** | **1.263** | **3.141** |

40x the translation speed costs 2x the endpoint error. Rotation is 11x worse
than any of them, and structurally so: the band's normal turns with it, so the
field at `t` is not a shifted copy of the field at `t=0` and nothing along the
time axis can be reused.

Raising the time axis past the cap does not help either -- 64 cells: 1,638,208
parameters and 0.122 px, against 16 cells: 1,036,930 and 0.116. Asking for 200
time cells returns the same 1,638,208, because at that point the table is full
and the extra resolution only changes which nodes collide.

The two error rows are there to be read together. On the default run the image
residual is essentially black -- mean 1.9/255, under 1% of pixels above a tenth
of its fixed 0-0.1 scale -- while the field still carries 0.116 px of endpoint
error. **The picture matches and the field is still wrong**, which is the whole
reason this benchmark scores the field. The pyramid row shows which level is
finest in each 64 px block: the finest blocks track the band as it travels
(centred at y = 437, 593, 704 over the three frames) and the flat regions
either side are carried by L0-L3.
