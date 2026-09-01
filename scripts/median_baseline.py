#!/usr/bin/env python
"""The obvious baseline: a running median in time, as three mp4s.

    python scripts/median_baseline.py                       # k = 3, 7, 15
    python scripts/median_baseline.py --windows 5 --field "redox midplane"

WHY THIS EXISTS.  The NGP movie shows the flicker leaving the data and landing
in the residual, and the reasonable objection is that a median filter over time
would do the same for nothing.  It half would.  This writes the movies that let
the two be watched side by side, and prints the three numbers that say where
they differ, because the difference is not visible in either movie.

FRAME-TO-FRAME COMPARABLE, deliberately: the same loader, the same frames in the
same order, the same viridis range from `display_range`, the same signed ramp
for the residual at the same 5%-of-span scale, the same panel height, the same
stacking and the same 30 fps as `gui_scalar_time._save_mp4`.  Only the middle
panel changes.  Anything else and a difference in the picture could be a
difference in the rendering.

WHAT A MEDIAN CANNOT DO, and what the numbers below measure.

  * It is not a representation.  It outputs one value per voxel, so it stores
    exactly what it was given: 256 x 664 x 1024 = 174 M values.  The fit stores
    32.5 M parameters and answers at any (x, y, t), including between frames and
    between pixels.  Denoising and compression are different jobs and only one
    of these does both.

  * It has one window for the whole volume.  A calcium transient rises in about
    a second and these frames are 0.914 s apart, so a window wide enough to kill
    the flicker is also wide enough to flatten the event -- everywhere, including
    where the signal is.  The fit has a level ladder and can be smooth in the
    background while staying sharp on a firing cell.

  * Its residual is not only noise.  The check is the lag-1 autocorrelation of
    the residual in time: independent noise gives roughly zero, and a residual
    that has eaten real dynamics gives a clearly positive number.  Printed per
    window, alongside the same quantity for the fit's residual if a fit is
    available, so the claim is settled by a number and not by watching.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ngp.webui import display_range, field_png, png_data_uri, signed_rgb  # noqa: E402
from ngp.utils import render                                          # noqa: E402
import gui_scalar_time as page                                           # noqa: E402

MP4_FPS = page.MP4_FPS


def median_stack(vol, k):
    """Running median over time, edges replicated, one output frame at a time.

    Frame by frame rather than one `unfold`: the unfolded view of 256 x 664 x
    1024 with k = 15 is 10.4 GB of temporaries for a result that is 0.7 GB, and
    this card is shared.  A window of k frames is 41 MB.
    """
    T = vol.shape[0]
    out = torch.empty_like(vol)
    half = k // 2
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        w = vol[lo:hi]
        if w.shape[0] < k:                      # replicate at the ends, so the
            pad_lo = half - (t - lo)            # first and last frames are
            pad_hi = k - w.shape[0] - pad_lo    # filtered rather than skipped
            parts = ([vol[lo:lo + 1].expand(pad_lo, -1, -1)] if pad_lo else []) \
                + [w] + ([vol[hi - 1:hi].expand(pad_hi, -1, -1)] if pad_hi else [])
            w = torch.cat(parts, 0)
        out[t] = w.median(dim=0).values
    return out


def lag1(resid, sample=200_000, seed=0):
    """Lag-1 correlation of the residual along time, over sampled pixels.

    Near zero: the residual is frame-to-frame independent, which is what a
    removed sensor artefact looks like.  Clearly positive: the residual carries
    something that persists across frames, and the only thing that persists is
    the signal -- the filter took it out of the reconstruction.
    """
    T, H, W = resid.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(H * W, generator=g)[:sample].to(resid.device)
    x = resid.reshape(T, H * W)[:, idx]
    x = x - x.mean(0, keepdim=True)
    a, b = x[:-1], x[1:]
    num = (a * b).sum(0)
    den = torch.sqrt((a * a).sum(0) * (b * b).sum(0)).clamp_min(1e-12)
    return float((num / den).mean())


def psnr_db(recon, raw):
    mse = float(((recon - raw) ** 2).mean())
    peak = float(raw.max() - raw.min())
    return 10.0 * np.log10(peak * peak / max(mse, 1e-20))


def bias(raw, recon, span):
    """The offset the reconstruction puts into each pixel's baseline.

    NO GROUND TRUTH IS NEEDED FOR THIS ONE.  Per pixel, average the residual
    over time; a method that only removes zero-mean noise leaves that average at
    zero, and a method that shifts the baseline does not.  Reported as the RMS
    of those per-pixel averages in parts per thousand of the display span, plus
    the whole-volume mean, which says whether the shift has a direction.

    A least-squares fit cannot fail this: its residual is orthogonal to the
    model and the model can represent a constant, so the per-pixel mean is zero
    by construction.  A running median can, and does when the fluctuation it
    filters is skewed -- the median of a skewed distribution is not its mean,
    so every pixel is pulled toward the side the noise is not on.
    """
    m = (recon - raw).mean(0)
    return 1000.0 * float(m.pow(2).mean().sqrt()) / span, 1000.0 * float(m.mean()) / span


def skewness(raw, sample=200_000, seed=0):
    """Skewness of each pixel's fluctuation about its own time-median.

    This is the quantity that decides whether a median filter is biased at all:
    at zero it is an unbiased estimate of the centre, and away from zero the
    median and the mean are different numbers and filtering picks the wrong one.
    """
    T, H, W = raw.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(H * W, generator=g)[:sample].to(raw.device)
    x = raw.reshape(T, -1)[:, idx]
    x = x - x.median(dim=0).values
    s = x.std(0).clamp_min(1e-6)
    return float((x / s).pow(3).mean(0).mean())


def kept_variance(raw, recon, frac=0.01):
    """Temporal variance kept, at the busiest pixels and at the quietest.

    One number per group.  A filter that removes only noise keeps nearly all of
    the variance where the cells are and throws away most of it in the
    background; one that is simply smoothing everything takes the same bite out
    of both, and the first number is how much real signal it cost.
    """
    T, H, W = raw.shape
    v = raw.var(0).reshape(-1)
    n = max(1, int(frac * v.numel()))
    order = v.argsort(descending=True)
    out = []
    for idx in (order[:n], order[-n:]):
        r = raw.reshape(T, -1)[:, idx].var(0).sum()
        f = recon.reshape(T, -1)[:, idx].var(0).sum()
        out.append(float(f / r.clamp_min(1e-12)))
    return out


def write_mp4(raw, recon, path, label, lo, hi, err_max, fps=MP4_FPS):
    """raw on top, the filtered volume under it, the residual below.

    The tiles go through the page's own `field_png` / `signed_rgb` so a frame
    here and the same frame of the NGP movie differ only in what is being shown.
    """
    import imageio.v2 as imageio
    T = raw.shape[0]
    t0 = time.perf_counter()
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as w:
        for i in range(T):
            a, b = raw[i].cpu().numpy(), recon[i].cpu().numpy()
            uris = (field_png(a, 0.0, lo=lo, hi=hi),
                    field_png(b, 0.0, lo=lo, hi=hi),
                    png_data_uri(signed_rgb(b - a, err_max)))
            tiles = [_upscale(np.asarray(Image.open(io.BytesIO(
                base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))) for u in uris]
            h, wd, _ = tiles[0].shape
            bar = 16
            im = Image.fromarray(np.zeros((3 * (h + bar), wd, 3), np.uint8))
            d = ImageDraw.Draw(im)
            for j, (tile, lab) in enumerate(zip(tiles, ("raw", label, "residual"))):
                y = j * (h + bar)
                d.text((4, y + 3), f"{lab}   frame {i + 1}/{T}", fill=(255, 255, 255))
                im.paste(Image.fromarray(tile), (0, y + bar))
            arr = np.asarray(im)
            if arr.shape[0] % 2 or arr.shape[1] % 2:
                arr = arr[:arr.shape[0] // 2 * 2, :arr.shape[1] // 2 * 2]
            w.append_data(arr)
    print(f"[mp4   ] {T} frames at {fps} fps -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)",
          flush=True)


def _upscale(tile, target=440):
    f = max(1, int(round(target / max(1, tile.shape[0]))))
    return tile if f == 1 else np.repeat(np.repeat(tile, f, 0), f, 1)


def write_grid_mp4(panels, labels, path, lo, hi, err_max, residual=None,
                   fps=MP4_FPS):
    """Four reconstructions of the same frame in a 2x2, same colour scale.

    Side by side rather than one movie each, because the argument is a
    comparison and a viewer cannot hold two movies in their head.  `residual`,
    when given, is the volume every panel is differenced against, and the grid
    shows what each method removed instead of what it kept -- which is the panel
    where the methods stop looking alike.
    """
    import imageio.v2 as imageio
    T = panels[0].shape[0]
    t0 = time.perf_counter()
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as w:
        for i in range(T):
            tiles = []
            for v in panels:
                a = v[i].cpu().numpy()
                if residual is None:
                    u = field_png(a, 0.0, lo=lo, hi=hi)
                else:
                    u = png_data_uri(signed_rgb(a - residual[i].cpu().numpy(),
                                                err_max))
                tiles.append(_upscale(np.asarray(Image.open(io.BytesIO(
                    base64.b64decode(u.split(",", 1)[1]))).convert("RGB"))))
            h, wd, _ = tiles[0].shape
            bar, gap = 16, 4
            im = Image.fromarray(np.zeros((2 * (h + bar) + gap, 2 * wd + gap, 3),
                                          np.uint8))
            d = ImageDraw.Draw(im)
            for j, (tile, lab) in enumerate(zip(tiles, labels)):
                x = (j % 2) * (wd + gap)
                y = (j // 2) * (h + bar + gap)
                d.text((x + 4, y + 3), f"{lab}   frame {i + 1}/{T}",
                       fill=(255, 255, 255))
                im.paste(Image.fromarray(tile), (x, y + bar))
            arr = np.asarray(im)
            if arr.shape[0] % 2 or arr.shape[1] % 2:
                arr = arr[:arr.shape[0] // 2 * 2, :arr.shape[1] // 2 * 2]
            w.append_data(arr)
    print(f"[mp4   ] {T} frames at {fps} fps -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)",
          flush=True)


def ngp_recon(cfg_name, vol, device):
    """Fit the same volume with the NGP and render it back frame by frame.

    The point is not the picture -- gui_scalar_time already makes that -- but
    the same three numbers as the median rows, computed the same way on the same
    data, so the comparison is a table and not two movies watched an hour apart.
    """
    import NGP_Main as main_mod
    cfg, _ = main_mod.load_config(cfg_name)
    T, h, w = vol.shape
    model, p = main_mod.build_from_config(cfg, (h, w), T, device)
    opt = main_mod.make_opt(cfg, model, cfg["training"]["lr"])
    steps, batch = cfg["training"]["steps"], cfg["training"]["batch"]
    t0 = time.perf_counter()
    for s in range(steps):
        main_mod.one_step(model, opt, vol, batch, device, T)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    dt = time.perf_counter() - t0
    out = torch.empty_like(vol)
    with torch.no_grad():
        for i in range(T):
            c = page.frame_coords(h, w, i / max(1, T - 1), device)
            out[i] = render(model, c, (h, w))
    n_enc, n_mlp = model.n_parameters()
    return out, dt, n_enc + n_mlp


def corrupt_by(mode, vol, amp, seed=0):
    """One of four artefacts, each of which one method handles and the other does not.

    Run on a video with no artefact of its own -- the bison clip, not the
    recording -- so that everything in the residual was put there by us and the
    truth is genuinely clean.  On zapbench the "truth" already contains the
    stripes being studied, which makes every number ambiguous.

      stripe:<p>  one-sided row stripes, redrawn every frame, present at a pixel
                  in a fraction p of the frames.  p is the whole argument: below
                  0.5 the artefact is a minority of any median window and the
                  filter removes it; above 0.5 it is the majority and the median
                  of the window IS a corrupted value.
      shot:<r>    isolated bright pixels at rate r, the textbook impulse noise a
                  median filter was invented for.
      gauss       zero-mean noise on every pixel of every frame.  Nothing is an
                  outlier, so there is nothing for a median to reject; the right
                  estimator is a mean, and a least-squares fit is one.
      skew        one-sided noise on every pixel, same energy as gauss.  Dense
                  like gauss but asymmetric like shot -- the case where the
                  median is not merely weak but BIASED, because the median of a
                  skewed distribution is not its mean.

    Returns (corrupted, mask) with the mask marking what was struck, so the
    offset can be measured where it was put rather than averaged over a volume
    that is mostly untouched.
    """
    T, H, W = vol.shape
    g = torch.Generator(device=vol.device).manual_seed(seed)
    kind, _, arg = mode.partition(":")
    if kind == "stripe":
        hit = torch.rand(T, H, 1, generator=g, device=vol.device) < float(arg)
        hit = hit.expand(T, H, W)
        return vol + hit.to(vol.dtype) * amp, hit
    if kind == "shot":
        hit = torch.rand(T, H, W, generator=g, device=vol.device) < float(arg)
        return vol + hit.to(vol.dtype) * amp, hit
    if kind == "gauss":
        n = torch.randn(T, H, W, generator=g, device=vol.device) * amp
        return vol + n, torch.ones_like(vol, dtype=torch.bool)
    if kind == "skew":
        # Exponential: one-sided, and scaled to carry the same variance as the
        # gaussian so the two differ in shape and not in how much was added.
        u = torch.rand(T, H, W, generator=g, device=vol.device).clamp_min(1e-9)
        return vol - amp * torch.log(u), torch.ones_like(vol, dtype=torch.bool)
    raise SystemExit(f"unknown mode {mode!r}")


def inject(vol, p, amp, seed=0):
    """Add a known one-sided stripe artefact, so there IS a ground truth.

    The objection to every number above is that the recording is all we have and
    nobody knows what the clean volume was.  So make one: take the recording as
    the truth, add stripes with statistics we choose, and ask which method gives
    the recording back.  The truth is real data with its own real noise; only
    the thing being removed is synthetic, which is exactly the part that has to
    be known.

    The stripes are ROWS, one-sided (a stripe adds signal, it does not take any
    away), and flicker independently every frame, so a pixel is hit in a
    fraction `p` of the frames.  `p` is the knob the whole argument turns on:

      p = 0.2   the artefact is a minority at every pixel -- the case a median
                filter is built for, and it wins
      p = 0.5   the boundary
      p = 0.8   the artefact is the MAJORITY of the window, so the median of the
                window IS a corrupted value; the filter does not merely fail to
                remove the stripe, it adopts it as the baseline

    Returns (corrupted, mask) with the mask marking every (frame, row) that was
    struck, so the offset can be measured where it was actually put.
    """
    T, H, W = vol.shape
    g = torch.Generator(device=vol.device).manual_seed(seed)
    hit = (torch.rand(T, H, 1, generator=g, device=vol.device) < p)
    return vol + hit.to(vol.dtype) * amp, hit.expand(T, H, W)


def fps_for(a, T):
    """Frames per second that makes the file last `--seconds`.

    A movie is watched at whatever length it is, not at whatever rate it was
    written, and two clips of different frame counts compared side by side
    should end together.
    """
    return MP4_FPS if a.seconds <= 0 else max(1.0, T / a.seconds)


def run_modes(vol, a, device, lo, hi, err_max, tag):
    """Every artefact against every method, on a clean video.

    The point is not a winner but a MAP: which corruption each method is right
    for, so the choice stops being a matter of taste.  A row where the median
    beats the fit is as much a result as one where it does not, and a row where
    both are worse than leaving the data alone is the most useful of all.
    """
    amp = a.amp if a.amp > 0 else 2.5 * float((vol[1:] - vol[:-1]).std())
    print(f"\n[modes ] clean truth: {tuple(vol.shape)}, values {float(vol.min()):.3g}"
          f" to {float(vol.max()):.3g}; artefact amplitude {amp:.4g}\n")
    print(f"  {'artefact':>12s} {'method':>12s} {'RMSE vs truth':>14s} "
          f"{'RMSE on stripes':>16s} {'RMSE where clean':>17s} {'vs doing nothing':>17s}")
    for mode in a.modes:
        corrupt, hit = corrupt_by(mode, vol, amp)
        base = float((corrupt - vol).pow(2).mean().sqrt())
        clean_n = int((~hit).sum())
        print(f"  {mode:>12s} {'uncorrected':>12s} {base:14.4f} "
              f"{float((corrupt - vol)[hit].pow(2).mean().sqrt()):16.4f} "
              f"{0.0:17.4f} {'':>17s}")
        recs = [(f"median k={k}", median_stack(corrupt, k)) for k in a.windows]
        if a.ngp:
            rec, _, _ = ngp_recon(a.ngp, corrupt, device)
            recs.append(("ngp fit", rec))
        for label, rec in recs:
            d = rec - vol
            r = float(d.pow(2).mean().sqrt())
            cl = float(d[~hit].pow(2).mean().sqrt()) if clean_n else float("nan")
            print(f"  {'':>12s} {label:>12s} {r:14.4f} "
                  f"{float(d[hit].pow(2).mean().sqrt()):16.4f} {cl:17.4f} "
                  f"{base / max(r, 1e-9):16.2f}x")
        if not a.no_mp4:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            slug = mode.replace(":", "")
            # One scale for every panel, taken from the errors themselves: the
            # page's +-5% of span is the size of a fit's own residual, and these
            # errors are the size of the artefact that was added.
            em = max(1e-9, float(torch.quantile(
                torch.stack([(r - vol).abs().flatten()[::13] for _, r in recs] +
                            [(corrupt - vol).abs().flatten()[::13]]).flatten(), 0.999)))
            names = [l for l, _ in recs]
            best_med = min([r for r in recs if r[0].startswith("median")],
                           key=lambda r: float((r[1] - vol).pow(2).mean()))
            keep = [("truth", vol), ("corrupted", corrupt), best_med,
                    recs[-1] if a.ngp else recs[0]]
            write_grid_mp4([v for _, v in keep], [n for n, _ in keep],
                           os.path.join(a.out, f"{tag}_{slug}_montage_{stamp}.mp4"),
                           lo, hi, em, fps=fps_for(a, vol.shape[0]))
            # THE ERROR GRID READS ACROSS THE TOP. Top-left is the artefact
            # itself -- corrupted minus truth is exactly what was injected, the
            # thing every method is being asked to remove -- and top-right is
            # the fit, so the pair the argument is about sits side by side. The
            # two medians go underneath, shortest and longest window, which is
            # the axis along which a median trades artefact for signal.
            meds = [r for r in recs if r[0].startswith("median")]
            err = [("corrupted", corrupt), recs[-1] if a.ngp else meds[0],
                   meds[0], meds[-1]]
            write_grid_mp4([v for _, v in err[:4]],
                           [f"{n} - truth" for n, _ in err[:4]],
                           os.path.join(a.out, f"{tag}_{slug}_montage_err_{stamp}.mp4"),
                           lo, hi, em, residual=vol, fps=fps_for(a, vol.shape[0]))
        del corrupt, hit, recs
    print("\n  the last column is the error of doing nothing divided by the error"
          "\n  of the method: above 1 it helped, below 1 it made the data worse.\n")


def run_injection(vol, a, device, lo, hi, err_max, tag):
    """The ground-truth experiment: who gives the original volume back?"""
    span = hi - lo
    amp = a.amp if a.amp > 0 else 2.5 * float((vol[1:] - vol[:-1]).std())
    print(f"\n[inject] one-sided row stripes of {amp:.2f} "
          f"({100 * amp / span:.1f}% of the display span), flickering every frame\n")
    print(f"  {'duty':>6s} {'method':>12s} {'RMSE vs truth':>14s} "
          f"{'RMSE on stripes':>20s} {'RMSE where clean':>17s}")
    for p in a.duty:
        corrupt, hit = inject(vol, p, amp)
        base = float((corrupt - vol).pow(2).mean().sqrt())
        print(f"  {p:6.2f} {'uncorrected':>12s} {base:14.3f} "
              f"{float((corrupt - vol)[hit].pow(2).mean().sqrt()):20.3f} {0.0:17.3f}")
        recs = [(f"median k={k}", median_stack(corrupt, k)) for k in a.windows]
        if a.ngp:
            rec, _, _ = ngp_recon(a.ngp, corrupt, device)
            recs.append(("ngp fit", rec))
        for label, rec in recs:
            d = rec - vol
            print(f"  {'':6s} {label:>12s} {float(d.pow(2).mean().sqrt()):14.3f} "
                  f"{float(d[hit].pow(2).mean().sqrt()):20.3f} "
                  f"{float(d[~hit].pow(2).mean().sqrt()):17.3f}")
        # The movies are written for ONE duty cycle, the last asked for, because
        # this is where the argument is decided and four more files of the easy
        # case would not be watched. Every residual panel here is a true error
        # map against the volume the stripes were added to, which is the one
        # thing the movies of the recording itself cannot show.
        if p == a.duty[-1] and not a.no_mp4:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            # ONE SCALE FOR ALL PANELS, SET BY THE ERRORS THEMSELVES. The same
            # ramp everywhere is the whole point -- panels on different scales
            # cannot be compared -- but the page's +-5% of the display span is
            # the scale of the recording's own residuals, and these errors are
            # the size of the stripe that was injected. At +-17.99 with errors
            # of 14.1 every panel sits at 78% of the ramp and reads as solid
            # red. Taking the 99.9th percentile over every method's error puts
            # the worst of them at the end of the ramp and leaves the rest
            # somewhere legible.
            errs = torch.stack([(r - vol).abs().flatten()[::97]
                                for _, r in recs] +
                               [(corrupt - vol).abs().flatten()[::97]])
            err_max = max(1e-6, float(torch.quantile(errs.flatten().float(), 0.999)))
            print(f"  {'':6s} [movies] signed ramp at +-{err_max:.2f} counts, "
                  f"the 99.9th percentile of every method's error", flush=True)
            names = ["corrupted"] + [l for l, _ in recs]
            vols = [corrupt] + [r for _, r in recs]
            for label, rec in zip(names, vols):
                slug = label.replace(" ", "").replace("=", "")
                write_mp4(vol, rec,
                          os.path.join(a.out, f"{tag}_inject{p:.1f}_{slug}_{stamp}.mp4"),
                          f"{label}, stripes at duty {p:.1f}", lo, hi, err_max)
            keep = [0, 1, len(vols) - 2, len(vols) - 1][:4]
            write_grid_mp4([vols[i] for i in keep], [names[i] for i in keep],
                           os.path.join(a.out, f"{tag}_inject{p:.1f}_montage_{stamp}.mp4"),
                           lo, hi, err_max)
            write_grid_mp4([vols[i] for i in keep], [f"{names[i]} - truth" for i in keep],
                           os.path.join(a.out,
                                        f"{tag}_inject{p:.1f}_montage_err_{stamp}.mp4"),
                           lo, hi, err_max, residual=vol)
        del corrupt, hit, recs
    print("\n  the two RMSE columns split the same error by where it happened:"
          "\n  'on stripes' is what is left where the artefact was put, and"
          "\n  'where clean' is the damage done to pixels it never touched --"
          "\n  what the filtering cost, paid everywhere.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="zapbench consecutive")
    ap.add_argument("--down", type=int, default=1)
    ap.add_argument("--windows", type=int, nargs="+", default=[3, 7, 15])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--modes", nargs="+", default=None,
                    help="artefact modes to map, e.g. stripe:0.2 shot:0.02 gauss skew")
    ap.add_argument("--inject", action="store_true",
                    help="the ground-truth experiment instead of the movies")
    ap.add_argument("--duty", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                    help="fraction of frames a pixel is struck by the stripe")
    ap.add_argument("--amp", type=float, default=0.0,
                    help="stripe amplitude; 0 = 2.5x the frame-to-frame std")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="make every mp4 last this long; 0 keeps the page's 30 fps")
    ap.add_argument("--montage", action="store_true",
                    help="one 2x2 mp4 of raw/ngp/two medians, and one of their "
                         "residuals, instead of a movie per window")
    ap.add_argument("--ngp", default=None,
                    help="config name; adds the fit's row to the same table")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    vol, T, h, w, store = page.load_field(a.field, max(1, a.down), device)
    lo, hi = display_range(vol)
    err_max = max(1e-6, 0.05 * (hi - lo))
    os.makedirs(a.out, exist_ok=True)
    tag = a.field.replace(" ", "_")
    print(f"[data  ] {a.field}: {T} frames of {w}x{h} = {T * h * w / 1e6:.1f} M "
          f"values, display range {lo:.4g} to {hi:.4g}, residual scale "
          f"+-{err_max:.4g}", flush=True)

    if a.modes:
        run_modes(vol, a, device, lo, hi, err_max, tag)
        return
    if a.inject:
        run_injection(vol, a, device, lo, hi, err_max, tag)
        return

    if a.montage:
        # TWO MONTAGES, WITH DIFFERENT PANELS, because the two ask different
        # questions. The values grid keeps raw as the reference the eye returns
        # to. The residual grid does NOT: raw differenced against itself is a
        # black square wasting a quarter of the frame, so that panel goes to the
        # third window and the grid becomes what it should be -- every median
        # beside the fit, all four showing what they removed on one scale.
        if not a.ngp:
            sys.exit("--montage needs --ngp <config> for the fit panel")
        rec, _, n_par = ngp_recon(a.ngp, vol, device)
        meds = [median_stack(vol, k) for k in a.windows]
        ngp_lab = f"ngp fit ({n_par / 1e6:.1f}M params)"
        med_lab = [f"median k={k} in time" for k in a.windows]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        write_grid_mp4([vol, rec] + meds[:2], ["raw", ngp_lab] + med_lab[:2],
                       os.path.join(a.out, f"{tag}_montage_{stamp}.mp4"),
                       lo, hi, err_max)
        write_grid_mp4(meds[:3] + [rec],
                       [l + " -- removed" for l in med_lab[:3]] +
                       [ngp_lab + " -- removed"],
                       os.path.join(a.out, f"{tag}_montage_resid_{stamp}.mp4"),
                       lo, hi, err_max, residual=vol)
        return

    print(f"[skew  ] pixel fluctuation about its own median: {skewness(vol):+.3f} "
          f"(0 = the median and the mean agree, so a median filter is unbiased)",
          flush=True)
    rows = []
    for k in a.windows:
        t0 = time.perf_counter()
        med = median_stack(vol, k)
        dt = time.perf_counter() - t0
        r = med - vol
        keep_hi, _ = kept_variance(vol, med)
        b_rms, b_mean = bias(vol, med, hi - lo)
        rows.append((k, psnr_db(med, vol), lag1(r), keep_hi, b_rms, b_mean, dt))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if not a.no_mp4:
            write_mp4(vol, med, os.path.join(a.out, f"{tag}_median{k}_{stamp}.mp4"),
                      f"median k={k} in time", lo, hi, err_max)
        del med, r

    if a.ngp:
        rec, dt, n_par = ngp_recon(a.ngp, vol, device)
        r = rec - vol
        kh, _ = kept_variance(vol, rec)
        b_rms, b_mean = bias(vol, rec, hi - lo)
        rows.append((f"ngp {n_par / 1e6:.1f}M", psnr_db(rec, vol), lag1(r), kh,
                     b_rms, b_mean, dt))
        del rec, r

    print(f"\n  {'method':>9s} {'PSNR vs raw':>12s} {'resid lag-1':>12s} "
          f"{'var kept, busy 1%':>18s} {'baseline shift':>15s} {'signed':>8s} "
          f"{'seconds':>8s}")
    for k, ps, l1, kh, br, bm, dt in rows:
        lab = k if isinstance(k, str) else f"k={k}"
        print(f"  {lab:<9s} {ps:12.2f} {l1:12.3f} {kh:18.3f} {br:15.3f} "
              f"{bm:+8.3f} {dt:8.1f}")
    print("\n  baseline shift: RMS over pixels of each pixel's mean residual, in"
          "\n  parts per thousand of the display span; signed: the same averaged"
          "\n  over the volume, so a number away from zero is a DC offset the"
          "\n  method injected into the recovered signal."
          "\n  lag-1 near 0 = the residual is frame-to-frame independent, which is"
          "\n  what a removed artefact looks like; clearly positive = the filter"
          "\n  also took signal.  'var kept, busy 1%' is the temporal variance left"
          "\n  at the most active pixels: 1.0 means the cells came through"
          "\n  untouched, and every point below that is real dynamics removed.\n")


if __name__ == "__main__":
    main()
