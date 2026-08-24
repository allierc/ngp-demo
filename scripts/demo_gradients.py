#!/usr/bin/env python
"""What "differentiable" buys you: derivatives of the fit w.r.t. its own coordinates.

    python scripts/demo_gradients.py

Reads the checkpoints written by fit_image.py and fit_field.py and checks the
derivatives autograd produces against something that knows the answer:

  A. image      d(RGB)/dx, dy  vs central differences of the same model, and vs
                central differences of the reference pixels.  The first is a
                check that the autograd path through the hash grid is right;
                the second says the model learned the image's gradients, not
                just its values.
  B. field      du/dt and the Laplacian vs the analytic field, and the residual
                of the advection-diffusion equation the field satisfies
                exactly.  Run for both interpolation modes: with linear
                weights the encoding's second derivative is zero everywhere, so
                the Laplacian is missing its dominant term.

None of this needs a training step -- it is all one backward (or two) through
the frozen fit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import AdvDiffField, load_checkpoint
from ngp.fields import _unit_grid
from ngp.utils import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-ckpt", default=os.path.join(ROOT, "out/image/model.pt"))
    p.add_argument("--field-ckpts", nargs="*", default=[
        os.path.join(ROOT, "out/field_smoothstep_xy/model.pt"),
        os.path.join(ROOT, "out/field_smoothstep/model.pt"),
        os.path.join(ROOT, "out/field_linear/model.pt")])
    p.add_argument("--out", default=os.path.join(ROOT, "out/gradients"))
    p.add_argument("--crop", type=float, nargs=4, default=[0.36, 0.30, 0.64, 0.58],
                   metavar=("X0", "Y0", "X1", "Y1"), help="image crop in [0,1]^2")
    p.add_argument("--field-res", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||."""
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


# ----------------------------------------------------------------- A. image


def _gaussian_blur(v, shape, sigma, device):
    """Blur a (N, 2) vector field laid out as `shape` by an isotropic Gaussian."""
    h, w = shape
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = k / k.sum()
    im = v.reshape(h, w, 2).permute(2, 0, 1)[:, None]
    im = torch.nn.functional.conv2d(im, k.view(1, 1, 1, -1), padding=(0, r))
    im = torch.nn.functional.conv2d(im, k.view(1, 1, -1, 1), padding=(r, 0))
    return im[:, 0].permute(1, 2, 0).reshape(-1, 2)


def image_gradients(args, device):
    model, ck = load_checkpoint(args.image_ckpt, device)
    ref = read_image(ck["image"])
    H, W = ref.shape[:2]
    x0, y0, x1, y1 = args.crop
    # Sample at exact pixel centres of the crop: the reference gradient only
    # exists on the pixel lattice, and sampling off it aliases the comparison.
    ix = torch.arange(int(x0 * W), int(x1 * W), device=device)
    iy = torch.arange(int(y0 * H), int(y1 * H), device=device)
    nw, nh = len(ix), len(iy)
    xv, yv = torch.meshgrid((ix + 0.5) / W, (iy + 0.5) / H, indexing="xy")
    xy = torch.stack((xv.reshape(-1), yv.reshape(-1)), dim=1)

    lum = lambda a: model(a).mean(-1, keepdim=True)  # noqa: E731

    a = xy.clone().requires_grad_(True)
    (grad_auto,) = torch.autograd.grad(lum(a).sum(), a)          # (N, 2) d/dx, d/dy

    # Central differences of the *same* model, in float64 so the step can be far
    # inside one grid cell without cancellation.  In float32 the only steps big
    # enough to survive rounding are also big enough to average the slope over
    # several cells and over the ReLU kinks of the decoder, which compares two
    # different quantities and reports a 30% "error" that is not one.
    fine = model.encoding.resolutions[-1]
    model64 = copy.deepcopy(model).double()
    lum64 = lambda a: model64(a).mean(-1, keepdim=True)  # noqa: E731
    xy64 = xy.double()
    h = 1e-4 / max(fine)

    a64 = xy64.clone().requires_grad_(True)
    (grad_auto64,) = torch.autograd.grad(lum64(a64).sum(), a64)

    with torch.no_grad():
        ex = torch.tensor([h, 0.0], device=device, dtype=torch.float64)
        ey = torch.tensor([0.0, h], device=device, dtype=torch.float64)
        fd = torch.cat([(lum64(xy64 + ex) - lum64(xy64 - ex)) / (2 * h),
                        (lum64(xy64 + ey) - lum64(xy64 - ey)) / (2 * h)], dim=1)
        value = lum(xy).reshape(nh, nw)
        # The fit is only piecewise smooth: where a stencil crosses a cell edge
        # or a ReLU boundary no finite difference can match the derivative, so
        # the median is the statistic that answers "is the autograd path right".
        per_point = ((grad_auto64 - fd).norm(dim=1)
                     / grad_auto64.norm(dim=1).clamp(min=1e-12))

    # Central differences of the reference pixels, on the same lattice.
    ref_t = torch.from_numpy(ref).to(device).mean(-1)
    PX, PY = torch.meshgrid(ix.clamp(1, W - 2), iy.clamp(1, H - 2), indexing="xy")
    ref_fd = torch.stack([(ref_t[PY, PX + 1] - ref_t[PY, PX - 1]) / (2.0 / W),
                          (ref_t[PY + 1, PX] - ref_t[PY - 1, PX]) / (2.0 / H)],
                         dim=-1).reshape(-1, 2)
    ref_crop = ref_t[PY, PX].reshape(nh, nw)

    cos = lambda u, v: torch.nn.functional.cosine_similarity(  # noqa: E731
        u, v, dim=1).mean().item()
    stats = {
        "autograd_vs_model_finite_difference_median_rel": per_point.median().item(),
        "float32_vs_float64_autograd_rel_l2": rel_l2(grad_auto.double(), grad_auto64),
        "finite_difference_stencils_across_a_kink": (per_point > 1e-6).float().mean().item(),
        "finite_difference_step": h,
        "crop_pixels": nh * nw,
        "crop": args.crop,
    }
    # Against the reference gradient, at the pixel scale and after low-passing:
    # the fit only reproduces the image's derivative above the scale its samples
    # constrain, and the single-pixel scale is below it.
    for sigma in (0.0, 1.0, 2.0):
        a_, b_ = ((grad_auto, ref_fd) if sigma == 0 else
                  (_gaussian_blur(grad_auto, (nh, nw), sigma, device),
                   _gaussian_blur(ref_fd, (nh, nw), sigma, device)))
        key = "pixel_scale" if sigma == 0 else f"blurred_{sigma:g}px"
        stats[f"vs_reference_{key}_rel_l2"] = rel_l2(a_, b_)
        stats[f"vs_reference_{key}_cosine"] = cos(a_, b_)

    mags = {k: v.float().norm(dim=1).reshape(nh, nw).detach().cpu().numpy()
            for k, v in (("autograd", grad_auto), ("model_fd", fd), ("reference_fd", ref_fd))}
    return value.detach().cpu().numpy(), ref_crop.cpu().numpy(), mags, stats


def image_figure(value, ref_crop, mags, stats, out):
    vmax = float(np.percentile(mags["reference_fd"], 99))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = [(ref_crop, "gray", (0, 1), "a", "reference"),
              (mags["autograd"], "magma", (0, vmax), "b", r"$|\nabla f|$ autograd"),
              (mags["model_fd"], "magma", (0, vmax), "c", r"$|\nabla f|$ finite diff."),
              (mags["reference_fd"], "magma", (0, vmax), "d", r"$|\nabla$ image$|$")]
    for a, (img, cmap, lim, tag, note) in zip(ax, panels):
        im = a.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
        a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.98, tag, transform=a.transAxes, va="top", ha="left",
               fontsize=16, fontweight="bold", color="w")
        a.text(0.98, 0.02, note, transform=a.transAxes, va="bottom", ha="right",
               fontsize=11, color="w")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "image_gradients.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- B. field


def field_derivatives(path, args, device):
    model, ck = load_checkpoint(path, device)
    field = AdvDiffField(device=device, **ck["field_kwargs"])
    n_train = ck["args"]["n_train_frames"]
    # A time strictly between two observed frames.
    t = float((0.5 / (n_train - 1)) + (n_train // 2 - 1) / (n_train - 1))

    res = args.field_res
    xy = _unit_grid(res, device)
    xyt = torch.cat([xy, torch.full_like(xy[:, :1], t)], dim=1)

    a = xyt.clone().requires_grad_(True)
    u = model(a)
    (g1,) = torch.autograd.grad(u.sum(), a, create_graph=True)
    lap = torch.zeros_like(u)
    for d in (0, 1):
        (gg,) = torch.autograd.grad(g1[:, d].sum(), a, create_graph=True)
        lap = lap + gg[:, d : d + 1]
    g1 = g1.detach()
    lap = lap.detach()

    g_true = field.grad(xyt)
    lap_true = field.laplacian(xyt)
    c = field.c

    resid = g1[:, 2:3] + (g1[:, :2] * c).sum(1, keepdim=True) - field.nu * lap
    resid_true = (g_true[:, 2:3] + (g_true[:, :2] * c).sum(1, keepdim=True)
                  - field.nu * lap_true)
    scale = g_true[:, 2:3].abs().mean()          # typical size of a term in the PDE

    stats = {
        "interpolation": (ck["model_kwargs"]["interpolation"]
                          if isinstance(ck["model_kwargs"]["interpolation"], str)
                          else "/".join(ck["model_kwargs"]["interpolation"])),
        "activation": ck["model_kwargs"]["activation"],
        "t": t,
        "rel_l2_value": rel_l2(model(xyt).detach(), field(xyt)),
        "rel_l2_grad_xy": rel_l2(g1[:, :2], g_true[:, :2]),
        "rel_l2_du_dt": rel_l2(g1[:, 2:3], g_true[:, 2:3]),
        "rel_l2_laplacian": rel_l2(lap, lap_true),
        "pde_residual_over_term_scale": (resid.abs().mean() / scale).item(),
        "pde_residual_analytic_over_term_scale": (resid_true.abs().mean() / scale).item(),
    }
    maps = {
        "du_dt": g1[:, 2].reshape(res, res).cpu().numpy(),
        "laplacian": lap.reshape(res, res).cpu().numpy(),
    }
    truth = {
        "du_dt": g_true[:, 2].reshape(res, res).cpu().numpy(),
        "laplacian": lap_true.reshape(res, res).cpu().numpy(),
    }
    return maps, truth, stats


def field_figure(results, out):
    """results: list of (maps, truth, stats), first entry supplies the truth panels."""
    (_, truth, _) = results[0]
    rows = ["du_dt", "laplacian"]
    labels = {"du_dt": r"$\partial u/\partial t$", "laplacian": r"$\nabla^2 u$"}
    ncol = 1 + len(results)
    fig, ax = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.6))
    tags = iter("abcdefghij")
    for i, row in enumerate(rows):
        lim = float(np.percentile(np.abs(truth[row]), 99))
        cols = [(truth[row], "analytic")] + [
            (m[row], f"{s['interpolation']}") for m, _, s in results]
        for j, (img, note) in enumerate(cols):
            a = ax[i, j]
            im = a.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            a.text(0.02, 0.98, next(tags), transform=a.transAxes, va="top", ha="left",
                   fontsize=16, fontweight="bold")
            a.text(0.98, 0.02, f"{labels[row]}  {note}", transform=a.transAxes,
                   va="bottom", ha="right", fontsize=12)
            fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "field_derivatives.png"), dpi=130)
    plt.close(fig)


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    report = {}

    if os.path.exists(args.image_ckpt):
        value, ref_crop, mags, stats = image_gradients(args, device)
        image_figure(value, ref_crop, mags, stats, args.out)
        report["image"] = stats
        print("A. image gradients (crop "
              f"{args.crop}, {stats['crop_pixels']:,} points)")
        print(f"   autograd vs finite differences of the model : "
              f"{stats['autograd_vs_model_finite_difference_median_rel']:.1e} median "
              f"relative error (float64, step {stats['finite_difference_step']:.1e})")
        print(f"   the remaining {stats['finite_difference_stencils_across_a_kink']*100:.1f}% "
              "of points sit on a kink -- a cell edge or a ReLU boundary -- where the")
        print("   fit is C^0 and no finite difference can agree with the derivative")
        print(f"   the float32 gradient you get at run time differs from the "
              f"float64 one by {stats['float32_vs_float64_autograd_rel_l2']:.1e} rel L2")
        print("   autograd vs finite differences of the reference image:")
        for key, note in (("pixel_scale", "at the pixel scale"),
                          ("blurred_1px", "low-passed, sigma = 1 px"),
                          ("blurred_2px", "low-passed, sigma = 2 px")):
            print(f"     {note:<26} rel L2 {stats[f'vs_reference_{key}_rel_l2']:.3f}, "
                  f"cos {stats[f'vs_reference_{key}_cosine']:.3f}")
    else:
        print(f"A. skipped, no {args.image_ckpt} (run scripts/fit_image.py first)")

    results = []
    for path in args.field_ckpts:
        if not os.path.exists(path):
            print(f"B. skipping {path} (not found)")
            continue
        results.append(field_derivatives(path, args, device))
    if results:
        field_figure(results, args.out)
        report["field"] = [s for _, _, s in results]
        print(f"\nB. field derivatives at t = {results[0][2]['t']:.4f} "
              "(a time never trained on)")
        hdr = f"   {'interp':<34}{'u':>10}{'grad_xy':>10}{'du/dt':>10}{'lap':>10}{'PDE res':>10}"
        print(hdr)
        for _, _, s in results:
            print(f"   {s['interpolation']:<34}{s['rel_l2_value']:>10.3f}"
                  f"{s['rel_l2_grad_xy']:>10.3f}{s['rel_l2_du_dt']:>10.3f}"
                  f"{s['rel_l2_laplacian']:>10.3f}"
                  f"{s['pde_residual_over_term_scale']:>10.3f}")
        print("   (rel L2 against the analytic field; PDE residual is normalised by "
              "mean|du/dt|,")
        print(f"    which is {results[0][2]['pde_residual_analytic_over_term_scale']:.1e} "
              "for the analytic field itself)")

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
