#!/usr/bin/env python
"""How the hash grid splits a deformation across its resolution levels.

    python scripts/show_hierarchy.py --deformation global_plus_local

Trains one NGP deformation field on the benchmark, then opens it up three ways:

  1. the cell grid of selected levels drawn over the painting, so the ladder of
     scales can be read against the brushwork;
  2. the field reconstructed from levels 0..k for increasing k, which is what
     the coarse-to-fine schedule releases in order;
  3. the displacement each level adds on its own, as a map -- the direct answer
     to "does it put capacity where the deformation actually is".

Caveat on (3): the decoder is a nonlinear MLP, so a level's "own" contribution
is the *marginal* effect of releasing it given every coarser level, not a term
in a linear decomposition.  That marginal is exactly what the coarse-to-fine
schedule releases, so it is the quantity worth looking at, but it does not sum
to the total independently of order.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp.deform import apply_mismatch, build_deformation, build_model, warp_image
from scripts.run_registration import (_dense_field, _feather, foreground_mask, load_image,
                                      train_one)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config/registration_benchmark.yaml"))
    p.add_argument("--deformation", default="global_plus_local")
    p.add_argument("--mismatch", default="matched")
    p.add_argument("--model", default="ngp")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "out/hierarchy"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps:
        cfg["training"]["steps"] = args.steps
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(cfg.get("seed", 0))

    source = load_image(cfg["image"], device)
    shape = tuple(source.shape[:2])
    fgc = cfg["image"]["foreground"]
    fg = foreground_mask(source, fgc)
    if fgc.get("zero_background"):
        source = source * _feather(fg, fgc.get("feather_px", 9))
    fg_idx = torch.nonzero(fg.reshape(-1), as_tuple=False).squeeze(1)

    built = {}
    for name, spec in {d["name"]: d for d in cfg["deformations"]}.items():
        built[name] = build_deformation(spec, built, shape, device, cfg.get("seed", 0), fg)
    u_gt = built[args.deformation]
    target = warp_image(source, u_gt, shape)
    xspec = {m["name"]: m for m in cfg["modality_mismatch"]}[args.mismatch]
    target = apply_mismatch(target, xspec, cfg.get("seed", 0))

    models = {m["name"]: m for m in cfg["models"]}
    spec = models[args.model]
    cfg["_model"] = spec
    model = build_model(spec, device)
    print(f"training {args.model} on {args.deformation} / {args.mismatch}")
    train_one(model, source, target, fg_idx, shape, cfg, device,
              xspec.get("loss", "l2"))

    enc = model.field.encoding
    levels = cfg["visualization"]["level_hierarchy"]["levels"]
    print("level resolutions (cells per axis):")
    for l in range(enc.n_levels):
        r = enc.resolutions[l]
        mark = " <-" if l in levels else ""
        print(f"  level {l:2d}  {r[0]:5d} x {r[1]:5d}   "
              f"{'dense' if enc.dense[l] else 'hashed'}{mark}")

    cells_figure(source, enc, levels, args.out, cfg)
    partial, per_level = level_fields(model, enc, shape, device)
    partial_figure(source, partial, per_level, levels, enc, args.out, shape, cfg)
    print(f"wrote {args.out}")


def cells_figure(source, enc, levels, out, cfg):
    """The grid of cells of each selected level, drawn over the image."""
    h, w = source.shape[:2]
    img = source.cpu().numpy()
    fig, ax = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 5.4))
    for a, lv in zip(np.atleast_1d(ax), levels):
        rx, ry = enc.resolutions[lv]
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        step_x, step_y = w / rx, h / ry
        # Drawing every cell edge above a few hundred is a grey wash; thin out.
        stride = max(1, int(round(rx / 48)))
        for i in range(0, rx + 1, stride):
            a.axvline(i * step_x, color="tab:cyan", lw=0.4, alpha=0.8)
        for j in range(0, ry + 1, max(1, int(round(ry / 48)))):
            a.axhline(j * step_y, color="tab:cyan", lw=0.4, alpha=0.8)
        a.set_xlim(0, w); a.set_ylim(h, 0)
        a.set_xticks([]); a.set_yticks([])
        note = (f"level {lv}: {rx}x{ry} cells, {step_x:.1f} px"
                + ("" if enc.dense[lv] else ", hashed")
                + (f"\n(every {stride}th edge drawn)" if stride > 1 else ""))
        a.text(0.98, 0.02, note, transform=a.transAxes, va="bottom", ha="right",
               fontsize=10, color="w")
    for i, a in enumerate(np.atleast_1d(ax)):
        a.text(0.02, 0.98, "abcdefgh"[i], transform=a.transAxes, va="top", ha="left",
               fontsize=16, fontweight="bold", color="w")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "level_cells.png"), dpi=cfg["visualization"]["dpi"])
    plt.close(fig)


@torch.no_grad()
def level_fields(model, enc, shape, device):
    """Displacement from levels 0..k for every k, and each level's own contribution."""
    partial = []
    for k in range(enc.n_levels + 1):
        enc.set_level_window(float(k))
        partial.append(_dense_field(model, shape, device).reshape(*shape, 2).cpu())
    enc.set_level_window(float(enc.n_levels))
    per_level = [partial[k + 1] - partial[k] for k in range(enc.n_levels)]
    return partial, per_level


def partial_figure(source, partial, per_level, levels, enc, out, shape, cfg):
    h, w = shape
    n = len(levels)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 9.2))
    vmax = float(np.percentile(partial[-1].norm(dim=-1).numpy(), 99))
    for j, lv in enumerate(levels):
        a = ax[0, j]
        im = a.imshow(partial[lv + 1].norm(dim=-1).numpy(), cmap="viridis",
                      vmin=0, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])
        a.text(0.98, 0.02, f"|u| from levels 0-{lv}", transform=a.transAxes,
               va="bottom", ha="right", fontsize=11, color="w")
        fig.colorbar(im, ax=a, fraction=0.046)

        a = ax[1, j]
        d = per_level[lv].norm(dim=-1).numpy()
        im = a.imshow(d, cmap="magma", vmin=0,
                      vmax=max(1e-6, float(np.percentile(d, 99.5))))
        a.set_xticks([]); a.set_yticks([])
        rx, ry = enc.resolutions[lv]
        a.text(0.98, 0.02, f"added by level {lv} alone ({rx}x{ry} cells)\n"
                           f"mean {d.mean():.3f} px, max {d.max():.2f} px",
               transform=a.transAxes, va="bottom", ha="right", fontsize=10, color="w")
        fig.colorbar(im, ax=a, fraction=0.046)
    for i, a in enumerate(ax.reshape(-1)):
        a.text(0.02, 0.98, "abcdefghijklmnop"[i], transform=a.transAxes, va="top",
               ha="left", fontsize=16, fontweight="bold", color="w")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "level_contributions.png"),
                dpi=cfg["visualization"]["dpi"])
    plt.close(fig)

    # How much displacement each level carries, over all levels.
    mags = [float(p.norm(dim=-1).mean()) for p in per_level]
    fig, a = plt.subplots(figsize=(8, 4.4))
    a.bar(range(len(mags)), mags, color="0.35")
    a.set_xlabel("level", fontsize=13)
    a.set_ylabel("mean |displacement added| (px)", fontsize=13)
    a.set_xticks(range(len(mags)))
    a.set_xticklabels([f"{l}\n{enc.resolutions[l][0]}" for l in range(len(mags))],
                      fontsize=9)
    a.text(0.98, 0.97, "x tick: level index / cells per axis", transform=a.transAxes,
           ha="right", va="top", fontsize=10, color="0.4")
    a.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "level_budget.png"), dpi=cfg["visualization"]["dpi"])
    plt.close(fig)


if __name__ == "__main__":
    main()
