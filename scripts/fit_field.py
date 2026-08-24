#!/usr/bin/env python
"""Stage 2: fit a time-evolving 2D field, f(x, y, t), and score the times it never saw.

    python scripts/fit_field.py --interpolation smoothstep --activation gelu

Ground truth is ngp.fields.AdvDiffField, the closed-form solution of
advection-diffusion on the periodic unit square.  The model is trained on
`--n-train-frames` equally spaced snapshots only; the reported "held-out" PSNR
is measured at the midpoints between them, where the analytic field gives an
exact target.  That separation is the point of the stage: a 3D hash grid is
only useful for a movie if it interpolates *between* frames rather than
memorising each one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import AdvDiffField, NGPField, save_checkpoint
from ngp.fields import _unit_grid
from ngp.utils import psnr, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="default: out/field_<interpolation>")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=1 << 17)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-train-frames", type=int, default=17,
                   help="observed times, equally spaced over [0, 1] inclusive")
    p.add_argument("--eval-res", type=int, default=256)
    p.add_argument("--n-modes", type=int, default=48)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--nu", type=float, default=5e-4)
    p.add_argument("--n-levels", type=int, default=14)
    p.add_argument("--n-features", type=int, default=2)
    p.add_argument("--log2-hashmap-size", type=int, default=19)
    p.add_argument("--base-resolution", type=int, default=8, help="cells per axis, level 0")
    p.add_argument("--per-level-scale", type=float, default=1.5)
    p.add_argument("--space-max-resolution", type=int, default=256,
                   help="cap on x/y cells. The field's finest wavelength is "
                        "1/k_max, so ~20 cells across it is plenty; refining past "
                        "that adds structure no sample constrains, which barely "
                        "moves PSNR but wrecks the Laplacian.")
    p.add_argument("--time-base-resolution", type=int, default=2)
    p.add_argument("--time-max-resolution", type=int, default=None,
                   help="cap on time cells; default n_train_frames - 1, i.e. the "
                        "frame spacing. Above that the grid resolves each observed "
                        "frame on its own and stops interpolating between them.")
    p.add_argument("--isotropic", action="store_true",
                   help="treat t exactly like x and y (no cap) -- the naive setting, "
                        "kept so the failure it causes can be reproduced")
    p.add_argument("--interpolation", default="smoothstep",
                   choices=["linear", "smoothstep", "smoothstep_xy"],
                   help="smoothstep_xy = smoothstep on x and y, linear on t: real "
                        "curvature in space without smoothstep's zero-derivative-at-"
                        "the-frame-times artefact on the time axis")
    p.add_argument("--activation", default="gelu", choices=["relu", "gelu", "softplus", "tanh"])
    p.add_argument("--n-neurons", type=int, default=64)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--movie-frames", type=int, default=121)
    p.add_argument("--no-movie", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.time_max_resolution is None:
        a.time_max_resolution = a.n_train_frames - 1
    if a.out is None:
        tag = "isotropic" if a.isotropic else a.interpolation
        a.out = os.path.join(ROOT, f"out/field_{tag}")
    return a


def evaluate(model, field, ts, res, device):
    """PSNR of the fit against the analytic field at each t in `ts`."""
    xy = _unit_grid(res, device)
    out = []
    for t in ts:
        xyt = torch.cat([xy, torch.full_like(xy[:, :1], float(t))], dim=1)
        gt = field(xyt).reshape(res, res)
        pred = render(model, xyt, (res, res))
        out.append(psnr(pred, gt))
    return out


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    field = AdvDiffField(n_modes=args.n_modes, k_max=args.k_max, nu=args.nu,
                         seed=args.seed, device=device)
    train_ts = torch.linspace(0, 1, args.n_train_frames, device=device)
    held_ts = (train_ts[:-1] + train_ts[1:]) / 2
    print(f"field: {args.n_modes} modes up to k={args.k_max}, nu={args.nu}, "
          f"drift {tuple(field.c.tolist())}")
    print(f"training on {len(train_ts)} frames, scoring {len(held_ts)} unseen midpoints")

    if args.isotropic:
        base_res, max_res = args.base_resolution, None
    else:
        b = args.base_resolution
        base_res = (b, b, args.time_base_resolution)
        max_res = (args.space_max_resolution, args.space_max_resolution,
                   args.time_max_resolution)

    model_kwargs = dict(
        n_input_dims=3, n_output_dims=1,
        n_neurons=args.n_neurons, n_hidden_layers=args.n_hidden_layers,
        activation=args.activation, output_activation="none",
        n_levels=args.n_levels, n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=base_res, per_level_scale=args.per_level_scale,
        max_resolution=max_res,
        interpolation=(("smoothstep", "smoothstep", "linear")
                       if args.interpolation == "smoothstep_xy"
                       else args.interpolation),
    )
    model = NGPField(**model_kwargs).to(device)
    n_enc, n_mlp = model.n_parameters()
    print(model.encoding)
    print(f"parameters: {n_enc:,} encoding + {n_mlp:,} decoder")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 30)

    history, t_train = [], 0.0
    for step in range(args.steps + 1):
        t0 = time.perf_counter()
        xy = torch.rand(args.batch, 2, device=device)
        t = train_ts[torch.randint(len(train_ts), (args.batch,), device=device)]
        xyt = torch.cat([xy, t.unsqueeze(1)], dim=1)
        with torch.no_grad():
            gt = field(xyt)
        loss = ((model(xyt) - gt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0

        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            tr = evaluate(model, field, train_ts.tolist(), args.eval_res, device)
            he = evaluate(model, field, held_ts.tolist(), args.eval_res, device)
            history.append({"step": step, "train_s": t_train, "loss": loss.item(),
                            "psnr_train_times": float(np.mean(tr)),
                            "psnr_held_out_times": float(np.mean(he))})
            print(f"step {step:6d}  {t_train:7.2f}s  loss {loss.item():.3e}  "
                  f"psnr train-t {np.mean(tr):6.2f} dB  held-out-t {np.mean(he):6.2f} dB")

    per_train = evaluate(model, field, train_ts.tolist(), args.eval_res, device)
    per_held = evaluate(model, field, held_ts.tolist(), args.eval_res, device)
    save_checkpoint(os.path.join(args.out, "model.pt"), model, model_kwargs,
                    args=vars(args),
                    field_kwargs=dict(n_modes=args.n_modes, k_max=args.k_max,
                                      nu=args.nu, seed=args.seed))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": history, "args": vars(args),
                   "train_ts": train_ts.tolist(), "held_ts": held_ts.tolist(),
                   "psnr_per_train_t": per_train, "psnr_per_held_t": per_held,
                   "n_enc": n_enc, "n_mlp": n_mlp}, f, indent=1)

    summary_figure(model, field, train_ts, held_ts, per_train, per_held, args, device)
    if not args.no_movie:
        write_movie(model, field, args, device)

    print(f"\nmean PSNR: {np.mean(per_train):.2f} dB at the {len(train_ts)} trained times, "
          f"{np.mean(per_held):.2f} dB at the {len(held_ts)} unseen midpoints "
          f"(gap {np.mean(per_train) - np.mean(per_held):.2f} dB)")
    print(f"wrote {args.out}")


def _panels(model, field, t, res, device):
    xy = _unit_grid(res, device)
    xyt = torch.cat([xy, torch.full_like(xy[:, :1], float(t))], dim=1)
    gt = field(xyt).reshape(res, res)
    pred = render(model, xyt, (res, res))
    return gt.cpu().numpy(), pred.cpu().numpy(), (pred - gt).cpu().numpy()


def summary_figure(model, field, train_ts, held_ts, per_train, per_held, args, device):
    t_show = float(held_ts[len(held_ts) // 2])
    gt, pred, err = _panels(model, field, t_show, args.eval_res, device)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, (img, cmap, lim, tag) in zip(ax, [
            (gt, "viridis", (0, 1), "a"), (pred, "viridis", (0, 1), "b"),
            (err, "RdBu_r", (-0.05, 0.05), "c")]):
        im = a.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1], origin="lower")
        a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.98, tag, transform=a.transAxes, va="top", ha="left",
               fontsize=16, fontweight="bold", color="w")
        fig.colorbar(im, ax=a, fraction=0.046)

    a = ax[3]
    # green = ground-truth-supervised times, black = the times never shown.
    a.plot(train_ts.cpu(), per_train, "o-", color="tab:green", lw=2, label="trained times")
    a.plot(held_ts.cpu(), per_held, "s--", color="k", lw=2, label="unseen midpoints")
    a.set_xlabel("t", fontsize=13)
    a.set_ylabel("PSNR vs analytic field (dB)", fontsize=13)
    a.legend(fontsize=11, frameon=False)
    a.grid(alpha=0.3)
    a.text(0.02, 0.98, "d", transform=a.transAxes, va="top", ha="left",
           fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "summary.png"), dpi=130)
    plt.close(fig)


def write_movie(model, field, args, device):
    cmap_v = plt.get_cmap("viridis")
    cmap_e = plt.get_cmap("RdBu_r")
    path = os.path.join(args.out, "field.mp4")
    with imageio.get_writer(path, fps=24, quality=8, macro_block_size=1) as wr:
        for t in np.linspace(0, 1, args.movie_frames):
            gt, pred, err = _panels(model, field, t, args.eval_res, device)
            row = np.concatenate([
                cmap_v(np.clip(gt, 0, 1))[..., :3],
                cmap_v(np.clip(pred, 0, 1))[..., :3],
                cmap_e(np.clip(err / 0.05, -1, 1) * 0.5 + 0.5)[..., :3],
            ], axis=1)
            wr.append_data((row[::-1] * 255).astype(np.uint8))
    print(f"movie: {path}  (ground truth | fit | error x20)")


if __name__ == "__main__":
    main()
