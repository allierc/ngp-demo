#!/usr/bin/env python
"""Stage 1: fit a 2D image with the pure-PyTorch hash encoding.

    python scripts/fit_image.py --steps 2000

Trains f(x, y) -> RGB on uniformly random coordinates, reports PSNR against the
full-resolution reference, and writes snapshots + a summary figure to
out/image/.  The checkpoint it saves is what scripts/demo_gradients.py reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import NGPField, save_checkpoint
from ngp.utils import BilinearImage, pixel_centers, psnr, read_image, render, write_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=os.path.join(ROOT, "assets/girl_with_a_pearl_earring.jpg"))
    p.add_argument("--out", default=os.path.join(ROOT, "out/image"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1 << 18, help="random samples per step")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-levels", type=int, default=16)
    p.add_argument("--n-features", type=int, default=2)
    p.add_argument("--log2-hashmap-size", type=int, default=19)
    p.add_argument("--base-resolution", type=int, default=16)
    p.add_argument("--per-level-scale", type=float, default=1.4)
    p.add_argument("--max-resolution", default="auto",
                   help="cells per axis the refinement stops at: 'auto' = the image's "
                        "own pixel count (W, H), 'none' = uncapped, or an integer. "
                        "Uncapped, the finest levels sit below the pixel spacing and "
                        "the fit acquires sub-pixel wiggle that no data constrains -- "
                        "harmless for PSNR, visible in the gradients.")
    p.add_argument("--interpolation", default="linear", choices=["linear", "smoothstep"])
    p.add_argument("--activation", default="relu", choices=["relu", "gelu", "softplus", "tanh"])
    p.add_argument("--n-neurons", type=int, default=64)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--n-snapshots", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ref = read_image(args.image)
    h, w, c = ref.shape
    target = BilinearImage(ref, device)
    coords = pixel_centers(h, w, device)
    ref_t = target(coords).reshape(h, w, c)
    print(f"image {w}x{h}x{c} = {h * w:,} pixels")

    if args.max_resolution == "auto":
        max_res = (w, h)
    elif args.max_resolution == "none":
        max_res = None
    else:
        max_res = int(args.max_resolution)

    model_kwargs = dict(
        n_input_dims=2, n_output_dims=c, max_resolution=max_res,
        n_neurons=args.n_neurons, n_hidden_layers=args.n_hidden_layers,
        activation=args.activation, output_activation="sigmoid",
        n_levels=args.n_levels, n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution, per_level_scale=args.per_level_scale,
        interpolation=args.interpolation,
    )
    model = NGPField(**model_kwargs).to(device)
    n_enc, n_mlp = model.n_parameters()
    print(model.encoding)
    print(f"parameters: {n_enc:,} encoding + {n_mlp:,} decoder = {n_enc + n_mlp:,}")
    print(f"            {(n_enc + n_mlp) / (h * w * c) * 100:.1f}% of the {h * w * c:,} "
          f"reference values")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    snap_at = {int(round(s)) for s in
               torch.logspace(0, torch.log10(torch.tensor(float(args.steps))),
                              args.n_snapshots).tolist()}

    history, t_train = [], 0.0
    for step in range(args.steps + 1):
        t0 = time.perf_counter()
        xy = torch.rand(args.batch, 2, device=device)
        pred = model(xy)
        with torch.no_grad():
            gt = target(xy)
        # instant-NGP's relative-L2: weights dark pixels up, matching its config.
        loss = ((pred - gt) ** 2 / (pred.detach() ** 2 + 1e-2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0

        if step in snap_at or step == 0 or step == args.steps:
            img = render(model, coords, (h, w, c)).clamp(0, 1)
            db = psnr(img, ref_t)
            history.append({"step": step, "train_s": t_train, "psnr": db,
                            "loss": loss.item()})
            print(f"step {step:6d}  {t_train:7.2f}s  loss {loss.item():.5f}  psnr {db:6.2f} dB")
            write_image(os.path.join(args.out, f"fit_{step:06d}.png"), img)

    fit = render(model, coords, (h, w, c)).clamp(0, 1)
    write_image(os.path.join(args.out, "reference.png"), ref_t)
    write_image(os.path.join(args.out, "fit_final.png"), fit)
    save_checkpoint(os.path.join(args.out, "model.pt"), model, model_kwargs,
                    args=vars(args), image=args.image)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": history, "args": vars(args),
                   "n_enc": n_enc, "n_mlp": n_mlp, "h": h, "w": w}, f, indent=1)

    summary_figure(ref_t, fit, history, args.out, n_enc + n_mlp, h * w * c)
    print(f"\nfinal PSNR {history[-1]['psnr']:.2f} dB after {t_train:.1f} s of training")
    print(f"wrote {args.out}")


def summary_figure(ref, fit, history, out, n_params, n_values):
    err = (fit - ref).abs().mean(-1)
    fig, ax = plt.subplots(1, 4, figsize=(17, 6))
    panels = [(ref.cpu(), None, "a"), (fit.cpu(), None, "b"),
              (err.cpu(), "inferno", "c")]
    for a, (img, cmap, tag) in zip(ax, panels):
        im = a.imshow(img.numpy(), cmap=cmap, vmin=0, vmax=None if cmap is None else 0.1)
        a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.98, tag, transform=a.transAxes, va="top", ha="left",
               fontsize=16, fontweight="bold", color="w")
        if cmap is not None:
            fig.colorbar(im, ax=a, fraction=0.046)

    a = ax[3]
    steps = [r["train_s"] for r in history]
    a.plot(steps, [r["psnr"] for r in history], "o-", color="0.2", lw=2)
    a.set_xlabel("training time (s)", fontsize=13)
    a.set_ylabel("PSNR vs reference (dB)", fontsize=13)
    a.set_xscale("log")
    a.grid(alpha=0.3)
    a.text(0.02, 0.98, "d", transform=a.transAxes, va="top", ha="left",
           fontsize=16, fontweight="bold")
    a.text(0.97, 0.05, f"{n_params/1e6:.2f}M params\n{n_params/n_values*100:.1f}% of "
                       f"{n_values/1e6:.1f}M RGB values",
           transform=a.transAxes, ha="right", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
