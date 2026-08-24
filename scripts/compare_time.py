#!/usr/bin/env python
"""Accuracy as a continuous function of t, for one or more field checkpoints.

    python scripts/compare_time.py

The per-frame summary in fit_field.py only scores the trained times and their
midpoints.  This samples t densely, which is what shows the shape of the
failure: an isotropic grid is exact at the observed frames and collapses
between them, so its error strobes at the frame rate, while a grid whose time
axis is capped at the frame spacing is flat.
"""

from __future__ import annotations

import argparse
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
from ngp.utils import psnr, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="*", default=[
        os.path.join(ROOT, "out/field_isotropic/model.pt"),
        os.path.join(ROOT, "out/field_smoothstep_xy/model.pt")])
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "out/time_comparison"))
    p.add_argument("--n-times", type=int, default=321)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    ts = np.linspace(0, 1, args.n_times)
    xy = _unit_grid(args.res, device)

    curves, labels, train_ts = [], [], None
    for i, path in enumerate(args.ckpts):
        model, ck = load_checkpoint(path, device)
        field = AdvDiffField(device=device, **ck["field_kwargs"])
        n_train = ck["args"]["n_train_frames"]
        train_ts = np.linspace(0, 1, n_train)
        db = []
        for t in ts:
            xyt = torch.cat([xy, torch.full_like(xy[:, :1], float(t))], dim=1)
            db.append(psnr(render(model, xyt, (args.res, args.res)),
                           field(xyt).reshape(args.res, args.res)))
        curves.append(db)
        cap = ck["model_kwargs"]["max_resolution"]
        auto = ("isotropic" if cap is None
                else f"t capped at {int(cap[2])} cells")
        labels.append(args.labels[i] if args.labels else auto)
        print(f"{labels[-1]:<24} mean {np.mean(db):6.2f} dB   "
              f"worst {np.min(db):6.2f} dB at t={ts[int(np.argmin(db))]:.3f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    for t in train_ts:
        ax.axvline(t, color="0.85", lw=1, zorder=0)
    # Two distinct sources, not ground truth vs prediction: red and blue.
    for db, lab, col in zip(curves, labels, ["tab:red", "tab:blue", "tab:purple", "0.3"]):
        ax.plot(ts, db, color=col, lw=2, label=lab)
    ax.set_xlabel("t", fontsize=13)
    ax.set_ylabel("PSNR vs analytic field (dB)", fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.text(0.01, 0.98, "grey lines: the 17 times the fit was trained on",
            transform=ax.transAxes, va="top", fontsize=11, color="0.4")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "psnr_vs_time.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(args.out, "curves.json"), "w") as f:
        json.dump({"t": ts.tolist(), "labels": labels, "psnr": curves,
                   "train_ts": train_ts.tolist()}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
