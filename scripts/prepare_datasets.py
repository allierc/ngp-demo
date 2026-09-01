#!/usr/bin/env python
"""Cut two real datasets down to (T, H, W) slices this repo's pages can fit.

    python scripts/prepare_datasets.py                  # all of them, into data/
    python scripts/prepare_datasets.py --only redox
    python scripts/gui_scalar_time.py --zarr-glob 'data/*/field.zarr'

Both sources live under GraphData/graphs_data and neither is a 2-D field, so
each needs a decision that is worth writing down rather than burying.

REDOX -- redox/Organoid_Redox_Ratio_Analysis_T200_3D_Every_10_min/
    69 TIFF stacks, one per timepoint, each (14, 512, 512) float32, a redox
    ratio in [0, 3.6] with no NaNs, imaged every 10 minutes under washout.  A
    stack is 14 planes deep, so "the middle plane" is plane 7, and the result is
    (69, 512, 512): a genuinely continuous field, slow in time.

ZAPBENCH -- zebrafish/zapbench/zapbench.zarr
    NOT A VOLUME.  The local store holds traces (7870 frames x 71721 neurons)
    and positions (71721 x 3), and every neuron has its own z -- 61,262 distinct
    values over 21 to 275 um -- so there are no planes to take.  "The middle
    plane" is therefore a SLAB about the median z, and each frame is those
    neurons painted at their own (x, y).  That makes a sparse field, which is
    what the data is: the pixels between neurons hold no measurement and are
    written as zero.  The slab thickness is chosen to put roughly a tenth of the
    grid under a neuron, and what it actually achieved is printed.

    Two cuts of it, because the two ask different questions of a fit:
      consecutive  256 frames in a row, 0.914 s apart -- 3.9 minutes
      every20      256 frames at every 20th, spanning 5,120 frames -- 78 minutes
    The first is a fit over a smooth stretch; the second is the same number of
    parameters asked to cover twenty times the span, which is where a time axis
    capped at the frame spacing stops being able to interpolate.

Each output is a zarr in the layout the pages already read -- one `v/grid` of
(T, 1, H, W) and a summary.json saying `part: fine` -- so nothing had to learn a
new format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHDATA = os.environ.get(
    "GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData") + "/graphs_data"
REDOX_DIR = (GRAPHDATA +
             "/redox/Organoid_Redox_Ratio_Analysis_T200_3D_Every_10_min")
ZAPBENCH = GRAPHDATA + "/zebrafish/zapbench/zapbench.zarr"


def write_store(out_dir, vol, meta):
    """(T, H, W) -> a zarr the scalar pages already know how to read."""
    import zarr
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "field.zarr")
    g = zarr.open(path, mode="w")
    a = g.create_dataset("v/grid", shape=(vol.shape[0], 1) + vol.shape[1:],
                         chunks=(8, 1) + vol.shape[1:], dtype="f4",
                         overwrite=True)
    a[:, 0] = vol
    g.create_dataset("v/colors", shape=(1, 3), dtype="f4", overwrite=True)[:] = 1.0
    t, h, w = vol.shape
    summary = {"part": "fine",
               "fields": {"v": {"res": f"{h}x{w}", "frames": int(t),
                                "std": float(vol.std()),
                                "lag1_autocorr": _lag1(vol)}},
               **meta}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {path}  ({t} x {h} x {w}, "
          f"{vol.nbytes / 1e6:.0f} MB uncompressed)", flush=True)
    return summary


def _lag1(v):
    """Lag-1 autocorrelation over the run: near 1 is resolved in time, near 0
    aliased.  The same statistic the toy generator reports, so the numbers from
    the two sources can be read side by side."""
    x = v.reshape(v.shape[0], -1)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    return float((x[:-1] * x[1:]).mean())


# ------------------------------------------------------------------- redox


def prepare_redox(out_root):
    import glob
    import tifffile
    files = sorted(glob.glob(os.path.join(REDOX_DIR, "*.tif")))
    if not files:
        print(f"  no tif under {REDOX_DIR}", flush=True)
        return
    first = tifffile.imread(files[0])
    z = first.shape[0] // 2
    print(f"redox: {len(files)} timepoints, each {first.shape}, middle plane {z}",
          flush=True)
    vol = np.stack([tifffile.imread(f)[z] for f in files]).astype(np.float32)
    print(f"  values {vol.min():.3f}..{vol.max():.3f}, "
          f"{float((vol == 0).mean()) * 100:.1f}% exactly zero", flush=True)
    write_store(os.path.join(out_root, "redox_midplane"), vol,
                {"source": REDOX_DIR, "plane": int(z),
                 "interval_min": 10, "condition": "washout"})


# ---------------------------------------------------------------- zapbench


def prepare_zapbench(out_root, n_frames=256, width=256, target_occupancy=0.10,
                     blob_px=2.0):
    import zarr
    g = zarr.open(ZAPBENCH, mode="r")
    pos = np.asarray(g["positions"])                       # (N, 3) x, y, z
    n_total = g["traces"].shape[0]
    z = pos[:, 2]
    zmid = float(np.median(z))

    # The slab: widened until about a tenth of the grid has a neuron in it. The
    # thickness is a consequence of the target, not a guess -- and the achieved
    # occupancy is printed, because it is the number that says how sparse the
    # field the page will be fitting really is.
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = pos[:, 1].min(), pos[:, 1].max()
    h = max(8, int(round(width * (y1 - y0) / (x1 - x0))))
    ix = np.clip(((pos[:, 0] - x0) / (x1 - x0) * (width - 1)).round(), 0,
                 width - 1).astype(np.int64)
    iy = np.clip(((pos[:, 1] - y0) / (y1 - y0) * (h - 1)).round(), 0,
                 h - 1).astype(np.int64)
    for half in (2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0):
        sel = np.abs(z - zmid) <= half
        occ = len(np.unique(iy[sel] * width + ix[sel])) / (h * width)
        if occ >= target_occupancy:
            break
    print(f"zapbench: {n_total} frames x {len(pos)} neurons; middle slab "
          f"z = {zmid:.1f} +- {half:.1f} um holds {int(sel.sum())} neurons, "
          f"{occ * 100:.1f}% of a {width}x{h} grid", flush=True)

    idx = np.flatnonzero(sel)
    flat = iy[idx] * width + ix[idx]
    # The blob: a Gaussian of `blob_px` sigma, scaled so its own peak is 1, so a
    # lone neuron reads at its trace value and overlapping ones add.  Separable,
    # and applied once per frame to the whole raster rather than per neuron.
    import torch
    r = max(1, int(round(3 * blob_px)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1).float() / blob_px) ** 2)
    k = k / k.max()
    cover = None

    def cut(frames, name, note):
        nonlocal cover
        t0 = time.perf_counter()
        vol = np.zeros((len(frames), h * width), dtype=np.float32)
        tr = g["traces"]
        for j, fr in enumerate(frames):
            row = np.asarray(tr[int(fr), :])[idx]
            np.add.at(vol[j], flat, row)
        v = torch.from_numpy(vol.reshape(len(frames), 1, h, width))
        v = torch.nn.functional.conv2d(v, k.view(1, 1, -1, 1), padding=(r, 0))
        v = torch.nn.functional.conv2d(v, k.view(1, 1, 1, -1), padding=(0, r))
        vol = v[:, 0].numpy()
        cover = float((np.abs(vol) > 1e-3).mean())
        print(f"  {name}: {len(frames)} frames [{frames[0]}..{frames[-1]}], "
              f"blobs of sigma {blob_px:g} px cover {cover * 100:.0f}% of the "
              f"frame, values {vol.min():.3f}..{vol.max():.3f}, read in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
        write_store(os.path.join(out_root, name), vol,
                    {"source": ZAPBENCH, "frames": [int(x) for x in
                                                    (frames[0], frames[-1])],
                     "slab_um": float(half), "z_mid_um": zmid,
                     "occupancy": float(occ), "blob_px": float(blob_px),
                     "coverage": cover, "note": note})

    start = (n_total - n_frames) // 2
    cut(np.arange(start, start + n_frames), "zapbench_consecutive",
        "256 frames in a row, 0.914 s apart")
    stride = 20
    span = n_frames * stride
    s0 = max(0, (n_total - span) // 2)
    cut(np.arange(s0, s0 + span, stride)[:n_frames], "zapbench_every20",
        f"256 frames at every {stride}th, spanning {span} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--only", choices=["redox", "zapbench"], default=None)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blob-px", type=float, default=2.0,
                    help="Gaussian sigma of a neuron's blob, in grid pixels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "redox"):
        prepare_redox(a.out)
    if a.only in (None, "zapbench"):
        prepare_zapbench(a.out, a.frames, a.width, blob_px=a.blob_px)
    print(f"\nfit them with:\n  python scripts/gui_scalar_time.py "
          f"--zarr-glob '{a.out}/*/field.zarr'")


if __name__ == "__main__":
    main()
