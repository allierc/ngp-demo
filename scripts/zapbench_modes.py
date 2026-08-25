#!/usr/bin/env python
"""How many modes does the zapbench flow field need? Masked, over the whole run.

    python scripts/zapbench_modes.py                 # the headline analysis
    python scripts/zapbench_modes.py --stride 20     # denser in time, slower
    python scripts/zapbench_modes.py --consecutive   # the aliasing control

Needs `tensorstore` (in the env specs) and a network; nothing else here does.

The question is whether a dense 4D displacement field carries dense information.
It does not: inside the cells, across all 7,879 frames, two spatial modes and
their time courses reconstruct the time-varying part to well under one unit.

Three things this gets right that a first pass gets wrong.

  * MASKED. The segmentation is 2048x1328x72, exactly 16x16x2 the flow grid, so
    a block-max over it says which flow voxels hold a cell. Only 21.6% do; the
    rest is background where the flow is extrapolated and means nothing, and
    including it flatters every statistic.
  * THE WHOLE RUN. Sampling 197 consecutive frames says nothing about drift over
    7,879. Frames are taken at a stride spanning the entire recording.
  * NOT ALIASED. Conversely, a stride hides anything faster than itself. Run with
    --consecutive to measure the same thing on adjacent frames; if the two agree,
    the stride is not inventing structure. They do agree, which is how the broad
    tail in the first pass was identified as aliasing rather than signal.

A caveat that no amount of analysis here can remove: THE FLOW ARRAY HAS NO UNITS.
`raw` and `segmentation` both declare 406 x 406 x 4000 nm; `flow_fields` declares
nothing. So the errors below are in the array's own units, which are either raw
voxels or flow-grid voxels -- a factor of 16 in x and y. Calibrating against the
raw data does not work either, because `raw` is already aligned: the measured
shift between frames 7,000 apart is under 0.01 voxel, so there is no residual
motion for the flow field to explain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

BUCKET = "zapbench-release"
FLOW = "volumes/20240930/flow_fields/"
SEG = "volumes/20240930/segmentation/"


def _open(path):
    import tensorstore as ts
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "gcs",
                    "bucket": BUCKET, "path": path}}, open=True, read=True).result()


def build_mask(flow_shape):
    """Per-flow-voxel 'a segmented cell is here', by block-max over the segmentation."""
    C, Z, Y, X, T = flow_shape
    seg = _open(SEG)
    SX, SY, SZ = seg.domain.shape
    fx, fy, fz = SX // X, SY // Y, SZ // Z
    print(f"  segmentation {SX}x{SY}x{SZ} is {fx}x{fy}x{fz} the flow grid", flush=True)
    mask = np.zeros((Z, Y, X), dtype=bool)
    t0 = time.perf_counter()
    for z in range(Z):
        s = seg[:, :, z * fz:(z + 1) * fz].read().result()
        mask[z] = (s > 0).reshape(X, fx, Y, fy, fz).any(axis=(1, 3, 4)).T
    print(f"  mask: {mask.sum():,} of {mask.size:,} flow voxels ({mask.mean()*100:.1f}%) "
          f"hold a cell   [{time.perf_counter()-t0:.0f} s]", flush=True)
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stride", type=int, default=45,
                   help="frame stride; the default spans the whole run in 176 frames")
    p.add_argument("--consecutive", action="store_true",
                   help="adjacent frames instead, as the aliasing control")
    p.add_argument("--frames", type=int, default=176)
    p.add_argument("--json", default=None, help="also write the table here")
    a = p.parse_args()
    try:
        import tensorstore                                       # noqa: F401
    except ImportError:
        sys.exit("this analysis needs tensorstore:  pip install tensorstore")

    flow = _open(FLOW)
    C, Z, Y, X, T = flow.domain.shape
    print(f"flow field {C}x{Z}x{Y}x{X}x{T} = {C*Z*Y*X*T/1e9:.2f}e9 values", flush=True)
    mask = build_mask(flow.domain.shape)
    idx = np.flatnonzero(mask.reshape(-1))

    ts_idx = (list(range(a.frames)) if a.consecutive
              else list(range(0, T, a.stride))[:a.frames])
    M = np.empty((3 * len(idx), len(ts_idx)), dtype=np.float32)
    t0 = time.perf_counter()
    for j, t in enumerate(ts_idx):
        M[:, j] = flow[:, :, :, :, t].read().result().reshape(3, -1)[:, idx].reshape(-1)
        if j % 40 == 0:
            print(f"    {j}/{len(ts_idx)}  [{time.perf_counter()-t0:.0f} s]", flush=True)
    span = f"t={ts_idx[0]}..{ts_idx[-1]}" + (" CONSECUTIVE" if a.consecutive else "")
    print(f"  read {len(ts_idx)} frames, {span}   [{time.perf_counter()-t0:.0f} s]\n",
          flush=True)

    mag = np.linalg.norm(M.reshape(3, len(idx), len(ts_idx)), axis=0)
    A = M - M.mean(1, keepdims=True)          # the time-VARYING part; the mean is free
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    tot = (A ** 2).sum()

    print(f"masked, mean removed: {len(idx):,} voxels x {len(ts_idx)} frames")
    print(f"  displacement inside the mask: mean |u| {mag.mean():.2f}, "
          f"largest {mag.max():.2f}  (ARRAY UNITS -- see the docstring)\n")
    print("    k   energy kept    RMS error    p95 error")
    out = {"n_voxels": int(len(idx)), "n_frames": len(ts_idx),
           "consecutive": bool(a.consecutive), "stride": a.stride,
           "mean_abs_u": float(mag.mean()), "units": "undocumented"}
    for k in (1, 2, 3, 5, 8, 12, 20, 40):
        if k >= min(A.shape):
            break
        R = A - (U[:, :k] * S[:k]) @ Vt[:k]
        err = np.linalg.norm(R.reshape(3, len(idx), len(ts_idx)), axis=0)
        rms, p95 = float(np.sqrt((err ** 2).mean())), float(np.percentile(err, 95))
        keep = float(1 - (R ** 2).sum() / tot)
        print(f"  {k:3d}    {keep*100:9.3f}%   {rms:10.3f}   {p95:10.3f}")
        out[f"rank_{k}"] = {"energy": keep, "rms": rms, "p95": p95}

    dense = C * Z * Y * X * T
    k = 3
    compact = (k + 1) * 3 * len(idx) + k * T
    print(f"\n  a rank-{k} model of the masked field: {compact:,} numbers "
          f"({k} modes + the mean, plus {k} time courses)")
    print(f"  the stored field: {dense:,} -> {dense/compact:,.0f}x larger")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()
