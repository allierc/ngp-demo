#!/usr/bin/env python
"""One (x, y, z, t) NGP over the whole of zapbench: what fits on a B300.

    python scripts/dimension_b300.py
    python scripts/dimension_b300.py --bf16 --levels 24
    python scripts/dimension_b300.py --card-gb 80 --speedup 1.0   # the H100 it was measured on

The planes are not independent -- z is a spatial axis of one brain, not a batch
dimension -- so this dimensions a SINGLE four-dimensional fit of
2048 x 1328 x 72 x 7879 = 1.543 T voxels, 3.086 TB at uint16.

MEASURED CONSTANTS, all from this repo, printed beside the answer so a stale one
is visible:

  compression   32,562,829 parameters reproduced 174.1 M voxels at 53.37 dB --
                5.35 voxels per parameter.  `-o fit zapbench_bench` on an H100.
  throughput    60.92 M samples/s at batch 262,144, THREE dimensions.
                `-o bench zapbench_bench_warp_sadam_bf16dec`.
  coverage      393 M samples over 174.1 M voxels and 32.6 M parameters: 2.26
                samples per voxel, or 12.1 samples per parameter.
  activations   734 B per sample marginal, two-point measured in run_bench.

FOUR DIMENSIONS COST A FACTOR OF TWO.  A level reads 2^D corners, so 16 instead
of 8: twice the gather and twice the atomic adds per sample per level, and the
backward was already 61.5% of the step.  `--corner-cost` derates the measured
3D throughput by that factor.  The equations are unchanged, and D=4 is now in
tests/impl_gate.py, which holds it to 3e-11 on the forward and 4e-07 relative on
the table gradient against the reference encoder.

TWO LIMITS BITE BEFORE THE CARD DOES.

  * THE INDEX IS INT32.  `_corner_index` returns wp.int32 and the level offset
    is added in int32, so the table cannot exceed 2^31 = 2.147 G rows, which at
    F = 2 is 4.29 G parameters and 51.5 GB with Adam -- a fifth of the card.
    Lifting it is a type change and a recompile, not a redesign, but it has not
    been done and nothing below pretends otherwise.

  * THE DATA DOES NOT FIT, and cannot.  3.086 TB against 288 GB.  The fit must
    stream, and HOW it streams decides whether the run is compute-bound or dead:
    uniformly random (x, y, z, t) means one 512 kB chunk read per sample, while
    sampling chunk by chunk -- load a window, sample it hard, move on -- costs
    one pass over the volume per epoch.  Both rates are printed below.

WHAT THIS CANNOT TELL YOU is the quality.  Every parameter count here implies a
compression far past the one measured point, and PSNR at 359 voxels per
parameter is not an extrapolation anybody should make from a single measurement
at 5.35.  The honest next step is a scaling run -- fit 4 planes x 1000 frames,
then 8 x 2000, at fixed voxels per parameter, and watch what the curve does --
before committing a card for a day.
"""

from __future__ import annotations

import argparse
import math

VOXELS_PER_PARAM_MEASURED = 174_129_152 / 32_562_829       # 5.35 at 53.37 dB
SAMPLES_PER_VOXEL = 1500 * 262_144 / 174_129_152           # 2.26
SAMPLES_PER_PARAM = 1500 * 262_144 / 32_562_829            # 12.1
H100_SAMPLES_PER_S = 60.92e6
ACT_B_PER_SAMPLE = 734
FIXED_GB = 1.5
INT32_ROWS = 2 ** 31

FULL = (2048, 1328, 72, 7879)
CHUNK_B = 512 * 512 * 2          # the store's native chunk, (512, 512, 1, 1) uint16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-gb", type=float, default=288.0)
    ap.add_argument("--speedup", type=float, default=2.0,
                    help="B300 step rate relative to the measured H100")
    ap.add_argument("--corner-cost", type=float, default=2.0,
                    help="4D cost over 3D; 2.0 is the corner count 16/8")
    ap.add_argument("--batch", type=int, default=4_194_304)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[5.35, 50.0, 100.0, 359.0, 1000.0],
                    help="voxels per parameter; 5.35 is the measured 53.37 dB point")
    ap.add_argument("--bf16", action="store_true",
                    help="hold the table in bf16 (2 B); Adam's moments stay fp32")
    ap.add_argument("--cache-gb", type=float, default=64.0,
                    help="volume kept resident as the streaming window")
    a = ap.parse_args()

    X, Y, Z, T = FULL
    voxels = X * Y * Z * T
    data_b = voxels * 2
    tb = 2 if a.bf16 else 4
    rate = H100_SAMPLES_PER_S * a.speedup / a.corner_cost
    act_gb = a.batch * ACT_B_PER_SAMPLE / 1e9

    print(f"\n  zapbench aligned: {X} x {Y} x {Z} x {T} = {voxels / 1e12:.3f} T "
          f"voxels, {data_b / 1e12:.3f} TB at uint16, ONE 4D fit")
    print(f"  card {a.card_gb:.0f} GB, batch {a.batch:,} ({act_gb:.1f} GB), "
          f"{a.levels} levels, table in {'bf16' if a.bf16 else 'fp32'}, "
          f"streaming window {a.cache_gb:.0f} GB")
    print(f"  step rate {rate / 1e6:.1f} M samples/s = the measured "
          f"{H100_SAMPLES_PER_S / 1e6:.1f} x {a.speedup:.1f} (B300) / "
          f"{a.corner_cost:.1f} (16 corners not 8)\n")

    print(f"  {'vx/param':>9s} {'params':>8s} {'log2 T':>7s} {'model GB':>9s} "
          f"{'total GB':>9s} {'fits':>5s} {'int32':>6s} "
          f"{'h, 1 pass':>10s} {'h, params':>10s}")
    for ratio in a.ratios:
        params = voxels / ratio
        rows = params / 2
        model_gb = params * (tb + 8) / 1e9
        total = model_gb + act_gb + a.cache_gb + FIXED_GB
        log2t = math.log2(max(rows / a.levels, 1))
        # TWO CLOCKS, and the larger one is the answer. A fit needs enough
        # samples to determine its parameters (12.1 each, measured) AND enough
        # to have looked at the data (one pass = one sample per voxel). At these
        # compressions the second is far larger, which is the useful fact: the
        # run is paced by the size of the volume, not the size of the model.
        h_pass = voxels / rate / 3600
        h_par = params * SAMPLES_PER_PARAM / rate / 3600
        print(f"  {ratio:9.1f} {params / 1e9:7.2f}G {log2t:7.1f} {model_gb:9.1f} "
              f"{total:9.1f} {'yes' if total <= a.card_gb else 'NO':>5s} "
              f"{'ok' if rows <= INT32_ROWS else 'OVER':>6s} "
              f"{h_pass:10.1f} {h_par:10.1f}")

    print(f"\n  int32 ceiling: {INT32_ROWS / 1e9:.2f} G rows = "
          f"{INT32_ROWS * 2 / 1e9:.2f} G parameters = "
          f"{INT32_ROWS * 2 * (tb + 8) / 1e9:.1f} GB with Adam, "
          f"{voxels / (INT32_ROWS * 2):.0f} voxels per parameter")
    rand_gb_s = rate * CHUNK_B / 1e9
    seq_gb_s = data_b / (voxels / rate) / 1e9
    print(f"\n  streaming, at {rate / 1e6:.1f} M samples/s:")
    print(f"    uniformly random (x,y,z,t): one {CHUNK_B / 1e3:.0f} kB chunk per "
          f"sample = {rand_gb_s / 1e3:,.1f} TB/s of reads -- impossible")
    print(f"    chunk by chunk, one pass:   {data_b / 1e12:.3f} TB per epoch = "
          f"{seq_gb_s:.2f} GB/s sustained -- local NVMe, comfortably")
    print(f"    so the sampler must walk the store, not the coordinate space:")
    print(f"    load a {a.cache_gb:.0f} GB window, sample it hard, move on.\n")


if __name__ == "__main__":
    main()
