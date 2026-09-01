#!/usr/bin/env python
"""How large an NGP fits on one B300, and how long it would take.

    python scripts/dimension_b300.py
    python scripts/dimension_b300.py --card-gb 80 --speedup 1.0     # the H100 it was measured on

Every constant below was MEASURED in this repo rather than assumed, and is
printed with the number so a stale one is visible:

  compression   32,562,829 parameters reproduced 256 x 1024 x 664 = 174.1 M
                voxels at 53.37 dB -- 5.35 voxels per parameter.  `-o fit
                zapbench_bench`, H100, 400 steps.
  throughput    60.92 M samples/s at batch 262,144 on an H100 with the warp
                stack.  `-o bench zapbench_bench_warp_sadam_bf16dec`.
  coverage      1500 steps x 262,144 = 393 M samples over 174.1 M voxels, so
                2.26 samples per voxel to reach that quality.
  activations   734 B per sample marginal, two-point measured in run_bench.

TWO CAVEATS ON THE HOURS, both of which push the same way.  The measured
throughput came from a 260 MB table, part of which sits in an H100's 50 MB L2;
the tables below are 13 to 48 GB and every gather is a cold HBM read, so the
locality that measurement enjoyed is gone.  And the coverage constant was fitted
at one scale -- 2.26 samples per voxel bought 53.37 dB on 174 M voxels, and a
volume 120x larger may want more passes, not the same number.  Treat every hour
below as a floor rather than an estimate.

THE ONE THING NOT MEASURED is the B300 itself, so its step rate is an estimate:
the kernel is bound by memory traffic and atomic throughput, B300 has roughly
2.4x an H100's bandwidth, and `--speedup` derates that to 2.0 by default.  Every
time below scales linearly with it, so a reader who disagrees can divide.

WHY THE VOLUME IS CUT INTO PLANES.  The full aligned zapbench is 2048 x 1328 x
72 x 7879 = 1.543 T voxels, 3.086 TB at uint16.  At the compression measured
that is 288 G parameters, which is 3.5 TB of optimiser state -- not a big card,
a small cluster.  One z plane is 21.4 G voxels and 42.9 GB, which fits on the
card WITH its model, and the 72 planes are independent fits.  That is the
dimensioning; the whole-volume row is printed to show what it would cost.
"""

from __future__ import annotations

import argparse

# measured, see the docstring
VOXELS_PER_PARAM = 174_129_152 / 32_562_829      # 5.35 at 53.37 dB
SAMPLES_PER_VOXEL = 1500 * 262_144 / 174_129_152  # 2.26
H100_SAMPLES_PER_S = 60.92e6
ACT_B_PER_SAMPLE = 734
FIXED_GB = 1.5                                    # allocator floor measured at 1.29-1.48

FULL = (2048, 1328, 72, 7879)


def dimension(name, voxels, data_b, card_gb, speedup, ratio, table_bytes,
              batch, hours_cap=None):
    """Parameters, memory and wall clock for one fit."""
    params = voxels / ratio
    # p + Adam's m and v.  The moments stay fp32 whatever the table is: they are
    # the running statistics and halving them is a convergence experiment, not a
    # memory optimisation, and it has not been run.
    model_gb = params * (table_bytes + 8) / 1e9
    act_gb = batch * ACT_B_PER_SAMPLE / 1e9
    data_gb = data_b / 1e9
    total = model_gb + act_gb + data_gb + FIXED_GB
    samples = voxels * SAMPLES_PER_VOXEL
    hours = samples / (H100_SAMPLES_PER_S * speedup) / 3600
    return {
        "name": name, "voxels": voxels, "params": params, "model_gb": model_gb,
        "data_gb": data_gb, "total_gb": total, "hours": hours,
        "fits": total <= card_gb,
    }


def ladder_for(params, n_levels):
    """log2 of the table size that gives this many parameters at F=2.

    Entries are shared across levels only in the sense that each hashed level
    gets its own T rows, so params = n_levels * T * 2 once every level is
    hashed, which it is at this scale.
    """
    import math
    T = params / (n_levels * 2)
    return math.log2(max(T, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-gb", type=float, default=288.0)
    ap.add_argument("--speedup", type=float, default=2.0,
                    help="B300 step rate relative to the measured H100")
    ap.add_argument("--batch", type=int, default=4_194_304)
    ap.add_argument("--ratios", type=float, nargs="+", default=[5.35, 10.0, 20.0],
                    help="voxels per parameter; 5.35 is the measured 53.37 dB point")
    ap.add_argument("--bf16", action="store_true",
                    help="hold the table in bf16 (2 B) rather than fp32 (4 B)")
    a = ap.parse_args()

    X, Y, Z, T = FULL
    tb = 2 if a.bf16 else 4
    cuts = [
        ("one z plane", X * Y * T, X * Y * T * 2),
        ("4 planes", X * Y * T * 4, X * Y * T * 4 * 2),
        ("whole volume", X * Y * Z * T, X * Y * Z * T * 2),
    ]
    print(f"\n  zapbench aligned: {X} x {Y} x {Z} x {T} = "
          f"{X * Y * Z * T / 1e12:.3f} T voxels, {X * Y * Z * T * 2 / 1e12:.3f} TB "
          f"at uint16")
    print(f"  card {a.card_gb:.0f} GB, batch {a.batch:,}, table in "
          f"{'bf16' if a.bf16 else 'fp32'}, B300 assumed {a.speedup:.1f}x the "
          f"measured H100\n")
    print(f"  {'cut':>13s} {'voxels':>9s} {'vx/param':>9s} {'params':>9s} "
          f"{'log2 T':>7s} {'model GB':>9s} {'data GB':>8s} {'total GB':>9s} "
          f"{'hours':>7s} {'fits':>5s}")
    for name, voxels, data_b in cuts:
        for ratio in a.ratios:
            d = dimension(name, voxels, data_b, a.card_gb, a.speedup, ratio, tb,
                          a.batch)
            print(f"  {name:>13s} {voxels / 1e9:8.1f}G {ratio:9.2f} "
                  f"{d['params'] / 1e9:8.2f}G {ladder_for(d['params'], 16):7.1f} "
                  f"{d['model_gb']:9.1f} {d['data_gb']:8.1f} {d['total_gb']:9.1f} "
                  f"{d['hours']:7.1f} {'yes' if d['fits'] else 'NO':>5s}")
        print()
    print("  hours are a FLOOR: measured throughput came from a 260 MB table that"
          "\n  partly fits in L2, and these tables are 13-48 GB of cold HBM reads."
          "\n  hours are for ONE fit at the measured coverage of "
          f"{SAMPLES_PER_VOXEL:.2f} samples per voxel;")
    print(f"  the 72 planes are independent, so {Z} cards finish in the time of one.\n")


if __name__ == "__main__":
    main()
