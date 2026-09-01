#!/usr/bin/env python
"""One NGP setting against an injected artefact with a known ground truth, as one job.

    python scripts/sweep_inject.py --mode shot:0.02 --levels 16 --log2t 22 --px 2 --frames 2
    python scripts/sweep_inject.py --table                  # every result so far

The truth is the bison clip, a natural video with no artefact of its own, so
everything in the residual was put there on purpose.  The recording cannot play
this role: zapbench already contains the stripes under study, and "recovering
the truth" there means recovering something that is itself striped.

WHAT THE KNOBS WERE EXPECTED TO DO, and what they did.  A stripe one row high
and one frame long is representable only by the finest levels, so coarsening the
ladder should force the fit to average the artefact away:

    px per finest cell     larger -> cannot resolve one row
    frames per finest cell larger -> cannot resolve one frame
    n_levels               fewer  -> the ladder stops before it gets there
    log2 hashmap size      smaller-> the fine levels collide and blur

HALF RIGHT.  Coarsening does reject the artefact, exactly as predicted: on
shot:0.02 the offset left at struck pixels falls from 0.045 at 2 px per cell to
0.008 at 8 px, a fivefold rejection.  But the total error goes the other way,
0.052 to 0.087, because the same coarsening destroys more signal than it saves
in artefact.  The best setting on every one of the three modes was the FINEST
one tried (1 px per cell, 3.45x the parameters), and it bought 12.1% on
shot:0.02, 4.5% on stripe:0.1 and 0.7% on stripe:0.8.

So the sweep does not change any ranking.  Where the median wins it still wins
after tuning -- median k=3 scores 0.028 on shot:0.02 against the best fit's
0.046 -- and where the fit wins, at stripe:0.8, it wins with the settings it
already had.  The two methods are not close enough for hyperparameters to
decide; they are suited to different artefacts, and that is the result.

Every axis is swept one at a time about the baseline, because a full grid is 300
runs of a thing whose answer is a curve, not a corner.

ONE JOB PER SETTING, results as json in `log/sweep_inject/`, so a rack of L4s
takes one each (20 s per fit) and `--table` reads whatever came back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import NGP_Main as main_mod                                          # noqa: E402
from ngp.utils import render                                         # noqa: E402
import gui_scalar_time as page                                       # noqa: E402
from median_baseline import corrupt_by, median_stack                 # noqa: E402


def fit_and_score(cfg, vol, truth, hit, device, steps):
    """Fit the corrupted volume, render it back, score against the truth."""
    T, h, w = vol.shape
    model, _ = main_mod.build_from_config(cfg, (h, w), T, device)
    opt = main_mod.make_opt(cfg, model, cfg["training"]["lr"])
    batch = cfg["training"]["batch"]
    t0 = time.perf_counter()
    for _ in range(steps):
        main_mod.one_step(model, opt, vol, batch, device, T)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_s = time.perf_counter() - t0
    out = torch.empty_like(vol)
    with torch.no_grad():
        for i in range(T):
            out[i] = render(model, page.frame_coords(h, w, i / max(1, T - 1), device),
                            (h, w))
    d = out - truth
    n_enc, n_mlp = model.n_parameters()
    return {
        "rmse": float(d.pow(2).mean().sqrt()),
        "rmse_struck": float(d[hit].pow(2).mean().sqrt()),
        "offset_struck": float(d[hit].mean()),
        "rmse_clean": float(d[~hit].pow(2).mean().sqrt()),
        "psnr_vs_corrupt": float(main_mod.psnr(out, vol)),
        "n_parameters": n_enc + n_mlp,
        "train_s": train_s,
    }


def run_table(log_dir):
    rows = []
    for f in sorted(os.listdir(log_dir)) if os.path.isdir(log_dir) else []:
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(log_dir, f))))
    if not rows:
        sys.exit(f"nothing under {log_dir}")
    for mode in sorted({r["mode"] for r in rows}):
        _one_table([r for r in rows if r["mode"] == mode], mode)


def _one_table(rows, mode):
    rows.sort(key=lambda r: r["rmse"])
    ref = [r for r in rows if r.get("is_baseline")]
    print(f"\n  {mode}: amplitude {rows[0]['amp']:.4g}, {rows[0]['steps']} steps, "
          f"{rows[0]['config']}\n")
    print(f"  {'L':>3s} {'log2T':>6s} {'px':>3s} {'frames':>7s} {'params':>10s} "
          f"{'RMSE vs truth':>14s} {'RMSE stripes':>13s} {'RMSE clean':>11s} "
          f"{'fit s':>7s} {'axis':>8s}")
    for r in rows:
        mark = " *" if r.get("is_baseline") else ""
        print(f"  {r['levels']:3d} {r['log2t']:6d} {r['px']:3d} {r['frames']:7d} "
              f"{r['n_parameters'] / 1e6:9.1f}M {r['rmse']:14.3f} "
              f"{r.get('rmse_struck', float('nan')):13.3f} {r['rmse_clean']:11.3f} "
              f"{r['train_s']:7.1f} {r.get('axis', '-') + mark:>8s}")
    if ref:
        b = ref[0]
        best = rows[0]
        print(f"\n  baseline {b['rmse']:.3f} -> best {best['rmse']:.3f} "
              f"(L={best['levels']}, T=2^{best['log2t']}, px={best['px']}, "
              f"frames={best['frames']}), "
              f"{100 * (b['rmse'] - best['rmse']) / b['rmse']:.1f}% lower error"
              f" with {best['n_parameters'] / b['n_parameters']:.2f}x the parameters")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bisons")
    ap.add_argument("--mode", default="shot:0.02",
                    help="artefact to tune against, as in median_baseline")
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--log2t", type=int, default=22)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--axis", default="-", help="which knob this run varies")
    ap.add_argument("--amp", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    log_dir = os.path.join(ROOT, "log", "sweep_inject")
    os.makedirs(log_dir, exist_ok=True)
    if a.table:
        return run_table(log_dir)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg, _ = main_mod.load_config(a.config)
    cfg["encoder"]["n_levels"] = a.levels
    cfg["encoder"]["log2_hashmap_size"] = a.log2t
    cfg["encoder"]["px_per_finest_cell"] = a.px
    cfg["encoder"]["frames_per_finest_cell"] = a.frames
    truth, T, h, w, _ = main_mod.get_data(cfg, device)
    # The same stripes for every job: the amplitude comes from the data and the
    # mask from a fixed seed, so two settings are scored on one corruption and
    # the difference between them is the setting.
    amp = a.amp if a.amp > 0 else 2.5 * float((truth[1:] - truth[:-1]).std())
    vol, hit = corrupt_by(a.mode, truth, amp)
    tag = f"{a.mode.replace(':', '')}_L{a.levels}_T{a.log2t}_px{a.px}_f{a.frames}"
    print(f"[sweep ] {tag}: {a.mode}, amplitude {amp:.4g}, {a.steps} steps "
          f"on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}",
          flush=True)
    out = fit_and_score(cfg, vol, truth, hit, device, a.steps)
    out.update(levels=a.levels, log2t=a.log2t, px=a.px, frames=a.frames,
               axis=a.axis, mode=a.mode, amp=amp, steps=a.steps, tag=tag,
               config=a.config,
               is_baseline=(a.levels == 16 and a.log2t == 22 and a.px == 2
                            and a.frames == 2))
    f = os.path.join(log_dir, f"{tag}.json")
    json.dump(out, open(f, "w"), indent=1)
    print(f"[sweep ] {tag}: RMSE {out['rmse']:.3f}, offset {out['offset_struck']:.3f}, "
          f"clean {out['rmse_clean']:.3f}, {out['n_parameters'] / 1e6:.1f}M params, "
          f"{out['train_s']:.0f} s -> {f}", flush=True)


if __name__ == "__main__":
    main()
