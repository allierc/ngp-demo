#!/usr/bin/env python
"""Who wins a hash collision?  A measurement of the claim in Müller et al. 2022 §3.

    python scripts/collision_audit.py              # ~4 min on one A6000
    python scripts/collision_audit.py --quick      # 400 steps, for a smoke test

The paper argues that a table entry shared by two distant points is not
corrupted, because "the gradients of the more important samples dominate the
collision average and the aliased table entry will naturally be optimized in
such a way that it reflects the needs of the higher-weighted point".  In a
radiance field "important" means density x visibility.  Here it is made an
explicit knob, which is the only way to test the claim rather than illustrate
it: the painting is split into a checkerboard of blocks, region A is fitted at
loss weight 1 and region B at weight lambda.

The checkerboard is what makes the test non-circular.  A and B are the same
picture at the same scales, interleaved, so they touch the fine levels equally
often -- panel (b) measures that they do.  Any asymmetry that appears when
lambda falls therefore comes from the gradient weight and not from one region
being bigger, smoother or more often sampled.

Six runs: lambda in {1, 0.1, 0.01} at a table small enough that the fine levels
collide, and the same three at a table large enough that every level is dense.
The dense runs are the control that separates "B learns less because its
gradients are small" from "B loses its shared entries to A".

Everything the figure states is measured here; nothing is asserted.  Writes
figures/collision_audit.png and figures/collision_audit.json.
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
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import NGPField
from ngp.utils import BilinearImage, pixel_centers, read_image, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = os.path.join(ROOT, "assets/girl_with_a_pearl_earring.jpg")

# gui_image.py's defaults, so the audit describes the encoder that page fits.
ENC = dict(n_input_dims=2, n_neurons=64, n_hidden_layers=2, activation="relu",
           output_activation="sigmoid", n_levels=18, n_features_per_level=2,
           base_resolution=4, per_level_scale=1.35, interpolation="linear")


def masked_psnr(pred, ref, mask):
    mse = ((pred - ref) ** 2)[mask].mean()
    return float(-10.0 * torch.log10(mse.clamp(min=1e-12)))


def checkerboard(h, w, block, device):
    """(H, W) bool: region A.  ~B is region B."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing="ij")
    return ((yy // block + xx // block) % 2) == 0


def region_of(xy, h, w, block):
    """Which region a continuous coordinate in [0,1]^2 falls in.  Same rule as
    checkerboard(), evaluated on the sample rather than on the pixel grid."""
    px = (xy[:, 0] * w).floor().long().clamp(0, w - 1)
    py = (xy[:, 1] * h).floor().long().clamp(0, h - 1)
    return ((py // block + px // block) % 2) == 0


def level_id(enc):
    """(n_entries,) int: which level each table row belongs to."""
    lid = torch.zeros(enc.n_entries, dtype=torch.long)
    for l, off in enumerate(enc.level_offsets):
        end = enc.level_offsets[l + 1] if l + 1 < enc.n_levels else enc.n_entries
        lid[off:end] = l
    return lid


@torch.enable_grad()
def touch_weight(model, coords, chunk=65536):
    """(n_entries,) sum of interpolation weight arriving at each table row.

    d/d(table row) of sum_i encoding(x_i) is exactly the total corner weight the
    given samples put on that row: a purely geometric quantity, with no residual
    and no training in it.  A row with weight 0 is one these samples never read.
    """
    enc = model.encoding
    acc = torch.zeros(enc.n_entries, device=coords.device)
    for i in range(0, coords.shape[0], chunk):
        g = torch.autograd.grad(enc(coords[i:i + chunk]).sum(), enc.table)[0]
        acc += g.abs().mean(-1)
    return acc.detach()


def train(img, target, coords, shape, mask_a, log2_t, lam, steps, batch, lr,
          device, block, mass_every=10, seed=0):
    """One fit.  Returns the model, per-region PSNR, and the gradient mass that
    each region delivered to each table row over the whole run."""
    h, w, c = shape
    torch.manual_seed(seed)
    max_res = (w, h)
    model = NGPField(n_output_dims=c, log2_hashmap_size=log2_t,
                     max_resolution=max_res, **ENC).to(device)
    enc = model.encoding
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mass_a = torch.zeros(enc.n_entries, device=device)
    mass_b = torch.zeros(enc.n_entries, device=device)
    t0 = time.perf_counter()
    for step in range(steps):
        xy = torch.rand(batch, 2, device=device)
        in_a = region_of(xy, h, w, block)
        tgt = target(xy)
        per = ((model(xy) - tgt) ** 2).mean(-1)
        # Split by region rather than weighted-then-summed, so the two halves of
        # the gradient can be read separately.  L_a + L_b is exactly the loss.
        l_a = per[in_a].sum() / batch
        l_b = lam * per[~in_a].sum() / batch
        if step % mass_every == 0:
            ga, gb = torch.autograd.grad(l_a, enc.table, retain_graph=True)[0], \
                     torch.autograd.grad(l_b, enc.table, retain_graph=True)[0]
            mass_a += ga.abs().sum(-1)
            mass_b += gb.abs().sum(-1)
        opt.zero_grad(set_to_none=True)
        (l_a + l_b).backward()
        opt.step()
    with torch.no_grad():
        pred = render(model, coords, shape)
    out = {"psnr_a": masked_psnr(pred, img, mask_a),
           "psnr_b": masked_psnr(pred, img, ~mask_a),
           "seconds": time.perf_counter() - t0,
           "n_hashed": int(sum(1 for d in enc.dense if not d)),
           "n_entries": int(enc.n_entries)}
    return model, out, mass_a.detach(), mass_b.detach()


@torch.no_grad()
def perturb(model, rows, img, coords, shape, mask_a, base_a, base_b):
    """Zero these table rows; return the PSNR each region loses."""
    if rows.numel() == 0:
        return None
    keep = model.encoding.table.data[rows].clone()
    model.encoding.table.data[rows] = 0.0
    pred = render(model, coords, shape)
    model.encoding.table.data[rows] = keep
    return {"n_rows": int(rows.numel()),
            "drop_a": base_a - masked_psnr(pred, img, mask_a),
            "drop_b": base_b - masked_psnr(pred, img, ~mask_a)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--block", type=int, default=32, help="checkerboard block, px")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=262144)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    ap.add_argument("--log2-small", type=int, default=10, help="the colliding table")
    ap.add_argument("--log2-large", type=int, default=19, help="the dense control")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the figure from an existing json, no training")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    # cuda:0 only: cuda:1 is in use for timing elsewhere.
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.quick:
        a.steps = 400
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)

    if a.replot:
        out = json.load(open(os.path.join(a.out, "collision_audit.json")))
        a.lambdas = [r["lam"] for r in out["runs"].values() if r["tag"] == "collide"]
        a.block = out["settings"]["block"]
        a.downsample = out["settings"]["downsample"]
        ref = torch.from_numpy(read_image(IMAGE))
        if a.downsample > 1:
            ref = F.avg_pool2d(ref.permute(2, 0, 1)[None], a.downsample)[0].permute(1, 2, 0)
        h, w, _ = ref.shape
        figure(out, ref.numpy(), checkerboard(h, w, a.block, ref.device).numpy(), a)
        print(f"replotted {os.path.join(a.out, 'collision_audit.png')}")
        return

    ref = torch.from_numpy(read_image(IMAGE)).to(dev)
    if a.downsample > 1:
        ref = F.avg_pool2d(ref.permute(2, 0, 1)[None], a.downsample)[0].permute(1, 2, 0)
    h, w, c = ref.shape
    shape = (h, w, c)
    target = BilinearImage(ref.cpu().numpy(), dev)
    coords = pixel_centers(h, w, dev)
    mask_a = checkerboard(h, w, a.block, dev)

    # Are the two regions the same picture?  If they are not, every later
    # asymmetry is confounded, so this is measured before anything is trained.
    gx = (ref[:, 1:] - ref[:, :-1]).abs().mean(-1)
    gy = (ref[1:] - ref[:-1]).abs().mean(-1)
    detail = torch.zeros(h, w, device=dev)
    detail[:, :-1] += gx
    detail[:-1] += gy
    content = {"detail_a": float(detail[mask_a].mean()),
               "detail_b": float(detail[~mask_a].mean()),
               "px_a": int(mask_a.sum()), "px_b": int((~mask_a).sum()),
               "mean_a": float(ref[mask_a].mean()), "mean_b": float(ref[~mask_a].mean())}
    print(f"image {w}x{h}, checkerboard {a.block} px: "
          f"A {content['px_a']:,} px / B {content['px_b']:,} px, "
          f"mean |grad I| {content['detail_a']:.4f} vs {content['detail_b']:.4f}",
          flush=True)

    runs, models, masses = {}, {}, {}
    for tag, log2_t in (("collide", a.log2_small), ("dense", a.log2_large)):
        for lam in a.lambdas:
            key = f"{tag}_lam{lam:g}"
            m, r, ma, mb = train(ref, target, coords, shape, mask_a, log2_t, lam,
                                 a.steps, a.batch, a.lr, dev, a.block)
            r.update(tag=tag, lam=lam, log2_t=log2_t)
            runs[key], models[key], masses[key] = r, m, (ma, mb)
            print(f"  {key:16s} T=2^{log2_t:<2d} {r['n_hashed']}/{ENC['n_levels']} levels "
                  f"hashed  psnr A {r['psnr_a']:6.2f} dB  B {r['psnr_b']:6.2f} dB  "
                  f"gap {r['psnr_a'] - r['psnr_b']:+5.2f}  [{r['seconds']:.0f} s]",
                  flush=True)

    # --- the geometry: which rows do the two regions actually share? ----------
    # Run on BOTH tables. Two regions can share a row for two different reasons:
    # they are adjacent (a node on the boundary between an A block and a B block
    # is legitimately read by both) or they collide (distant nodes hashed to one
    # row). The dense table can only do the first, so it is the adjacency
    # baseline and the excess in the small table is what collisions added.
    def census_of(key):
        enc = models[key].encoding
        lid = level_id(enc).to(dev)
        wa = touch_weight(models[key], coords[mask_a.reshape(-1)])
        wb = touch_weight(models[key], coords[(~mask_a).reshape(-1)])
        shared = (wa > 0) & (wb > 0)
        rows_out = []
        for l in range(enc.n_levels):
            rows = lid == l
            n = max(1, int(rows.sum()))
            sh = shared & rows
            rows_out.append({
                "level": l, "cells": int(enc.resolutions[l][0]),
                "px_per_cell": round(w / enc.resolutions[l][0], 2),
                "dense": bool(enc.dense[l]), "rows": int(rows.sum()),
                "touched_a": int(((wa > 0) & rows).sum()),
                "touched_b": int(((wb > 0) & rows).sum()),
                "shared": int(sh.sum()), "shared_frac": float(sh.sum() / n),
                # the balance check: matched content should put the same
                # interpolation weight on a shared row from either side
                "weight_ratio": (float(wa[sh].sum() / wb[sh].sum().clamp(min=1e-9))
                                 if int(sh.sum()) else float("nan"))})
        return rows_out, lid, shared

    ref_key = f"collide_lam{a.lambdas[0]:g}"
    census, lid, shared = census_of(ref_key)
    census_dense, _, _ = census_of(f"dense_lam{a.lambdas[0]:g}")
    enc = models[ref_key].encoding
    hashed_row = torch.tensor([not d for d in enc.dense], device=dev)[lid]
    n_shared_hashed = int((shared & hashed_row).sum())
    print(f"\nrows touched by both regions: {int(shared.sum()):,} of {enc.n_entries:,} "
          f"in the small table; {n_shared_hashed:,} of them on hashed levels",
          flush=True)
    for lc, ld in zip(census, census_dense):      # not c/d: c is the channel count
        if not lc["dense"]:
            print(f"    level {lc['level']:2d}  {lc['cells']:4d} cells "
                  f"({lc['px_per_cell']:6.2f} px)  shared {lc['shared_frac']*100:5.1f}% "
                  f"of {lc['rows']:,} rows, against {ld['shared_frac']*100:5.1f}% "
                  f"of {ld['rows']:,} when the same level is dense", flush=True)

    # --- the claim: where does a shared row's gradient mass come from? --------
    dom, perturbs = {}, {}
    for lam in a.lambdas:
        key = f"collide_lam{lam:g}"
        ma, mb = masses[key]
        sel = shared & hashed_row & ((ma + mb) > 0)
        d = (ma[sel] / (ma[sel] + mb[sel])).cpu().numpy()
        dom[key] = d
        r = runs[key]
        print(f"  {key:16s} dominance of A at shared hashed rows: "
              f"median {np.median(d):.3f}, mean {d.mean():.3f}, "
              f"{(d > 0.9).mean()*100:5.1f}% above 0.9  ({sel.sum().item():,} rows)",
              flush=True)
        # does the mass split predict who the row ends up serving?
        idx = torch.nonzero(sel).squeeze(-1)
        dt = torch.from_numpy(d).to(dev)
        bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.0001)]
        pb = []
        for lo, hi in bins:
            rows = idx[(dt >= lo) & (dt < hi)]
            res = perturb(models[key], rows, ref, coords, shape, mask_a,
                          r["psnr_a"], r["psnr_b"])
            if res:
                res.update(lo=lo, hi=hi)
                pb.append(res)
                print(f"      dominance {lo:.1f}-{hi:.1f}: zero {res['n_rows']:6,} rows "
                      f"-> A loses {res['drop_a']:5.2f} dB, B loses {res['drop_b']:5.2f} dB",
                      flush=True)
        perturbs[key] = pb

    # the headline, in the terminal: what lambda alone costs B, and what
    # lambda plus collisions costs B
    print("")
    for lam in a.lambdas:
        cr, dr = runs[f"collide_lam{lam:g}"], runs[f"dense_lam{lam:g}"]
        print(f"  lambda {lam:<5g}  A-B gap  dense {dr['psnr_a']-dr['psnr_b']:+5.2f} dB   "
              f"colliding {cr['psnr_a']-cr['psnr_b']:+5.2f} dB   "
              f"B pays {dr['psnr_b']-cr['psnr_b']:5.2f} dB for the shared table, "
              f"A pays {dr['psnr_a']-cr['psnr_a']:5.2f}", flush=True)

    out = {"content": content, "runs": runs, "census": census,
           "census_dense": census_dense,
           "n_shared_hashed": n_shared_hashed, "n_entries": int(enc.n_entries),
           # the histogram, not just its summary, so --replot can rebuild the
           # figure from the json alone
           "dominance": {k: {"median": float(np.median(v)), "mean": float(v.mean()),
                             "frac_above_0.9": float((v > 0.9).mean()),
                             "n": int(v.size),
                             "hist": np.histogram(v, bins=60, range=(0, 1))[0].tolist()}
                         for k, v in dom.items()},
           "perturb": perturbs,
           "settings": {"downsample": a.downsample, "block": a.block,
                        "steps": a.steps, "batch": a.batch, "lr": a.lr,
                        "log2_small": a.log2_small, "log2_large": a.log2_large,
                        "image": f"{w}x{h}x{shape[2]}", "encoder": ENC}}
    jp = os.path.join(a.out, "collision_audit.json")
    json.dump(out, open(jp, "w"), indent=1)
    figure(out, ref.cpu().numpy(), mask_a.cpu().numpy(), a)
    print(f"\nwrote {jp}\nwrote {os.path.join(a.out, 'collision_audit.png')}")


def figure(out, ref, mask_a, a):
    lam = a.lambdas
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.4))
    for k, x in zip("abcdef", ax.ravel()):
        x.text(0.012, 0.985, k, transform=x.transAxes, fontsize=15, fontweight="bold",
               va="top", ha="left")
        x.tick_params(labelsize=9)

    # (a) the two regions
    v = ref.copy()
    v[~mask_a] *= 0.35
    ax[0, 0].imshow(np.clip(v, 0, 1))
    ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
    ax[0, 0].set_xlabel(f"A bright, B dimmed -- {a.block} px blocks; mean |grad I| "
                        f"{out['content']['detail_a']:.4f} vs "
                        f"{out['content']['detail_b']:.4f}", fontsize=9)

    # (b) the geometry of sharing, level by level
    ce, cd = out["census"], out["census_dense"]
    lv = [c["level"] for c in ce]
    ax[0, 1].bar(lv, [c["shared_frac"] * 100 for c in ce], 0.8,
                 color=["#9aa0a6" if c["dense"] else "#4da3ff" for c in ce],
                 label="small table (grey = still dense)")
    ax[0, 1].plot(lv, [c["shared_frac"] * 100 for c in cd], "o-", color="#111",
                  ms=3.5, lw=1.2, label="dense table: adjacency only")
    ax[0, 1].set_xlabel("level")
    ax[0, 1].set_ylabel("table rows touched by BOTH regions  (%)")
    ax[0, 1].legend(fontsize=8, loc="lower right")
    rr = [c["weight_ratio"] for c in ce]
    tw = ax[0, 1].twinx()
    tw.plot(lv, rr, "s-", color="#e5484d", ms=3, lw=1)
    tw.axhline(1.0, color="#e5484d", lw=0.7, ls=":")
    tw.set_ylim(0.9, 1.1)      # measured range is 0.996-1.003; an autoscaled
                               # axis magnifies that into apparent structure
    tw.set_ylabel("interpolation weight A / B on shared rows", color="#e5484d",
                  fontsize=9)
    tw.tick_params(labelsize=8, colors="#e5484d")

    # (c) where a shared row's gradient mass comes from
    ctr = np.linspace(0, 1, 61)[:-1] + 1 / 120
    for l, col in zip(lam, ("#2ea043", "#e5a23c", "#e5484d")):
        dd = out["dominance"][f"collide_lam{l:g}"]
        y = np.array(dd["hist"], dtype=float)
        ax[0, 2].step(ctr, y / y.sum() * 60, where="mid", lw=1.6, color=col,
                      label=f"lambda = {l:g}  (median {dd['median']:.2f})")
    ax[0, 2].axvline(0.5, color="#666", lw=0.8, ls=":")
    ax[0, 2].set_xlabel("share of the gradient mass arriving from A")
    ax[0, 2].set_ylabel("shared hashed rows (density)")
    ax[0, 2].legend(fontsize=8, loc="upper center")

    # (d) the outcome, per region
    for tag, ls in (("collide", "-"), ("dense", "--")):
        for reg, col in (("a", "#1f77b4"), ("b", "#d62728")):
            y = [out["runs"][f"{tag}_lam{l:g}"][f"psnr_{reg}"] for l in lam]
            ax[1, 0].plot(lam, y, ls, marker="o", ms=4, color=col,
                          label=f"{reg.upper()}, {'colliding' if tag=='collide' else 'dense'}")
    ax[1, 0].set_xscale("log"); ax[1, 0].invert_xaxis()
    ax[1, 0].set_xlabel("loss weight on region B  (lambda)")
    ax[1, 0].set_ylabel("PSNR of the region (dB)")
    ax[1, 0].legend(fontsize=8)

    # (e) what the collisions cost, region by region
    wdt = 0.35
    xs = np.arange(len(lam))
    for i, (reg, col) in enumerate((("a", "#1f77b4"), ("b", "#d62728"))):
        y = [out["runs"][f"dense_lam{l:g}"][f"psnr_{reg}"]
             - out["runs"][f"collide_lam{l:g}"][f"psnr_{reg}"] for l in lam]
        ax[1, 1].bar(xs + (i - 0.5) * wdt, y, wdt, color=col, label=f"region {reg.upper()}")
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels([f"{l:g}" for l in lam])
    ax[1, 1].set_xlabel("loss weight on region B  (lambda)")
    ax[1, 1].set_ylabel("dB lost to collisions  (dense minus colliding)")
    ax[1, 1].axhline(0, color="#444", lw=0.8)
    ax[1, 1].legend(fontsize=8)

    # (f) who does a shared row serve?  Damage share against mass share, which
    # is comparable across bins of very different size in a way that raw dB is
    # not: the bins below hold anywhere from 30 to 10,094 rows.
    ax[1, 2].plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--")
    for l, col in zip(lam, ("#2ea043", "#e5a23c", "#e5484d")):
        pb = out["perturb"][f"collide_lam{l:g}"]
        xs = [(p["lo"] + min(p["hi"], 1.0)) / 2 for p in pb]
        ys = [p["drop_a"] / max(1e-9, p["drop_a"] + p["drop_b"]) for p in pb]
        ax[1, 2].plot(xs, ys, "o-", color=col, ms=5, lw=1.4, label=f"lambda = {l:g}")
        for x, y, p in zip(xs, ys, pb):
            ax[1, 2].annotate(f"{p['n_rows']:,}", (x, y), textcoords="offset points",
                              xytext=(4, -10), fontsize=7, color=col)
    ax[1, 2].set_xlim(0, 1); ax[1, 2].set_ylim(0, 1)
    ax[1, 2].set_xlabel("A's share of the row's gradient mass  (bin centre; "
                        "labels = rows zeroed)")
    ax[1, 2].set_ylabel("A's share of the dB lost when those rows are zeroed")
    ax[1, 2].legend(fontsize=8, loc="lower right")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.text(0.5, 0.012,
             "Instant-NGP collision audit: two interleaved regions of one painting, "
             "region B fitted at loss weight lambda.  Every number measured by "
             "scripts/collision_audit.py.", ha="center", fontsize=9, color="#444")
    fig.savefig(os.path.join(a.out, "collision_audit.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
