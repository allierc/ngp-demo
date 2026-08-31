#!/usr/bin/env python
"""Fit a scalar field f(x, y, t) from a zarr with the hash encoding, in a browser.

    python scripts/gui_scalar_time.py                     # http://localhost:8025
    python scripts/gui_scalar_time.py --zarr-glob "/path/to/*/field.zarr"

Written for `Plexus/prototype/graphcast/log/toy2d*/field.zarr`, which is two PDEs
laid on top of each other, and that is exactly why it is worth fitting: the two
components want opposite things from one representation.

Measured on that store before any of this was written:

  coarse `u`, 256^2      0.0% of its energy above 32 cycles across the frame;
                         lag-1 autocorrelation 0.998, and still correlated with
                         frame 0 at 0.5 eighteen frames later.
  fine `v`, 1024^2       73.6% of its energy ABOVE 32 cycles, 15 px per cycle
                         inside a disc, on 15.4% of the pixels; lag-1
                         autocorrelation 0.829, and past 0.5 after ONE frame.

So the sum needs about 4-8 px per finest cell in space, and a time axis at the
frame spacing rather than under it -- the opposite end of ngp-demo's stage 2,
where capping t below the frame rate was the whole result.  Here the data really
does have per-frame content, and a cap under 201 cells throws it away.  The
`field` selector fits either component alone or the sum, so that trade can be
seen rather than argued about.

The encoder is settled the way every page here settles it: levels L, max entries
per level T, and px per finest cell, with N_max following and b derived.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import NGPField
from ngp.utils import pixel_centers, render
from ngp.webui import ABOUT_HTML, CSS, cmap_png, field_png, signed_rgb, png_data_uri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The generator writes one store per run and names the directory after it, so a
# hard-coded path goes stale the next time it is regenerated -- it already did
# once while this was written.  Each store carries a summary.json saying which
# part it holds, which is what the dataset toggle is built from rather than the
# directory name.
ZARR_GLOB = "/workspace/Plexus/prototype/graphcast/log/toy2d*/field.zarr"


def datasets(pattern=None):
    """{'coarse'|'fine'|'sum': store path} for whichever runs are on disk.

    `coarse` holds only u (256^2, the slow wave), `fine` only v (1024^2, the
    Kuramoto discs), and the run whose summary says "both" holds u and v and is
    offered as `sum`.  Newest wins if a part was generated more than once.
    """
    out = {}
    for d in sorted(glob.glob(pattern or ZARR_GLOB), key=os.path.getmtime):
        part, name = None, os.path.basename(os.path.dirname(d))
        try:
            with open(os.path.join(os.path.dirname(d), "summary.json")) as f:
                part = json.load(f).get("part")
        except OSError:
            pass
        if part is None:                       # no summary: fall back on the name
            part = ("coarse" if "coarse" in name else
                    "fine" if "fine" in name else "both")
        out["sum" if part == "both" else part] = d
    return out


DATASETS = datasets()
DEFAULT_ZARR = DATASETS.get("sum") or next(iter(DATASETS.values()), "")

LEVEL_LUT_MAX = 20
ERROR_LUT_MAX = 0.20                 # signed, fixed: the field itself is +-1

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "note": "", "stamp": 0, "frames": [],
       "per_frame": [], "levels": {}, "levels_live": 0.0}
LOCK = threading.Lock()
STOP = threading.Event()
DATA = {}


# ----------------------------------------------------------------- the data


def load_field(which, down, device, stores=None):
    """(T, H, W) on the device, plus the frame count and the spatial size.

    The dataset toggle picks the STORE, and each store defines its own field:
    `coarse` is u alone, `fine` is v alone, and `sum` is the run holding both,
    with u carried up to v's grid and added.  Bilinear, not nearest, for the
    upsampling: u has 0.0% of its energy above 32 cycles on a 256 grid, so
    carrying it to 1024 invents nothing.
    """
    stores = stores or DATASETS
    path = stores.get(which)
    if not path:
        raise FileNotFoundError(
            f"no {which} dataset under {ZARR_GLOB} -- found "
            f"{sorted(stores) or 'nothing'}")
    key = (path, which, down, str(device))
    if key in DATA:
        return DATA[key]
    import zarr
    g = zarr.open(path, mode="r")
    have = set(g.group_keys())

    def grid(k):
        if k not in have:
            raise KeyError(f"{os.path.basename(os.path.dirname(path))} has no "
                           f"{k}/grid, only {sorted(have)}")
        return torch.from_numpy(np.asarray(g[f"{k}/grid"][:, 0])).to(device)

    if which == "coarse":
        a = grid("u")
    elif which == "fine":
        a = grid("v")
    else:
        v = grid("v")
        u = F.interpolate(grid("u")[:, None], size=v.shape[-2:], mode="bilinear",
                          align_corners=False)[:, 0]
        a = u + v
    if down > 1:
        a = F.avg_pool2d(a[:, None], down)[:, 0]
    DATA[key] = (a.contiguous(), a.shape[0], a.shape[1], a.shape[2], path)
    return DATA[key]


@torch.no_grad()
def sample_field(vol, xyt):
    """Trilinear lookup into (T, H, W) at xyt in [0,1]^3 -> (N,).

    Minus half a voxel on every axis, for the reason ngp/utils.BilinearImage now
    carries: sample j covers [j/n, (j+1)/n) and sits at its centre, and without
    the shift a query at a sample's own centre returns the average of it and its
    neighbour -- a half-voxel blur of the whole target.
    """
    T, H, W = vol.shape
    size = torch.tensor([W, H, T], device=xyt.device, dtype=xyt.dtype)
    p = (xyt * size - 0.5).clamp(min=torch.zeros(3, device=xyt.device),
                                 max=size - 1)
    i0 = torch.floor(p)
    f = (p - i0).unsqueeze(-1)
    i0 = i0.long()
    x0, y0, t0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    t1 = (t0 + 1).clamp(max=T - 1)
    fx, fy, ft = f[:, 0, 0], f[:, 1, 0], f[:, 2, 0]
    c00 = vol[t0, y0, x0] * (1 - fx) + vol[t0, y0, x1] * fx
    c01 = vol[t0, y1, x0] * (1 - fx) + vol[t0, y1, x1] * fx
    c10 = vol[t1, y0, x0] * (1 - fx) + vol[t1, y0, x1] * fx
    c11 = vol[t1, y1, x0] * (1 - fx) + vol[t1, y1, x1] * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - ft) + c1 * ft


def frame_coords(h, w, t, device):
    xy = pixel_centers(h, w, device)
    return torch.cat([xy, torch.full((xy.shape[0], 1), float(t), device=device)], 1)


# ------------------------------------------------------------------- the fit


def build(p, w, h, n_frames):
    n_lv = int(p["n_levels"])
    n_min = (8, 8, 2)
    ppc = max(1.0, float(p["px_per_finest_cell"]))
    n_max = (max(9, round(w / ppc)), max(9, round(h / ppc)),
             max(3, int(p["time_cells"])))
    scale = tuple(math.exp((math.log(mx) - math.log(mn)) / max(1, n_lv - 1))
                  for mn, mx in zip(n_min, n_max))
    model = NGPField(
        n_input_dims=3, n_output_dims=1, n_neurons=64, n_hidden_layers=2,
        activation="gelu", output_activation="none",
        n_levels=n_lv, n_features_per_level=2,
        log2_hashmap_size=int(p["log2_hashmap_size"]),
        base_resolution=n_min, per_level_scale=scale, max_resolution=n_max,
        # linear on t, smoothstep in space: on an axis whose cells line up with
        # the sampled frames, smoothstep forces df/dt to zero at every one.
        interpolation=("smoothstep", "smoothstep", "linear"))
    return model


@torch.no_grad()
def level_map_at(model, h, w, t, device, block_px=64, sub=4, thresh=0.08):
    """Per block, the finest level contributing at time t.

    This dataset is the one where the panel has something to find: the fine
    component lives on 15% of the pixels and the coarse one everywhere, so the
    discs should read finer than the background if the encoder is spending
    itself where the data is.
    """
    enc = model.encoding
    hs, ws = max(8, h // sub), max(8, w // sub)
    xyt = frame_coords(hs, ws, t, device)
    bs = max(2, block_px // sub)
    prev, deltas = None, []
    for k in range(enc.n_levels + 1):
        enc.set_level_window(float(k))
        out = model(xyt).reshape(hs, ws)
        if prev is not None:
            deltas.append((out - prev).abs())
        prev = out
    enc.set_level_window(float(enc.n_levels))
    D = torch.stack(deltas)
    nb = F.avg_pool2d(D[None], bs, stride=bs, ceil_mode=True)[0]
    peak = nb.amax(0, keepdim=True)
    alive = peak > 0.02 * float(nb.max())
    sig = (nb > thresh * peak) & alive
    lev = torch.arange(nb.shape[0], device=nb.device)[:, None, None].expand_as(nb)
    dom = torch.where(sig, lev, torch.zeros_like(lev)).amax(0)
    dom = torch.where(sig.any(0), dom, nb.argmax(0)).cpu().numpy()
    blocks = []
    for j in range(dom.shape[0]):
        for i in range(dom.shape[1]):
            l = int(dom[j, i])
            blocks.append({"x": i * block_px, "y": j * block_px,
                           "w": min(block_px, w - i * block_px),
                           "h": min(block_px, h - j * block_px),
                           "level": l,
                           "cell_px": round(w / enc.resolutions[l][0], 2)})
    return {"blocks": blocks, "w": w, "h": h, "n_levels": enc.n_levels}


def train_job(p, device):
    try:
        down = max(1, int(p["downsample"]))
        vol, n_frames, h, w, store = load_field(p["field"], down, device)
        vmax = float(vol.abs().max())
        torch.manual_seed(0)
        model = build(p, w, h, n_frames).to(device)
        enc = model.encoding
        n_enc, n_mlp = model.n_parameters()
        # Held-out frames: every k-th one is never sampled, and the score on
        # them is what says whether the fit interpolates in time or memorises.
        hold = int(p["holdout"])
        held = set(range(1, n_frames - 1, hold)) if hold > 1 else set()
        train_t = np.array([i for i in range(n_frames) if i not in held],
                           dtype=np.float32) / max(1, n_frames - 1)
        train_t = torch.from_numpy(train_t).to(device)
        n_values = n_frames * h * w
        compression = n_values / max(1, n_enc + n_mlp)
        picks = [0, n_frames // 2, n_frames - 1]
        note = (f"{p['field']} dataset "
                f"({os.path.basename(os.path.dirname(store))}), "
                f"{n_frames} frames of {w}x{h}, "
                f"values +-{vmax:.2f}; "
                f"{enc.resolutions[0][0]}..{enc.resolutions[-1][0]} cells in "
                f"space ({w / enc.resolutions[-1][0]:.1f} px per finest cell), "
                f"{enc.resolutions[-1][2]} along t of {n_frames} frames"
                + (f"; {len(held)} frames held out" if held else ""))
        print(f"[run] {note}", flush=True)
        print(f"[params] {n_enc + n_mlp:,} total = {n_enc:,} in the hash table "
              f"({enc.table.shape[0]:,} entries x {enc.table.shape[1]} features) "
              f"+ {n_mlp:,} in the decoder", flush=True)
        print(f"[size  ] {n_enc + n_mlp:,} parameters against {n_values:,} stored "
              f"values = {compression:.1f}x compression "
              f"({100 / compression:.1f}% of the field)", flush=True)
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], per_frame=[], levels={}, note=note, frames=picks,
                       metrics={"n_parameters": n_enc + n_mlp, "n_table": n_enc,
                                "n_values": n_values, "compression": compression,
                                "n_decoder": n_mlp, "n_levels_total": enc.n_levels,
                                "n_frames": n_frames, "vmax": vmax,
                                "n_held": len(held)},
                       images={f"target{i}": field_png(
                           vol[t].cpu().numpy(), vmax)
                           for i, t in enumerate(picks)},
                       stamp=JOB["stamp"] + 1)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps, batch = int(p["steps"]), int(p["batch"])
        every = max(1, steps // 30)
        t0 = time.perf_counter()
        for step in range(steps + 1):
            if STOP.is_set():
                break
            if int(p["coarse_to_fine"]):
                a = 4 + (enc.n_levels - 4) * min(1.0, step / max(1, steps * 0.5))
                enc.set_level_window(a)
                JOB["levels_live"] = round(float(a), 2)
            else:
                JOB["levels_live"] = float(enc.n_levels)
            xy = torch.rand(batch, 2, device=device)
            ti = train_t[torch.randint(len(train_t), (batch,), device=device)]
            xyt = torch.cat([xy, ti[:, None]], 1)
            pred = model(xyt)[:, 0]
            loss = ((pred - sample_field(vol, xyt)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % every == 0 or step == steps:
                with torch.no_grad():
                    imgs, errs, lv = {}, [], {}
                    for i, fr in enumerate(picks):
                        c = frame_coords(h, w, fr / max(1, n_frames - 1), device)
                        fit = render(model, c, (h, w))
                        d = (fit - vol[fr])
                        imgs[f"fit{i}"] = field_png(fit.cpu().numpy(), vmax)
                        imgs[f"err{i}"] = png_data_uri(
                            signed_rgb(d.cpu().numpy(), ERROR_LUT_MAX))
                        errs.append(float(d.pow(2).mean()))
                        lv[str(i)] = level_map_at(model, h, w,
                                                  fr / max(1, n_frames - 1), device)
                    # every frame, on a coarse grid: cheap, and the only way to
                    # see a fit that is good at the ends and lost in between
                    hs, ws = h // 4, w // 4
                    per = []
                    for fr in range(0, n_frames, max(1, n_frames // 60)):
                        c = frame_coords(hs, ws, fr / max(1, n_frames - 1), device)
                        f2 = render(model, c, (hs, ws))
                        ref = F.avg_pool2d(vol[fr][None, None], 4)[0, 0]
                        mse = float((f2 - ref).pow(2).mean())
                        per.append({"t": fr, "psnr": psnr_db(mse, vmax),
                                    "held": fr in held})
                mse = float(np.mean(errs))
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = time.perf_counter() - t0
                    JOB["curve"].append({"step": step, "psnr": psnr_db(mse, vmax),
                                         "loss": loss.detach().item()})
                    JOB["per_frame"] = per
                    JOB["metrics"] = {**JOB["metrics"], "psnr": psnr_db(mse, vmax),
                                      "psnr_held": (float(np.mean(
                                          [10 ** (-q["psnr"] / 10) for q in per
                                           if q["held"]])) if held else None)}
                    if held:
                        JOB["metrics"]["psnr_held"] = -10 * math.log10(
                            max(1e-12, JOB["metrics"]["psnr_held"]))
                    JOB["images"].update(imgs)
                    JOB["levels"] = lv
                    JOB["stamp"] += 1
    except Exception as e:
        print(f"[run] failed: {type(e).__name__}: {e}", flush=True)
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1
            m = JOB["metrics"]
        print(f"[done ] {JOB['seconds']:.1f}s  psnr {m.get('psnr', 0):.2f} dB"
              + (f"  held-out {m['psnr_held']:.2f} dB" if m.get("psnr_held")
                 else ""), flush=True)


def psnr_db(mse, vmax):
    """Peak signal to noise against the field's own peak-to-peak (2*vmax)."""
    return float(10 * math.log10((2 * vmax) ** 2 / max(mse, 1e-12)))


KNOBS = {
    "field": [
        {"name": "n_levels", "label": "levels L", "min": 2, "max": 16,
         "default": 10, "step": 1},
        {"name": "log2_hashmap_size", "label": "max entries per level T",
         "min": 10, "max": 24, "default": 20, "step": 1, "pow2": True},
        # 4 px per cell, because the fine component measures 15 px per cycle
        # inside a disc and two cells per cycle is the floor.
        {"name": "px_per_finest_cell", "label": "px per finest cell",
         "choices": [1, 2, 4, 8, 16, 32, 64], "default": 4},
        # AT the frame spacing, not under it. The fine component decorrelates in
        # one frame, so there is per-frame content and a coarser t axis throws it
        # away -- the opposite of stage 2, where the field was smooth in time and
        # a fine t axis memorised frames instead of interpolating.
        {"name": "time_cells", "label": "cells along t", "min": 4, "max": 256,
         "default": 200, "step": 4},
    ],
    "train": [
        {"name": "lr", "label": "learning rate", "min": 1e-4, "max": 1e-1,
         "default": 1e-2, "step": 1e-4, "log": True},
        {"name": "steps", "label": "iterations", "min": 100, "max": 8000,
         "default": 1500, "step": 100},
        {"name": "batch", "label": "batch size (random x, y, t)", "min": 4096,
         "max": 1048576, "default": 262144, "step": 4096},
    ],
}
DEFAULTS = {k["name"]: k.get("default") for g in KNOBS.values() for k in g}
DEFAULTS.update(field="sum" if "sum" in DATASETS else next(iter(DATASETS), "sum"),
                downsample=2, coarse_to_fine=0, holdout=1)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>a scalar field in time</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>two PDEs, one encoder</h1>
<p class="sub">f(x, y, t) &rarr; a scalar, fitted from a zarr. The runs hold a
<b>coarse slow wave</b> (2 cycles across the frame, correlated with itself for 18
frames) and a <b>fast Kuramoto on four discs</b> (15 px per cycle, on 15% of the
pixels, decorrelating in <b>one</b> frame). They want opposite settings from the
same encoder, and the generator writes all three as separate runs &mdash; the
<b>dataset</b> toggle picks which one to fit.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is an ngp?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="setup" id="setup"></div>
<div class="row equal" style="margin-top:14px">
  <div class="panel"><canvas id="c_target0" width="300" height="300"></canvas>
    <div class="cap">target &mdash; first frame</div></div>
  <div class="panel"><canvas id="c_target1" width="300" height="300"></canvas>
    <div class="cap">middle</div></div>
  <div class="panel"><canvas id="c_target2" width="300" height="300"></canvas>
    <div class="cap">last</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_fit0" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit1" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_fit2" width="300" height="300"></canvas>
    <div class="cap">fit</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_err0" width="300" height="300"></canvas>
    <div class="cap">fit &minus; target &mdash; blue/red, fixed &plusmn;__ERRMAX__</div></div>
  <div class="panel"><canvas id="c_err1" width="300" height="300"></canvas>
    <div class="cap">error</div></div>
  <div class="panel"><canvas id="c_err2" width="300" height="300"></canvas>
    <div class="cap">error</div></div>
</div>
<div class="row equal" style="margin-top:8px">
  <div class="panel"><canvas id="c_lev0" width="300" height="300"></canvas>
    <div class="cap">finest level contributing, per 64 px block</div></div>
  <div class="panel"><canvas id="c_lev1" width="300" height="300"></canvas>
    <div class="cap">levels</div></div>
  <div class="panel"><canvas id="c_lev2" width="300" height="300"></canvas>
    <div class="cap">levels</div></div>
</div>
<div id="levlegend" class="note"></div>
<div class="row" style="margin-top:14px">
  <div class="panel"><canvas id="c_curve" width="470" height="300"></canvas>
    <div class="cap">psnr against iteration</div></div>
  <div class="panel"><canvas id="c_frames" width="470" height="300"></canvas>
    <div class="cap">psnr per frame, on a 4x coarser grid &mdash; held-out in amber</div></div>
</div>
<div class="stats" id="stats"></div>
<div class="modal" id="about" onclick="if(event.target===this)closeAbout()">
  <div class="sheet">__ABOUT__</div></div>
</div><script>
const KNOBS=__KNOBS__, DEF=__DEF__, LUTMAX=__LUTMAX__;
const DATASETS=__DATASETS__;
const knob=Object.assign({}, DEF);
const IMG={}; let POLL=null, SEEN=false, LAST=-1, LASTLEV={};
window.addEventListener("unhandledrejection", e=>
  fetch("/api/clienterror?msg="+encodeURIComponent(String(e.reason))));

const C=document.getElementById("controls");
function seg(title, opts, key){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=title;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{ const [txt,val]=Array.isArray(o)?o:[String(o),o];
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", knob[key]===val);
    b.onclick=()=>{ knob[key]=val;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); setup(); };
    s.appendChild(b); });
  g.append(l,s); C.appendChild(g);
}
seg("dataset", DATASETS.map(k=>[
      {sum:"sum (u+v)", coarse:"coarse wave u", fine:"fine kuramoto v"}[k], k]),
    "field");
seg("downsample", [1,2,4], "downsample");
seg("coarse to fine", [["off",0],["on",1]], "coarse_to_fine");
seg("hold out every", [["none",1],["4th frame",4],["8th",8]], "holdout");

const K=document.getElementById("knobs");
function panel(title, list){
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  K.appendChild(t);
  list.forEach(p=>{
    if(p.choices){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML=`<span>${p.label}</span>`;
      const s=document.createElement("div"); s.className="seg";
      p.choices.forEach(o=>{ const b=document.createElement("button");
        b.textContent=String(o);
        b.setAttribute("aria-pressed", knob[p.name]===o);
        b.onclick=()=>{ knob[p.name]=o;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b));
          setup(); };
        s.appendChild(b); });
      d.append(lab,s); K.appendChild(d); return;
    }
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.pow2?`2^${Math.round(v)} = ${Math.pow(2,Math.round(v)).toLocaleString()}`
                       :(p.log?(+v).toExponential(1)
                              :(p.step<1?(+v).toFixed(3):String(Math.round(v))));
    const raw=v=>p.log?Math.log10(v):v, un=r=>p.log?Math.pow(10,r):+r;
    val.textContent=fmt(knob[p.name]);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=raw(p.min); r.max=raw(p.max); r.step=p.log?0.02:p.step;
    r.value=raw(knob[p.name]);
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${fmt(p.min)}</span><span>${fmt(p.max)}</span>`;
    r.oninput=()=>{ knob[p.name]=un(r.value); val.textContent=fmt(knob[p.name]);
                    setup(); };
    d.append(lab,r,ends); K.appendChild(d);
  });
}
panel("the encoder", KNOBS.field);
panel("training", KNOBS.train);

let PARAMS="";
function setup(){
  const px=knob.px_per_finest_cell*knob.downsample;
  document.getElementById("setup").innerHTML=
    `<span class="dim">the</span> <b>${knob.field}</b> <span class="dim">dataset at</span> `
   +`<b>1/${knob.downsample}</b> <span class="dim">resolution,</span> `
   +`<b>${knob.px_per_finest_cell}</b> <span class="dim">px per finest cell</span> `
   +`<span class="dim">(${px} px of the original),</span> `
   +`<b>${knob.time_cells}</b> <span class="dim">cells along t</span>` + PARAMS;
}
setup();

async function startRun(){
  SEEN=false;
  await fetch("/api/start?"+new URLSearchParams(knob));
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 600); poll();
}
document.getElementById("run").onclick=startRun;
document.getElementById("stop").onclick=()=>fetch("/api/stop");
function openAbout(){ document.getElementById("about").classList.add("on"); }
function closeAbout(){ document.getElementById("about").classList.remove("on"); }

function drawImg(id, src){
  if(!src) return;
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  g.imageSmoothingEnabled=false;
  g.drawImage(im,(cv.width-im.width*s)/2,(cv.height-im.height*s)/2,
              im.width*s, im.height*s);
}
function levColor(t){
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
function drawLevels(id, bk, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!bk||!bk.blocks) return;
  const s=Math.min(cv.width/bk.w, cv.height/bk.h);
  const ox=(cv.width-bk.w*s)/2, oy=(cv.height-bk.h*s)/2;
  const im=IMG[src];
  if(im){ g.globalAlpha=0.35; g.imageSmoothingEnabled=false;
          g.drawImage(im,ox,oy,bk.w*s,bk.h*s); g.globalAlpha=1; }
  bk.blocks.forEach(b=>{
    const col=levColor(b.level/LUTMAX), cw=b.cell_px*s;
    g.strokeStyle=col; g.lineWidth=1.0;
    if(cw<3){ g.fillStyle=col; g.globalAlpha=0.42;
      g.fillRect(ox+b.x*s, oy+b.y*s, b.w*s, b.h*s); g.globalAlpha=1; }
    else {
      const c=b.cell_px;
      for(let x=Math.ceil(b.x/c)*c; x<=b.x+b.w; x+=c){ g.beginPath();
        g.moveTo(ox+x*s, oy+b.y*s); g.lineTo(ox+x*s, oy+(b.y+b.h)*s); g.stroke(); }
      for(let y=Math.ceil(b.y/c)*c; y<=b.y+b.h; y+=c){ g.beginPath();
        g.moveTo(ox+b.x*s, oy+y*s); g.lineTo(ox+(b.x+b.w)*s, oy+y*s); g.stroke(); }
    }
  });
  const used=[...new Set(bk.blocks.map(b=>b.level))].sort((a,b)=>a-b);
  let bar=""; for(let i=0;i<=LUTMAX;i++)
    bar+=`<span style="display:inline-block;width:14px;height:9px;`
        +`background:${levColor(i/LUTMAX)}"></span>`;
  document.getElementById("levlegend").innerHTML=
    `<div>level colour scale, fixed 0&ndash;${LUTMAX}</div>`
   +`<div style="line-height:0;margin:3px 0">${bar}</div>`
   +`<div>levels contributing: ` + used.map(l=>{
       const b=bk.blocks.find(q=>q.level===l);
       return `<span style="color:${levColor(l/LUTMAX)}">&#9632; ${l} `
             +`(${b.cell_px} px cells)</span>`; }).join(" &nbsp; ")
   +`. The fine component sits on 15% of the pixels, so a fit spending itself `
   +`where the data is should read finer on the discs than off them.</div>`;
}
function axes(g,cv,pad,xs,ys,xlab,ylab){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  g.strokeStyle="#333"; g.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad.t+(cv.height-pad.t-pad.b)*i/4;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(cv.width-pad.r,y); g.stroke(); }
  g.fillStyle="#9a9a9a"; g.font="10px sans-serif"; g.textAlign="center";
  g.fillText(xlab, (pad.l+cv.width-pad.r)/2, cv.height-6);
}
function drawCurve(cur){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const pad={l:44,r:10,t:12,b:22};
  axes(g,cv,pad,null,null,"iteration");
  if(!cur||!cur.length) return;
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const lo=Math.min(...cur.map(q=>q.psnr)), hi=Math.max(...cur.map(q=>q.psnr));
  const X=i=>pad.l+W*i/Math.max(1,cur.length-1);
  const Y=v=>pad.t+H*(1-(v-lo)/Math.max(1e-6,hi-lo));
  g.strokeStyle="#fff"; g.lineWidth=1.6; g.beginPath();
  cur.forEach((q,i)=> i?g.lineTo(X(i),Y(q.psnr)):g.moveTo(X(i),Y(q.psnr)));
  g.stroke();
  g.fillStyle="#9a9a9a"; g.textAlign="right"; g.font="10px sans-serif";
  g.fillText(hi.toFixed(1), pad.l-4, pad.t+8);
  g.fillText(lo.toFixed(1), pad.l-4, pad.t+H);
}
function drawFrames(pf){
  const cv=document.getElementById("c_frames"), g=cv.getContext("2d");
  const pad={l:44,r:10,t:12,b:22};
  axes(g,cv,pad,null,null,"frame");
  if(!pf||!pf.length) return;
  const W=cv.width-pad.l-pad.r, H=cv.height-pad.t-pad.b;
  const lo=Math.min(...pf.map(q=>q.psnr)), hi=Math.max(...pf.map(q=>q.psnr));
  const tmax=Math.max(...pf.map(q=>q.t));
  const X=t=>pad.l+W*t/Math.max(1,tmax);
  const Y=v=>pad.t+H*(1-(v-lo)/Math.max(1e-6,hi-lo));
  g.strokeStyle="#4da3ff"; g.lineWidth=1.4; g.beginPath();
  pf.forEach((q,i)=> i?g.lineTo(X(q.t),Y(q.psnr)):g.moveTo(X(q.t),Y(q.psnr)));
  g.stroke();
  g.fillStyle="#e5a23c";
  pf.filter(q=>q.held).forEach(q=>
    g.fillRect(X(q.t)-1.5, Y(q.psnr)-1.5, 3, 3));
  g.fillStyle="#9a9a9a"; g.textAlign="right"; g.font="10px sans-serif";
  g.fillText(hi.toFixed(1), pad.l-4, pad.t+8);
  g.fillText(lo.toFixed(1), pad.l-4, pad.t+H);
}
async function poll(){
  const r=await (await fetch("/api/state")).json();
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";
  const m=r.metrics||{};
  if(m.n_parameters!==undefined){
    PARAMS=` <span class="dim">&middot;</span> <span style="color:#fff">`
      +`${m.n_parameters.toLocaleString()} parameters, `
      +`${m.n_table.toLocaleString()} in the hash table, `
      +`<b>${m.compression.toFixed(1)}x</b> compression against the `
      +`${m.n_values.toLocaleString()} stored values</span>`;
    setup();
  }
  if(r.stamp!==LAST){
    LAST=r.stamp; LASTLEV=r.levels||{};
    for(let i=0;i<3;i++){
      drawImg("c_target"+i, (r.images||{})["target"+i]);
      drawImg("c_fit"+i, (r.images||{})["fit"+i]);
      drawImg("c_err"+i, (r.images||{})["err"+i]);
      drawLevels("c_lev"+i, LASTLEV[String(i)], "c_target"+i);
    }
    drawCurve(r.curve); drawFrames(r.per_frame);
    document.getElementById("stats").innerHTML = m.psnr===undefined ? "press run"
      : `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `
       +`${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; levels live `
       +`<b>${r.levels_live}</b> of ${m.n_levels_total}<br>`
       +`psnr over the three drawn frames, at full resolution `
       +`<b>${m.psnr.toFixed(2)}</b> dB`
       + (m.psnr_held ? ` &nbsp;&middot;&nbsp; on the <b>${m.n_held}</b> `
                       +`held-out frames <b>${m.psnr_held.toFixed(2)}</b> dB` : "")
       +`<br><span style="color:#7a7a7a">${r.note}</span>`;
  }
  if(r.running) SEEN=true;
  if(SEEN && !r.running && POLL){ clearInterval(POLL); POLL=null; }
}
poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    device = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            page = (PAGE.replace("__CSS__", CSS)
                        .replace("__ABOUT__", ABOUT_HTML)
                        .replace("__KNOBS__", json.dumps(KNOBS))
                        .replace("__DEF__", json.dumps(DEFAULTS))
                        .replace("__DATASETS__",
                                 json.dumps([k for k in ("sum", "coarse", "fine")
                                             if k in DATASETS]))
                        .replace("__LUTMAX__", str(LEVEL_LUT_MAX))
                        .replace("__ERRMAX__", f"{ERROR_LUT_MAX:g}"))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            with LOCK:
                JOB["running"] = True
                JOB["stamp"] += 1
            p = dict(DEFAULTS)
            for k, v in q.items():
                p[k] = float(v) if _isnum(v) else v
            threading.Thread(target=train_job, args=(p, self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            STOP.set()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clienterror":
            print(f"[client] {q.get('msg', '')}", flush=True)
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/state":
            with LOCK:
                return self._send(json.dumps(JOB), "application/json")
        self.send_error(404)


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8025)
    ap.add_argument("--zarr-glob", default=ZARR_GLOB,
                    help="where to look for the toy2d runs")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    globals()["ZARR_GLOB"] = a.zarr_glob
    DATASETS.clear()
    DATASETS.update(datasets(a.zarr_glob))
    if not DATASETS:
        sys.exit(f"no field.zarr found under {a.zarr_glob}")
    DEFAULTS["field"] = "sum" if "sum" in DATASETS else sorted(DATASETS)[0]
    Handler.device = torch.device(a.device)
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except OSError as e:
        if e.errno == 98:
            sys.exit(f"port {a.port} is already in use")
        raise
    for k in ("sum", "coarse", "fine"):
        if k in DATASETS:
            print(f"  {k:7s} {os.path.basename(os.path.dirname(DATASETS[k]))}")
    print(f"http://localhost:{a.port}   (device {a.device})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
