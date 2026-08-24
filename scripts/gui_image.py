#!/usr/bin/env python
"""Browser GUI for stage 1: fit the painting, and see what each encoder setting costs.

    python scripts/gui_image.py           # http://localhost:8022

Every knob of the hash encoding is a control, and the page shows what that
setting actually builds *before* you spend a minute training it: the resolution
ladder level by level, which levels have to hash, the table size, and how many
pixels one cell of the finest level covers. That last number is the one worth
watching -- refining below the pixel spacing costs parameters, barely moves
PSNR, and shows up later as noise in any derivative you take.

Finished runs stay on the curve panel, so successive settings can be compared
on quality against wall-clock and against parameter count rather than one at a
time.
"""

from __future__ import annotations

import argparse
import json
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
from ngp.utils import BilinearImage, pixel_centers, psnr, read_image, render
from ngp.webui import CSS, cmap_png, gray_png

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(ROOT, "assets/girl_with_a_pearl_earring.jpg")
# Level colours come from a fixed 0..20 scale rather than from each run's own
# range, so a colour means the same level whatever L is set to and two runs can
# be compared by eye.
LEVEL_LUT_MAX = 20
# The error panel is on a fixed scale too: auto-scaling to each refresh's own
# 99th percentile made the panel brighten as the fit improved, which reads as
# the opposite of what happened.
ERROR_LUT_MAX = 0.10

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "ladder": [], "note": "", "stamp": 0,
       "history": [], "blocks": {}}
LOCK = threading.Lock()
STOP = threading.Event()
IMAGES = {}                          # downsample factor -> (tensor, target, coords)


def get_image(path, down, device):
    # Keyed by device as well: /api/preview asks for the shape on the CPU, and a
    # cache hit from that would hand the CUDA training loop CPU tensors.
    key = (path, down, str(device))
    if key in IMAGES:
        return IMAGES[key]
    ref = read_image(path)
    t = torch.from_numpy(ref).to(device)
    if down > 1:
        t = F.avg_pool2d(t.permute(2, 0, 1)[None], down)[0].permute(1, 2, 0)
    arr = t.cpu().numpy()
    target = BilinearImage(arr, device)
    coords = pixel_centers(arr.shape[0], arr.shape[1], device)
    IMAGES[key] = (t, target, coords, arr.shape)
    return IMAGES[key]


SHAPES = {}


def image_shape(path, down):
    """(h, w, c) after downsampling, without building any tensor."""
    key = (path, down)
    if key not in SHAPES:
        h, w, c = read_image(path).shape
        SHAPES[key] = (h // down, w // down, c)
    return SHAPES[key]


def build(p, shape):
    """The NGPField a given set of controls asks for, plus its resolution ladder."""
    h, w, c = shape
    max_res = (w, h) if int(p["max_resolution"]) <= 0 else int(p["max_resolution"])
    kwargs = dict(
        n_input_dims=2, n_output_dims=c,
        n_neurons=int(p["n_neurons"]), n_hidden_layers=int(p["n_hidden_layers"]),
        activation=p["activation"], output_activation="sigmoid",
        n_levels=int(p["n_levels"]), n_features_per_level=int(p["n_features"]),
        log2_hashmap_size=int(p["log2_hashmap_size"]),
        base_resolution=int(p["base_resolution"]),
        per_level_scale=float(p["per_level_scale"]),
        max_resolution=max_res, interpolation=p["interpolation"])
    model = NGPField(**kwargs)
    enc = model.encoding
    ladder = [{"level": i, "rx": enc.resolutions[i][0], "ry": enc.resolutions[i][1],
               "dense": bool(enc.dense[i]),
               "px": round(w / enc.resolutions[i][0], 2)}
              for i in range(enc.n_levels)]
    return model, ladder


def describe(model, shape):
    h, w, c = shape
    n_enc, n_mlp = model.n_parameters()
    enc = model.encoding
    return {"n_enc": n_enc, "n_mlp": n_mlp, "n_total": n_enc + n_mlp,
            "n_values": h * w * c,
            "fraction_of_values": (n_enc + n_mlp) / (h * w * c),
            "hashed_levels": sum(1 for d in enc.dense if not d),
            "finest_px_per_cell": w / enc.resolutions[-1][0],
            "width": w, "height": h}


@torch.no_grad()
def level_maps(model, shape, device, block_px=64, sub=3):
    """Where each resolution level does its work.

    Renders the image with levels 0..k enabled for every k and differences
    consecutive renders, so `deltas[l]` is how much the picture changes when
    level l is released.  The decoder is nonlinear, so this is the marginal
    effect of a level given the coarser ones, not a term of a linear
    decomposition -- but that marginal is exactly what "this level is doing the
    work here" means.

    Returns a per-pixel effective level (amplitude-weighted mean) and, per
    block, the level that dominates it, which is what the overlay draws cells
    for.

    Evaluated on a grid `sub` times coarser than the image: this costs L+1 full
    renders, and at full resolution that is far more than the fit itself, which
    would make the panel affordable only once at the end.  Block statistics do
    not need the pixels.  Everything returned is still in image pixel units.
    """
    enc = model.encoding
    h, w, c = shape
    hs, ws = max(8, h // sub), max(8, w // sub)
    coords = pixel_centers(hs, ws, device)
    bs = max(2, block_px // sub)
    prev, deltas = None, []
    for k in range(enc.n_levels + 1):
        enc.set_level_window(float(k))
        out = render(model, coords, (hs, ws, c))
        if prev is not None:
            deltas.append((out - prev).abs().mean(-1))
        prev = out
    enc.set_level_window(float(enc.n_levels))
    D = torch.stack(deltas)                                   # (L, H, W)

    lev = torch.arange(D.shape[0], device=D.device, dtype=D.dtype)
    eff = (D * lev[:, None, None]).sum(0) / D.sum(0).clamp(min=1e-9)

    nb = F.avg_pool2d(D[None], bs, stride=bs, ceil_mode=True)[0]
    dom = nb.argmax(0).cpu().numpy()                          # (Hb, Wb)
    blocks = []
    for j in range(dom.shape[0]):
        for i in range(dom.shape[1]):
            l = int(dom[j, i])
            blocks.append({"x": i * block_px, "y": j * block_px,
                           "w": min(block_px, w - i * block_px),
                           "h": min(block_px, h - j * block_px),
                           "level": l,
                           "cell_px": round(w / enc.resolutions[l][0], 2)})
    return eff.cpu().numpy(), blocks


def train_job(p, device):
    try:
        down = max(1, int(p["downsample"]))
        img, target, coords, shape = get_image(DEFAULT_IMAGE, down, device)
        h, w, c = shape
        torch.manual_seed(0)
        model, ladder = build(p, shape)
        model = model.to(device)
        info = describe(model, shape)
        label = (f"L{int(p['n_levels'])} F{int(p['n_features'])} "
                 f"T2^{int(p['log2_hashmap_size'])} "
                 f"{'auto' if int(p['max_resolution']) <= 0 else int(p['max_resolution'])} "
                 f"{p['interpolation'][:6]}")
        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], ladder=ladder, metrics=info,
                       note=f"{w}x{h}x{c}, {info['n_total']:,} parameters "
                            f"({info['fraction_of_values']*100:.1f}% of the "
                            f"{info['n_values']:,} reference values), "
                            f"{info['hashed_levels']}/{len(ladder)} levels hashed, "
                            f"{info['finest_px_per_cell']:.2f} px per finest cell",
                       images={"reference": gray_png(img)}, stamp=JOB["stamp"] + 1)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps = int(p["steps"])
        batch = int(p["batch"])
        every = max(1, steps // 40)
        ref_t = target(coords).reshape(h, w, c)
        t_train = 0.0
        for step in range(steps + 1):
            if STOP.is_set():
                break
            t0 = time.perf_counter()
            xy = torch.rand(batch, 2, device=device)
            pred = model(xy)
            with torch.no_grad():
                gt = target(xy)
            if p["loss"] == "relative_l2":
                loss = ((pred - gt) ** 2 / (pred.detach() ** 2 + 1e-2)).mean()
            else:
                loss = ((pred - gt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_train += time.perf_counter() - t0

            if step % every == 0 or step == steps:
                fit = render(model, coords, (h, w, c)).clamp(0, 1)
                db = psnr(fit, ref_t)
                err = (fit - ref_t).abs().mean(-1).cpu().numpy()
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = t_train
                    JOB["curve"].append({"step": step, "t": t_train, "psnr": db,
                                         "loss": float(loss)})
                    JOB["metrics"] = {**info, "psnr": db, "loss": float(loss)}
                    JOB["images"]["fit"] = gray_png(fit)
                    JOB["images"]["error"] = cmap_png(err, ERROR_LUT_MAX)
                    JOB["stamp"] += 1
                eff, blocks = level_maps(model, shape, device)
                with LOCK:
                    JOB["images"]["levels"] = cmap_png(eff, LEVEL_LUT_MAX, "viridis")
                    JOB["blocks"] = {"blocks": blocks, "w": w, "h": h,
                                     "n_levels": len(ladder)}
                    JOB["stamp"] += 1
        with LOCK:
            if JOB["curve"]:
                JOB["history"].append({
                    "label": label, "params": info["n_total"],
                    "psnr": JOB["curve"][-1]["psnr"], "seconds": t_train,
                    "curve": [{"t": q["t"], "psnr": q["psnr"]} for q in JOB["curve"]]})
                JOB["history"] = JOB["history"][-8:]
    except Exception as e:
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1


KNOBS = [
    {"name": "n_levels", "label": "levels L", "min": 2, "max": 20, "default": 16,
     "step": 1},
    {"name": "n_features", "label": "features per level F", "min": 1, "max": 8,
     "default": 2, "step": 1},
    {"name": "log2_hashmap_size", "label": "log2 table size T", "min": 10,
     "max": 24, "default": 19, "step": 1},
    {"name": "base_resolution", "label": "coarsest cells per axis", "min": 2,
     "max": 64, "default": 16, "step": 1},
    {"name": "per_level_scale", "label": "growth per level b", "min": 1.1,
     "max": 2.0, "default": 1.4, "step": 0.05},
    {"name": "max_resolution", "label": "finest cells per axis (0 = the pixel count)",
     "min": 0, "max": 4096, "default": 0, "step": 32},
    {"name": "n_neurons", "label": "decoder width", "min": 16, "max": 256,
     "default": 64, "step": 16},
    {"name": "n_hidden_layers", "label": "decoder hidden layers", "min": 1,
     "max": 6, "default": 2, "step": 1},
]
TRAIN_KNOBS = [
    {"name": "lr", "label": "learning rate", "min": 1e-4, "max": 1e-1,
     "default": 1e-2, "step": 1e-4, "log": True},
    {"name": "steps", "label": "iterations", "min": 100, "max": 8000,
     "default": 1500, "step": 100},
    {"name": "batch", "label": "batch size (random pixels)", "min": 4096,
     "max": 1048576, "default": 262144, "step": 4096},
]
DEFAULTS = {k["name"]: k["default"] for k in KNOBS + TRAIN_KNOBS}
DEFAULTS.update(interpolation="linear", activation="relu", loss="relative_l2",
                downsample=2)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>fitting the painting</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>instant-ngp &mdash; fitting a painting</h1>
<p class="sub">Random pixel coordinates in, RGB out. The panel on the right of the
ladder is the one to watch: <b style="color:#e5a23c">px per finest cell</b>. Below
one pixel the encoder is resolving structure no sample constrains &mdash; it costs
parameters, barely moves PSNR, and turns up later as noise in every derivative
taken through the fit. Finished runs stay on the curve, so settings can be compared
on quality against time and against parameter count.</p>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs_enc"></div>
<div class="knobs" id="knobs_train"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button>
  <button id="clear">clear history</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="note" id="note"></div>
<div class="row equal" style="margin-top:18px">
  <div class="panel"><canvas id="c_ref" width="330" height="460"></canvas>
    <div class="cap">reference</div></div>
  <div class="panel"><canvas id="c_fit" width="330" height="460"></canvas>
    <div class="cap">fit</div></div>
  <div class="panel"><canvas id="c_err" width="330" height="460"></canvas>
    <div class="cap">absolute error &mdash; fixed scale 0&ndash;__ERRMAX__</div></div>
  <div class="panel"><canvas id="c_levels" width="330" height="460"></canvas>
    <div class="cap">cells at the scale of the level dominating each block</div></div>
</div>
<div id="levlegend" class="note"></div>
<div class="row" style="margin-top:18px">
  <div class="panel"><canvas id="c_curve" width="520" height="460"></canvas>
    <div class="cap">psnr against training time &mdash; this run and the last few</div></div>
  <div class="panel"><canvas id="c_effmap" width="330" height="460"></canvas>
    <div class="cap">effective level per pixel &mdash; same 0&ndash;20 scale</div></div>
</div>
<div class="row" style="margin-top:20px">
  <div class="panel"><div class="label">resolution ladder</div>
    <div id="ladder"></div></div>
  <div class="panel"><div class="label">runs so far</div>
    <div id="history"></div></div>
</div>
<div class="stats" id="stats"></div>
</div><script>
const KNOBS=__KNOBS__, TRAIN=__TRAIN__, DEF=__DEF__, LUTMAX=__LUTMAX__;
const knob=Object.assign({}, DEF);
let LAST=-1, POLL=null, IMG={};

const C=document.getElementById("controls");
function seg(name, opts, key, after){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=String(o).replace(/_/g," ");
    b.setAttribute("aria-pressed", knob[key]===o);
    b.onclick=()=>{ knob[key]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      if(after) after(); preview(); };
    s.appendChild(b); });
  g.append(l,s); C.appendChild(g);
}
seg("interpolation", ["linear","smoothstep"], "interpolation");
seg("decoder activation", ["relu","gelu","softplus","tanh"], "activation");
seg("loss", ["relative_l2","l2"], "loss");
seg("downsample", [1,2,4], "downsample");

function panel(el, title, list){
  el.innerHTML="";
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  el.appendChild(t);
  list.forEach(p=>{
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.log ? (+v).toExponential(1)
                       : (p.step<1 ? (+v).toFixed(2) : String(Math.round(v)));
    const rawOf=v=>p.log ? Math.log10(v) : v;
    const valOf=r=>p.log ? Math.pow(10, r) : +r;
    val.textContent=fmt(knob[p.name]);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=rawOf(p.min); r.max=rawOf(p.max); r.step=p.log?0.02:p.step;
    r.value=rawOf(knob[p.name]);
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${fmt(p.min)}</span><span>${fmt(p.max)}</span>`;
    r.oninput=()=>{ knob[p.name]=valOf(r.value); val.textContent=fmt(knob[p.name]);
                    preview(); };
    d.append(lab,r,ends); el.appendChild(d);
  });
}
panel(document.getElementById("knobs_enc"), "encoding and decoder", KNOBS);
panel(document.getElementById("knobs_train"), "training", TRAIN);

let pv=null;
async function preview(){
  clearTimeout(pv);
  pv=setTimeout(async()=>{
    const r=await (await fetch("/api/preview?"+new URLSearchParams(knob))).json();
    if(r.error) return;
    drawLadder(r.ladder, r.info);
    if(!JOBRUNNING) document.getElementById("note").textContent=r.note;
  }, 120);
}
let JOBRUNNING=false;

function drawLadder(ladder, info){
  if(!ladder) return;
  let h='<table class="ladder"><tr><th>level</th><th>cells</th>'
       +'<th>px / cell</th><th>table</th></tr>';
  ladder.forEach(L=>{
    h+=`<tr class="${L.dense?'':'hashed'}"><td>${L.level}</td>`
      +`<td>${L.rx} &times; ${L.ry}</td><td>${L.px.toFixed(2)}</td>`
      +`<td>${L.dense?'dense':'hashed'}</td></tr>`; });
  h+="</table>";
  if(info) h+=`<div class="note">${info.n_enc.toLocaleString()} table + `
             +`${info.n_mlp.toLocaleString()} decoder parameters</div>`;
  document.getElementById("ladder").innerHTML=h;
}

document.getElementById("run").onclick=async()=>{
  await fetch("/api/start?"+new URLSearchParams(knob));
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 400); poll();
};
document.getElementById("stop").onclick=()=>fetch("/api/stop");
document.getElementById("clear").onclick=async()=>{ await fetch("/api/clear"); poll(); };

function drawImg(id, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(!src){ g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height); return; }
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  g.drawImage(im,(cv.width-im.width*s)/2,(cv.height-im.height*s)/2,
              im.width*s, im.height*s);
}

const HCOL=["#4da3ff","#e5a23c","#2ea043","#cf6bd6","#e5484d","#6bd6c9",
            "#9aa4b2","#d6c96b"];
// viridis-ish ramp: coarse levels dark blue, fine levels yellow
function levColor(t){
  const S=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},`
        +`${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
function drawLevels(bk){
  const cv=document.getElementById("c_levels"), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!bk || !bk.blocks){ document.getElementById("levlegend").textContent=
    "run a fit to see the level decomposition"; return; }
  const im=IMG["c_ref"];
  const s=Math.min(cv.width/bk.w, cv.height/bk.h);
  const ox=(cv.width-bk.w*s)/2, oy=(cv.height-bk.h*s)/2;
  if(im){ g.globalAlpha=0.55; g.drawImage(im,ox,oy,bk.w*s,bk.h*s); g.globalAlpha=1; }
  const maxl=LUTMAX;
  bk.blocks.forEach(b=>{
    const col=levColor(b.level/maxl);
    // Cells finer than ~3 screen px would be a solid wash: tint the block
    // instead, so "too fine to draw" still reads as "very fine".
    const cw=b.cell_px*s;
    g.strokeStyle=col; g.lineWidth=0.6; g.globalAlpha=0.85;
    if(cw < 3){
      g.fillStyle=col; g.globalAlpha=0.30;
      g.fillRect(ox+b.x*s, oy+b.y*s, b.w*s, b.h*s); g.globalAlpha=0.85;
    } else {
      // Step along the GLOBAL cell lattice and clip to the block. Starting each
      // block at its own origin makes the last cell of one block and the first
      // of the next share a boundary that is not a cell boundary, and the grid
      // reads as unevenly spaced.
      const c=b.cell_px;
      for(let x=Math.ceil(b.x/c)*c; x<=b.x+b.w; x+=c){ g.beginPath();
        g.moveTo(ox+x*s, oy+b.y*s); g.lineTo(ox+x*s, oy+(b.y+b.h)*s); g.stroke(); }
      for(let y=Math.ceil(b.y/c)*c; y<=b.y+b.h; y+=c){ g.beginPath();
        g.moveTo(ox+b.x*s, oy+y*s); g.lineTo(ox+(b.x+b.w)*s, oy+y*s); g.stroke(); }
    }
    g.globalAlpha=1;
  });
  const used=[...new Set(bk.blocks.map(b=>b.level))].sort((a,b)=>a-b);
  let bar="";
  for(let i=0;i<=LUTMAX;i++)
    bar+=`<span style="display:inline-block;width:16px;height:10px;`
        +`background:${levColor(i/LUTMAX)}"></span>`;
  let ticks="";
  for(let i=0;i<=LUTMAX;i+=5)
    ticks+=`<span style="display:inline-block;width:80px">${i}</span>`;
  document.getElementById("levlegend").innerHTML =
    `<div style="margin-bottom:2px">level colour scale, fixed 0&ndash;${LUTMAX}`
    +` &nbsp; (applies to the grid and to the effective-level map)</div>`
    +`<div style="line-height:0">${bar}</div><div>${ticks}</div>`
    +`<div style="margin-top:5px">dominant level per 64 px block in this fit: `
    + used.map(l=>{
        const b=bk.blocks.find(q=>q.level===l);
        return `<span style="color:${levColor(l/LUTMAX)}">&#9632; ${l} `
              +`(${b.cell_px} px cells)</span>`; }).join(" &nbsp; ") + "</div>";
}
function drawCurve(cur, hist){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const W=cv.width, H=cv.height;
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  const all=(hist||[]).map((h,i)=>({pts:h.curve, col:HCOL[i%HCOL.length],
                                    lab:h.label}))
    .concat(cur && cur.length ? [{pts:cur, col:"#ffffff", lab:"current"}] : []);
  if(!all.length) return;
  const pts=all.flatMap(a=>a.pts);
  const tmax=Math.max(...pts.map(p=>p.t), 1e-3);
  const dbs=pts.map(p=>p.psnr).filter(v=>isFinite(v));
  const lo=Math.min(...dbs), hi=Math.max(...dbs);
  const pad={l:52,r:12,t:14,b:34};
  const X=t=>pad.l+Math.log10(1+t)/Math.log10(1+tmax)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-lo)/((hi-lo)||1))*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.fillStyle="#666"; g.font="10px monospace";
  for(let i=0;i<=4;i++){ const v=lo+(hi-lo)*i/4, y=Y(v);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.fillText(v.toFixed(1), 8, y+3); }
  all.forEach(a=>{
    g.strokeStyle=a.col; g.lineWidth=a.col==="#ffffff"?2.2:1.5; g.beginPath();
    a.pts.forEach((p,i)=>{ const x=X(p.t), y=Y(p.psnr);
      i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke(); });
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  g.fillText("training time (s, log)", W/2-56, H-10);
  g.save(); g.translate(14,H/2+34); g.rotate(-Math.PI/2);
  g.fillText("psnr (dB)",0,0); g.restore();
}

function drawHistory(hist){
  if(!hist || !hist.length){ document.getElementById("history").innerHTML=
    '<div class="note">no finished runs yet</div>'; return; }
  let h='<table class="ladder"><tr><th></th><th>settings</th><th>params</th>'
       +'<th>psnr</th><th>s</th></tr>';
  hist.forEach((r,i)=>{
    h+=`<tr><td style="color:${HCOL[i%HCOL.length]}">&#9632;</td>`
      +`<td style="text-align:left">${r.label}</td>`
      +`<td>${r.params.toLocaleString()}</td><td>${r.psnr.toFixed(2)}</td>`
      +`<td>${r.seconds.toFixed(1)}</td></tr>`; });
  document.getElementById("history").innerHTML=h+"</table>";
}

async function poll(){
  const r=await (await fetch("/api/state")).json();
  JOBRUNNING=r.running;
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";
  if(r.note) document.getElementById("note").textContent=r.note;
  if(r.stamp!==LAST){
    LAST=r.stamp;
    drawImg("c_ref", r.images.reference); drawImg("c_fit", r.images.fit);
    drawImg("c_err", r.images.error);   drawImg("c_effmap", r.images.levels);
    drawLevels(r.blocks);
    drawCurve(r.curve, r.history); drawLadder(r.ladder, r.metrics);
    drawHistory(r.history);
    const m=r.metrics||{};
    document.getElementById("stats").innerHTML = m.psnr===undefined ? "press run"
      : `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `
       +`${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; psnr <b>${m.psnr.toFixed(2)}</b> dB`
       +`<br><b>${m.n_total.toLocaleString()}</b> parameters `
       +`(<b>${(m.fraction_of_values*100).toFixed(1)}%</b> of the `
       +`${m.n_values.toLocaleString()} reference RGB values) &nbsp;&middot;&nbsp; `
       +`finest cell covers <b>${m.finest_px_per_cell.toFixed(2)}</b> px`
       +` &nbsp;&middot;&nbsp; ${m.hashed_levels} levels hashed`;
  }
  if(!r.running && POLL){ clearInterval(POLL); POLL=null; }
}
preview(); poll();
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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            page = (PAGE.replace("__CSS__", CSS)
                        .replace("__KNOBS__", json.dumps(KNOBS))
                        .replace("__TRAIN__", json.dumps(TRAIN_KNOBS))
                        .replace("__DEF__", json.dumps(DEFAULTS))
                        .replace("__LUTMAX__", str(LEVEL_LUT_MAX))
                        .replace("__ERRMAX__", f"{ERROR_LUT_MAX:g}"))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/preview":
            p = _params(q)
            try:
                shape = image_shape(DEFAULT_IMAGE, max(1, int(p["downsample"])))
                model, ladder = build(p, shape)
                info = describe(model, shape)
                note = (f"{info['width']}x{info['height']}, {info['n_total']:,} "
                        f"parameters ({info['fraction_of_values']*100:.1f}% of the "
                        f"{info['n_values']:,} reference values), "
                        f"{info['hashed_levels']}/{len(ladder)} levels hashed, "
                        f"{info['finest_px_per_cell']:.2f} px per finest cell")
                return self._send(json.dumps({"ladder": ladder, "info": info,
                                              "note": note}), "application/json")
            except Exception as e:
                return self._send(json.dumps({"error": str(e)}), "application/json")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            threading.Thread(target=train_job, args=(_params(q), self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            STOP.set()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/clear":
            with LOCK:
                JOB["history"] = []
                JOB["stamp"] += 1
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/state":
            with LOCK:
                return self._send(json.dumps(JOB), "application/json")
        self.send_error(404)


def _params(q):
    p = dict(DEFAULTS)
    for k, v in q.items():
        p[k] = float(v) if _isnum(v) else v
    return p


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    global DEFAULT_IMAGE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    DEFAULT_IMAGE = a.image
    Handler.device = torch.device(a.device)
    print(f"http://localhost:{a.port}   (device {a.device})")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
