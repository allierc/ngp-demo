#!/usr/bin/env python
"""Browser GUI for the registration benchmark: fit a known warp, watch it converge.

    python scripts/gui.py                 # http://localhost:8021
    python scripts/gui.py --port 8030

Everything the benchmark decides from a YAML file is a control here instead:
which ground-truth deformation to recover, how mismatched the two "modalities"
are, which parameterisation fits it, and -- the part a config file hides -- the
training schedule itself. Learning rate, iteration count, batch size and the two
regulariser weights are sliders, because on this problem they change the answer
as much as the model does.

The panels are chosen so that the two things that separate the methods are both
on screen at once: the warped image (which almost always looks fine) and the
endpoint error against the analytic field (which does not).
"""

from __future__ import annotations

import argparse
import base64
import io
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
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp.deform import (apply_mismatch, build_deformation, build_model, field_jacobian,
                        lncc_loss, patch_offsets, pixel_grid, sample_bilinear, warp_image)
from ngp.utils import psnr
from ngp.webui import ABOUT_HTML
from scripts.run_registration import (_dense_field, _feather, foreground_mask, load_image,
                                      resolve_inherits, sample_points)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPLAY_H = 460                      # panel height in px; images are sent downsampled

JOB = {"running": False, "step": 0, "steps": 0, "seconds": 0.0, "curve": [],
       "metrics": {}, "images": {}, "grid": {}, "note": "", "stamp": 0}
LOCK = threading.Lock()
STOP = threading.Event()
SCENE = {}                           # cached image / masks / ground truth per key


# --------------------------------------------------------------- rendering


def _png(rgb: np.ndarray) -> str:
    im = Image.fromarray(rgb)
    if im.height > DISPLAY_H:
        im = im.resize((max(1, round(im.width * DISPLAY_H / im.height)), DISPLAY_H),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_png(t: torch.Tensor) -> str:
    a = np.clip(t.detach().cpu().numpy(), 0, 1)
    return _png((np.stack([a] * 3, -1) * 255).astype(np.uint8))


def cmap_png(a: np.ndarray, vmax: float, name="inferno") -> str:
    x = np.clip(a / max(vmax, 1e-6), 0, 1)
    return _png((matplotlib.colormaps[name](x)[..., :3] * 255).astype(np.uint8))


def grid_lines(u: np.ndarray, spacing: int, shape):
    """Polylines of a regular grid carried through x -> x + u(x), in image px."""
    h, w = shape
    ys = np.arange(0, h, spacing)
    xs = np.arange(0, w, spacing)
    out = []
    for y in ys:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for x in xs])
    for x in xs:
        out.append([(float(x + u[y, x, 0]), float(y + u[y, x, 1])) for y in ys])
    return out


# ------------------------------------------------------------------ scene


def get_scene(cfg, dname, xname, device):
    key = (dname, xname)
    if key in SCENE:
        return SCENE[key]
    src = load_image(cfg["image"], device)
    shape = tuple(src.shape[:2])
    fgc = cfg["image"]["foreground"]
    fg = foreground_mask(src, fgc)
    if fgc.get("zero_background"):
        src = src * _feather(fg, fgc.get("feather_px", 9))
    band_r = cfg["metrics"]["evaluation"]["boundary_band_px"]
    grown = F.max_pool2d(fg.float()[None, None], 2 * band_r + 1, stride=1,
                         padding=band_r)[0, 0] > 0.5
    built = {}
    for name, spec in {d["name"]: d for d in cfg["deformations"]}.items():
        built[name] = build_deformation(spec, built, shape, device, cfg.get("seed", 0), fg)
    u_gt = built[dname]
    clean = warp_image(src, u_gt, shape)
    xspec = {m["name"]: m for m in cfg["modality_mismatch"]}[xname]
    obs = apply_mismatch(clean, xspec, cfg.get("seed", 0))
    ugt = _dense_field(u_gt, shape, device).reshape(*shape, 2)
    SCENE[key] = {
        "source": src, "clean": clean, "observed": obs, "u_gt": u_gt,
        "ugt_dense": ugt, "shape": shape, "loss": xspec.get("loss", "l2"),
        "fg_idx": torch.nonzero(fg.reshape(-1), as_tuple=False).squeeze(1),
        "masks": {"foreground": fg, "background": ~grown, "boundary_band": grown & ~fg},
    }
    return SCENE[key]


# --------------------------------------------------------------- training


def _evaluate(model, sc, device, spacing):
    shape = sc["shape"]
    h, w = shape
    px = torch.tensor([w, h], device=device, dtype=torch.float32)
    warped = warp_image(sc["source"], model, shape)
    u = _dense_field(model, shape, device).reshape(h, w, 2)
    epe = (u - sc["ugt_dense"]).norm(dim=-1)
    sub = pixel_grid(h // 3, w // 3, device)
    J = field_jacobian(model, sub, px)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    m = {
        "psnr": psnr(warped, sc["clean"]),
        "epe_mean": float(epe.mean()),
        "epe_fg": float(epe[sc["masks"]["foreground"]].mean()),
        "epe_band": float(epe[sc["masks"]["boundary_band"]].mean()),
        "epe_bg": float(epe[sc["masks"]["background"]].mean()),
        "det_min": float(det.min()),
        "folded": float((det < 0).float().mean()),
        "folded_count": int((det < 0).sum()),
        "jacobian_samples": int(det.numel()),
    }
    un = u.detach().cpu().numpy()
    images = {"warped": gray_png(warped),
              "epe": cmap_png(epe.cpu().numpy(),
                              max(1.0, float(np.percentile(epe.cpu().numpy(), 99))))}
    grid = {"fit": grid_lines(un, spacing, shape),
            "gt": grid_lines(sc["ugt_dense"].cpu().numpy(), spacing, shape),
            "w": shape[1], "h": shape[0],
            "epe_vmax": max(1.0, float(np.percentile(epe.cpu().numpy(), 99)))}
    return m, images, grid


def train_job(cfg, p, device):
    """One fit, publishing to JOB as it goes. Runs on its own thread."""
    try:
        sc = get_scene(cfg, p["deformation"], p["mismatch"], device)
        shape = sc["shape"]
        h, w = shape
        px = torch.tensor([w, h], device=device, dtype=torch.float32)

        spec = {m["name"]: m for m in resolve_inherits(cfg["models"])}[p["model"]]
        spec = json.loads(json.dumps(spec))                     # deep copy
        if spec["kind"] == "hash_grid":
            spec["encoding"].update(
                n_levels=int(p["n_levels"]),
                log2_hashmap_size=int(p["log2_hashmap_size"]),
                max_resolution=int(p["max_resolution"]),
                interpolation=p["interpolation"])
            spec.setdefault("coarse_to_fine", {})
            spec["coarse_to_fine"] = {"enabled": bool(p["coarse_to_fine"]),
                                      "start_levels": 4,
                                      "full_at_step": max(1, int(p["steps"] * 0.5))}
        else:
            g = int(p["grid"])
            spec["grid"] = [g, g]
        spec["output_scale_px"] = float(p["output_scale_px"])

        torch.manual_seed(cfg.get("seed", 0))
        model = build_model(spec, device)
        n_a, n_b = model.n_parameters()
        is_hash = spec["kind"] == "hash_grid"
        enc = model.field.encoding if is_hash else None
        note = (f"{enc.resolutions[0][0]}..{enc.resolutions[-1][0]} cells, "
                f"{sum(1 for d in enc.dense if not d)}/{enc.n_levels} levels hashed, "
                f"{w / enc.resolutions[-1][0]:.1f} px per finest cell" if is_hash
                else f"{spec['grid'][0]}x{spec['grid'][1]} control points, "
                     f"{w / spec['grid'][1]:.0f} px apart")

        with LOCK:
            JOB.update(running=True, step=0, steps=int(p["steps"]), seconds=0.0,
                       curve=[], metrics={"n_parameters": n_a + n_b}, note=note,
                       images={"source": gray_png(sc["source"]),
                               "target": gray_png(sc["observed"])},
                       grid={}, stamp=JOB["stamp"] + 1)

        opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
        steps = int(p["steps"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=float(p["lr"]) * 0.03)
        offs = patch_offsets(cfg["training"]["loss"]["lncc"]["window_px"], shape, device)
        n_patch = cfg["training"]["loss"]["lncc"]["n_patches"]
        frac = cfg["training"]["sampling"]["foreground_fraction"]
        reg_n = cfg["training"].get("reg_batch", 8192)
        w_smooth, w_fold = float(p["w_smooth"]), float(p["w_fold"])
        every = max(1, steps // 40)
        t0 = time.perf_counter()

        for step in range(steps + 1):
            if STOP.is_set():
                break
            if is_hash and spec["coarse_to_fine"]["enabled"]:
                a = 4 + (enc.n_levels - 4) * min(
                    1.0, step / spec["coarse_to_fine"]["full_at_step"])
                enc.set_level_window(a)
            if sc["loss"] == "lncc":
                c = sample_points(sc["fg_idx"], n_patch, frac, shape, device)
                xy = (c[:, None, :] + offs[None]).reshape(-1, 2)
            else:
                xy = sample_points(sc["fg_idx"], int(p["batch"]), frac, shape, device)
            u = model(xy)
            pred = sample_bilinear(sc["source"], xy + u / px)
            gt = sample_bilinear(sc["observed"], xy)
            loss = (lncc_loss(pred, gt, n_patch) if sc["loss"] == "lncc"
                    else ((pred - gt) ** 2).mean())
            photo = float(loss)
            if w_smooth > 0 or w_fold > 0:
                xr = sample_points(sc["fg_idx"], reg_n, frac, shape, device)
                J = field_jacobian(model, xr, px, create_graph=True)
                if w_smooth > 0:
                    loss = loss + w_smooth * ((J - torch.eye(2, device=device)) ** 2
                                              ).sum((1, 2)).mean()
                if w_fold > 0:
                    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                    loss = loss + w_fold * F.relu(0.1 - det).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if step % every == 0 or step == steps:
                m, images, grid = _evaluate(model, sc, device, int(p["grid_spacing"]))
                with LOCK:
                    JOB["step"] = step
                    JOB["seconds"] = time.perf_counter() - t0
                    JOB["curve"].append({"step": step, "loss": photo,
                                         "epe_fg": m["epe_fg"], "epe_bg": m["epe_bg"]})
                    JOB["metrics"] = {**m, "n_parameters": n_a + n_b, "loss": photo,
                                      "loss_kind": sc["loss"]}
                    JOB["images"].update(images)
                    JOB["grid"] = grid
                    JOB["stamp"] += 1
    except Exception as e:                                   # surface, don't swallow
        with LOCK:
            JOB["note"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            JOB["running"] = False
            JOB["stamp"] += 1


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>registration &mdash; ngp vs control grid</title>
<style>
  :root { --fg:#fff; --bg:#000; --dim:#9a9a9a; --red:#e5484d; --blue:#4da3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1360px; margin:0 auto; padding:26px 22px 48px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 6px; }
  .sub { font-size:12px; color:var(--dim); margin:0 0 22px; max-width:940px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:18px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .seg button:disabled { opacity:.35; cursor:default; }
  .knobs { display:flex; flex-wrap:wrap; gap:26px; margin:0 0 14px;
           padding:14px 16px; border:1px solid #333; }
  .knobs .title { width:100%; font-size:10px; letter-spacing:.16em;
                  text-transform:uppercase; color:var(--dim); margin-bottom:2px; }
  .knob { display:flex; flex-direction:column; gap:5px; min-width:250px; flex:1; }
  .knob .kl { font-size:11px; color:var(--dim); display:flex;
              justify-content:space-between; gap:12px; }
  .knob .kl b { color:var(--fg); font-weight:600;
                font-variant-numeric:tabular-nums; }
  .knob .ends { display:flex; justify-content:space-between; font-size:9px;
                color:#666; font-variant-numeric:tabular-nums; }
  input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                      height:1px; background:var(--fg); outline:none; margin:6px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
    appearance:none; width:13px; height:13px; background:var(--fg);
    border:1px solid var(--fg); cursor:pointer; border-radius:0; }
  input[type=range]::-moz-range-thumb { width:13px; height:13px;
    background:var(--fg); border:1px solid var(--fg); cursor:pointer;
    border-radius:0; }
  .row { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid #333; }
  .cap { font-size:10px; letter-spacing:.14em; text-transform:uppercase;
         color:var(--dim); }
  .cap i { color:var(--red); font-style:normal; }
  .cap b { color:var(--blue); font-weight:600; }
  .stats { font-size:12px; color:var(--dim); margin-top:18px;
           font-variant-numeric:tabular-nums; line-height:1.9; }
  .stats b { color:var(--fg); font-weight:600; }
  .stats .bad { color:var(--red); }
  .bar { height:2px; background:#222; margin:14px 0 0; }
  .bar i { display:block; height:2px; background:var(--fg); width:0; }
  .note { font-size:11px; color:#7a7a7a; margin-top:6px; }
</style></head><body><div class="wrap">
<h1>registration &mdash; hash grid against a control grid</h1>
<p class="sub">The source is warped by a known analytic field to make the target,
so the fit is scored on the <b style="color:#fff">field</b> it recovered, not only on how well the
images line up. Endpoint error is split into the textured foreground, the black
background where nothing constrains the warp, and the band between &mdash; that split
is where the two parameterisations disagree, long after the warped image stops
telling them apart.</p>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button onclick="openAbout()">what is an ngp?</button></div>
</div></div>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs_model"></div>
<div class="knobs" id="knobs_train"></div>
<div class="controls"><div class="group"><div class="label">&nbsp;</div>
  <div class="seg"><button id="run">run</button><button id="stop">stop</button></div>
</div></div>
<div class="bar"><i id="prog"></i></div>
<div class="note" id="note"></div>
<div class="row" style="margin-top:18px">
  <div class="panel"><canvas id="c_source" width="390" height="460"></canvas>
    <div class="cap">source</div></div>
  <div class="panel"><canvas id="c_target" width="390" height="460"></canvas>
    <div class="cap">target &mdash; source warped, then remapped</div></div>
  <div class="panel"><canvas id="c_warp" width="390" height="460"></canvas>
    <div class="cap">source warped by the fit</div></div>
</div>
<div class="row" style="margin-top:18px">
  <div class="panel"><canvas id="c_epe" width="390" height="460"></canvas>
    <div class="cap">endpoint error, px</div></div>
  <div class="panel"><canvas id="c_grid" width="390" height="460"></canvas>
    <div class="cap">grid &mdash; <i>ground truth</i> vs <b>fit</b></div></div>
  <div class="panel"><canvas id="c_curve" width="520" height="460"></canvas>
    <div class="cap">loss and endpoint error against iteration</div></div>
</div>
<div class="stats" id="stats"></div>
<div class="modal" id="about" onclick="if(event.target===this)closeAbout()">
  <div class="sheet">__ABOUT__</div></div>
</div><script>
function openAbout(){document.getElementById("about").classList.add("open");}
function closeAbout(){document.getElementById("about").classList.remove("open");}
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeAbout();});
const SPEC=__SPEC__, KNOBS=__KNOBS__, DEF=__DEF__;
const sel={deformation:SPEC.deformation[0], mismatch:SPEC.mismatch[0],
           model:SPEC.model[0]};
const knob=Object.assign({}, DEF);
let LAST=-1, POLL=null, IMG={};

const C=document.getElementById("controls");
function group(name, opts, key){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=String(o).replace(/_/g," ");
    b.setAttribute("aria-pressed", sel[key]===o);
    b.onclick=()=>{ sel[key]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      // The L2 data term sits near 1e-4 and 1-LNCC near 3e-1, so the same
      // absolute regulariser weight is ~1000x weaker under LNCC. Follow the
      // loss when the mismatch changes instead of leaving a stale weight.
      if(key==="mismatch"){ const r=SPEC.reg[SPEC.loss[o]];
        knob.w_smooth=r.w_smooth; knob.w_fold=r.w_fold; }
      buildKnobs(); };
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g);
}
group("deformation", SPEC.deformation, "deformation");
group("mismatch", SPEC.mismatch, "mismatch");
group("parameterisation", SPEC.model, "model");

function isHash(){ return SPEC.kind[sel.model]==="hash_grid"; }
function knobPanel(el, title, list){
  el.innerHTML="";
  const t=document.createElement("div"); t.className="title"; t.textContent=title;
  el.appendChild(t);
  list.forEach(p=>{
    if(p.name==="interpolation"){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML="<span>interpolation</span>";
      const s=document.createElement("div"); s.className="seg";
      ["linear","smoothstep"].forEach(o=>{
        const b=document.createElement("button"); b.textContent=o;
        b.setAttribute("aria-pressed", knob.interpolation===o);
        b.onclick=()=>{ knob.interpolation=o;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); };
        s.appendChild(b); });
      d.append(lab,s); el.appendChild(d); return;
    }
    if(p.name==="coarse_to_fine"){
      const d=document.createElement("div"); d.className="knob";
      const lab=document.createElement("div"); lab.className="kl";
      lab.innerHTML="<span>coarse to fine (level window)</span>";
      const s=document.createElement("div"); s.className="seg";
      [["on",1],["off",0]].forEach(([txt,v])=>{
        const b=document.createElement("button"); b.textContent=txt;
        b.setAttribute("aria-pressed", knob.coarse_to_fine===v);
        b.onclick=()=>{ knob.coarse_to_fine=v;
          [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b)); };
        s.appendChild(b); });
      d.append(lab,s); el.appendChild(d); return;
    }
    if(knob[p.name]===undefined) knob[p.name]=p.default;
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b");
    const fmt=v=>p.log ? (+v).toExponential(1)
                       : (p.step<1 ? (+v).toFixed(3) : String(Math.round(v)));
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
    r.oninput=()=>{ knob[p.name]=valOf(r.value); val.textContent=fmt(knob[p.name]); };
    d.append(lab,r,ends); el.appendChild(d);
  });
}
function buildKnobs(){
  knobPanel(document.getElementById("knobs_model"),
            isHash() ? "encoding" : "control grid",
            isHash() ? KNOBS.hash : KNOBS.control);
  knobPanel(document.getElementById("knobs_train"), "training", KNOBS.train);
}
buildKnobs();

document.getElementById("run").onclick=async()=>{
  const q=new URLSearchParams(Object.assign({}, sel, knob));
  await fetch("/api/start?"+q);
  if(POLL) clearInterval(POLL);
  POLL=setInterval(poll, 400); poll();
};
document.getElementById("stop").onclick=()=>fetch("/api/stop");

function drawImg(id, src){
  const cv=document.getElementById(id), g=cv.getContext("2d");
  if(!src){ g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height); return; }
  if(IMG[id] && IMG[id].src===src){ blit(g,cv,IMG[id]); return; }
  const im=new Image(); im.onload=()=>{ IMG[id]=im; blit(g,cv,im); }; im.src=src;
}
function blit(g,cv,im){
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  const s=Math.min(cv.width/im.width, cv.height/im.height);
  const w=im.width*s, h=im.height*s;
  g.drawImage(im,(cv.width-w)/2,(cv.height-h)/2,w,h);
}

function drawGrid(gr){
  const cv=document.getElementById("c_grid"), g=cv.getContext("2d");
  g.fillStyle="#000"; g.fillRect(0,0,cv.width,cv.height);
  if(!gr || !gr.gt) return;
  const s=Math.min(cv.width/gr.w, cv.height/gr.h);
  const ox=(cv.width-gr.w*s)/2, oy=(cv.height-gr.h*s)/2;
  const paint=(lines,col,lw,dash)=>{
    g.strokeStyle=col; g.lineWidth=lw; g.setLineDash(dash||[]);
    lines.forEach(L=>{ g.beginPath();
      L.forEach((p,i)=>{ const X=ox+p[0]*s, Y=oy+p[1]*s;
        i?g.lineTo(X,Y):g.moveTo(X,Y); }); g.stroke(); });
    g.setLineDash([]);
  };
  paint(gr.gt, "#e5484d", 1.2);
  paint(gr.fit, "#4da3ff", 1.0, [4,3]);
}

function drawCurve(curve){
  const cv=document.getElementById("c_curve"), g=cv.getContext("2d");
  const W=cv.width, H=cv.height;
  g.fillStyle="#000"; g.fillRect(0,0,W,H);
  if(!curve || curve.length<2) return;
  const pad={l:56,r:14,t:16,b:34};
  const xs=curve.map(c=>c.step);
  const x0=Math.min(...xs), x1=Math.max(...xs)||1;
  const vals=curve.flatMap(c=>[c.loss,c.epe_fg,c.epe_bg]).filter(v=>v>0);
  const lo=Math.max(1e-8, Math.min(...vals)), hi=Math.max(...vals);
  const X=v=>pad.l+(v-x0)/((x1-x0)||1)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/
                    ((Math.log10(hi)-Math.log10(lo))||1))*(H-pad.t-pad.b);
  g.strokeStyle="#222"; g.lineWidth=1;
  for(let d=Math.floor(Math.log10(lo)); d<=Math.ceil(Math.log10(hi)); d++){
    const y=Y(Math.pow(10,d)); if(y<pad.t||y>H-pad.b) continue;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillStyle="#666"; g.font="10px monospace";
    g.fillText("1e"+d, 8, y+3);
  }
  const line=(key,col)=>{ g.strokeStyle=col; g.lineWidth=1.8; g.beginPath();
    curve.forEach((c,i)=>{ const X_=X(c.step), Y_=Y(c[key]);
      i?g.lineTo(X_,Y_):g.moveTo(X_,Y_); }); g.stroke(); };
  line("loss","#e8e8e8"); line("epe_fg","#4da3ff"); line("epe_bg","#e5a23c");
  g.fillStyle="#8a8a8a"; g.font="11px sans-serif";
  g.fillText("iteration", W/2-22, H-10);
  [["loss","#e8e8e8"],["epe foreground (px)","#4da3ff"],
   ["epe background (px)","#e5a23c"]].forEach(([t,c],i)=>{
    g.fillStyle=c; g.fillRect(W-pad.r-160, pad.t+i*15-8, 9, 2);
    g.fillStyle="#9a9a9a"; g.fillText(t, W-pad.r-145, pad.t+i*15-3); });
}

async function poll(){
  const r=await (await fetch("/api/state")).json();
  document.getElementById("prog").style.width=
    (r.steps ? (r.step/r.steps*100) : 0)+"%";
  document.getElementById("note").textContent=r.note||"";
  if(r.stamp!==LAST){
    LAST=r.stamp;
    drawImg("c_source", r.images.source); drawImg("c_target", r.images.target);
    drawImg("c_warp", r.images.warped);   drawImg("c_epe", r.images.epe);
    drawGrid(r.grid); drawCurve(r.curve); stats(r);
  }
  if(!r.running && POLL){ clearInterval(POLL); POLL=null; }
}
function stats(r){
  const m=r.metrics||{};
  if(m.psnr===undefined){ document.getElementById("stats").innerHTML=
    r.running ? "running&hellip;" : "press run"; return; }
  const fold=m.folded_count>0
    ? `<span class="bad">${m.folded_count} of ${m.jacobian_samples} samples</span>`
    : `<b>none</b> of ${m.jacobian_samples} samples`;
  document.getElementById("stats").innerHTML=
    `iteration <b>${r.step}</b> / ${r.steps} &nbsp;&middot;&nbsp; `+
    `${r.seconds.toFixed(1)} s &nbsp;&middot;&nbsp; `+
    `${m.loss_kind} loss <b>${m.loss.toExponential(2)}</b><br>`+
    `endpoint error &nbsp; foreground <b>${m.epe_fg.toFixed(3)}</b> px`+
    ` &nbsp; boundary band <b>${m.epe_band.toFixed(3)}</b>`+
    ` &nbsp; background <b>${m.epe_bg.toFixed(3)}</b><br>`+
    `psnr vs the clean warp <b>${m.psnr.toFixed(2)}</b> dB`+
    ` &nbsp;&middot;&nbsp; min det J <b>${m.det_min.toFixed(3)}</b>`+
    ` &nbsp; folded ${fold}`+
    ` &nbsp;&middot;&nbsp; <b>${m.n_parameters.toLocaleString()}</b> parameters`;
}
poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    cfg = None
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
            cfg = self.cfg
            resolved = resolve_inherits(cfg["models"])
            models = [m["name"] for m in resolved]
            kinds = {m["name"]: m["kind"] for m in resolved}
            sm = cfg["training"]["loss"]["smoothness"]["weight"]
            fo = cfg["training"]["loss"]["folding"]["weight"]
            spec = {"deformation": [d["name"] for d in cfg["deformations"]],
                    "mismatch": [m["name"] for m in cfg["modality_mismatch"]],
                    "model": models, "kind": kinds,
                    "loss": {m["name"]: m.get("loss", "l2")
                             for m in cfg["modality_mismatch"]},
                    "reg": {k: {"w_smooth": sm[k] if isinstance(sm, dict) else sm,
                                "w_fold": fo[k] if isinstance(fo, dict) else fo}
                            for k in ("l2", "lncc")}}
            page = (PAGE.replace("__ABOUT__", ABOUT_HTML)
                        .replace("__SPEC__", json.dumps(spec))
                        .replace("__KNOBS__", json.dumps(KNOB_SPEC))
                        .replace("__DEF__", json.dumps(KNOB_DEFAULTS)))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/start":
            if JOB["running"]:
                return self._send(json.dumps({"error": "already running"}),
                                  "application/json")
            STOP.clear()
            p = dict(KNOB_DEFAULTS)
            p.update({k: (float(v) if _isnum(v) else v) for k, v in q.items()})
            threading.Thread(target=train_job, args=(self.cfg, p, self.device),
                             daemon=True).start()
            return self._send(json.dumps({"ok": True}), "application/json")
        if u.path == "/api/stop":
            STOP.set()
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


KNOB_SPEC = {
    "hash": [
        {"name": "n_levels", "label": "levels", "min": 2, "max": 16,
         "default": 12, "step": 1},
        {"name": "log2_hashmap_size", "label": "log2 hash table size",
         "min": 10, "max": 22, "default": 16, "step": 1},
        {"name": "max_resolution", "label": "finest cells per axis",
         "min": 16, "max": 1024, "default": 512, "step": 16},
        {"name": "interpolation"},
        {"name": "coarse_to_fine"},
    ],
    "control": [
        {"name": "grid", "label": "control points per axis", "min": 4,
         "max": 512, "default": 16, "step": 4},
    ],
    "train": [
        {"name": "lr", "label": "learning rate", "min": 1e-4, "max": 3e-1,
         "default": 1e-2, "step": 1e-4, "log": True},
        {"name": "steps", "label": "iterations", "min": 100, "max": 6000,
         "default": 1500, "step": 100},
        {"name": "batch", "label": "batch size (sample points)", "min": 4096,
         "max": 262144, "default": 65536, "step": 4096},
        {"name": "w_smooth", "label": "smoothness weight", "min": 1e-6,
         "max": 1e-1, "default": 1e-3, "step": 1e-6, "log": True},
        {"name": "w_fold", "label": "folding penalty weight", "min": 1e-6,
         "max": 1.0, "default": 1e-2, "step": 1e-6, "log": True},
        {"name": "output_scale_px", "label": "displacement scale (px)", "min": 4,
         "max": 160, "default": 40, "step": 4},
        {"name": "grid_spacing", "label": "overlay grid spacing (px)", "min": 8,
         "max": 128, "default": 32, "step": 8},
    ],
}
KNOB_DEFAULTS = {k["name"]: k.get("default") for g in KNOB_SPEC.values() for k in g
                 if "default" in k}
KNOB_DEFAULTS.update(interpolation="smoothstep", coarse_to_fine=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config/registration_benchmark.yaml"))
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    Handler.cfg = cfg
    Handler.device = torch.device(a.device)
    print(f"http://localhost:{a.port}   (device {a.device})")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
