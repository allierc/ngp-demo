#!/usr/bin/env python
"""Browser viewer for the zapbench flow field: the mask in 3D, and the flow on it.

    python scripts/zapbench_view.py            # http://localhost:8023
    python scripts/zapbench_view.py --stride 2 # every 2nd masked voxel, lighter

Reads `gs://zapbench-release` anonymously, so it needs `tensorstore` and a
network, and neither is required by the rest of this repo:

    pip install tensorstore

The flow field is 3 x 36 x 83 x 128 x 7879 and the segmentation is
2048 x 1328 x 72 -- exactly 16 x 16 x 2 the flow grid -- so a block-max over the
segmentation gives a per-flow-voxel mask of "a segmented cell is here". That is
21.6% of the volume; the other 78% is background where the flow is extrapolated
and means nothing, which is the whole reason a sparse representation is being
considered at all.

Rendering is deliberately dependency-free: the point cloud is projected in
JavaScript and written straight into an ImageData buffer, so there is no CDN, no
WebGL and nothing to install in the browser. Drag to rotate, scroll to zoom,
drag the time slider to move through the run.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp.webui import CSS

BUCKET = "zapbench-release"
FLOW = "volumes/20240930/flow_fields/"
SEG = "volumes/20240930/segmentation/"

STATE: dict = {"ready": False, "note": "opening the volumes..."}
LOCK = threading.Lock()
CACHE: dict = {}


def _open(path):
    import tensorstore as ts
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "gcs",
                    "bucket": BUCKET, "path": path}}, open=True, read=True).result()


def build(stride: int):
    """Mask + geometry, once. The mask read is the slow part, about 30 s."""
    t0 = time.perf_counter()
    flow = _open(FLOW)
    C, Z, Y, X, T = flow.domain.shape
    seg = _open(SEG)
    SX, SY, SZ = seg.domain.shape
    fx, fy, fz = SX // X, SY // Y, SZ // Z
    with LOCK:
        STATE["note"] = f"building the mask from {SX}x{SY}x{SZ} segmentation..."
    mask = np.zeros((Z, Y, X), dtype=bool)
    for z in range(Z):
        s = seg[:, :, z * fz:(z + 1) * fz].read().result()
        mask[z] = (s > 0).reshape(X, fx, Y, fy, fz).any(axis=(1, 3, 4)).T
    idx = np.flatnonzero(mask.reshape(-1))
    if stride > 1:
        idx = idx[::stride]
    zz, yy, xx = np.unravel_index(idx, (Z, Y, X))
    # centred and scaled to [-1, 1] on the longest axis, so the client can rotate
    # it without knowing the voxel geometry
    pts = np.stack([xx, yy, zz], 1).astype(np.float32)
    ctr = np.array([X, Y, Z], dtype=np.float32) / 2
    pts = (pts - ctr) / max(X, Y, Z) * 2
    with LOCK:
        CACHE.update(flow=flow, idx=idx, shape=(Z, Y, X), T=T, pts=pts)
        STATE.update(ready=True, n_points=int(len(idx)), n_masked=int(mask.sum()),
                     n_total=int(mask.size), T=int(T), stride=stride,
                     note=f"{int(mask.sum()):,} of {mask.size:,} flow voxels "
                          f"({mask.mean()*100:.1f}%) contain a segmented cell; "
                          f"showing {len(idx):,} of them "
                          f"({time.perf_counter()-t0:.0f} s to load)")
    print("[ready] " + STATE["note"], flush=True)


FRAMES: "collections.OrderedDict" = None


def frame(t: int):
    """Per-point displacement at time t: magnitude, and the three components.

    Cached, because playback revisits the same frames on every pass and each miss
    is a chunk read from the bucket -- about half a second, which is what bounds
    the frame rate.
    """
    global FRAMES
    import collections
    if FRAMES is None:
        FRAMES = collections.OrderedDict()
    t = int(t)
    if t in FRAMES:
        FRAMES.move_to_end(t)
        return FRAMES[t]
    flow, idx = CACHE["flow"], CACHE["idx"]
    a = flow[:, :, :, :, t].read().result().reshape(3, -1)[:, idx]        # (3, N)
    out = (a, np.linalg.norm(a, axis=0))
    FRAMES[t] = out
    while len(FRAMES) > 96:                       # ~96 x 3 x 83k floats = 95 MB
        FRAMES.popitem(last=False)
    return out


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a)).decode()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>zapbench flow field</title>
<style>__CSS__
  canvas#view { cursor:grab; }
  canvas#view:active { cursor:grabbing; }
</style></head><body><div class="wrap">
<h1>zapbench &mdash; the flow field on the cells</h1>
<p class="sub">Every point is one flow-field voxel that a segmented cell occupies.
The other 78% of the volume is background, where the flow is extrapolated and means
nothing &mdash; it is not drawn, which is the point. Drag to rotate, scroll to zoom,
move the slider to travel through the run.</p>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="note" id="note">loading...</div>
<div class="row" style="margin-top:14px">
  <div class="panel"><canvas id="view" width="900" height="620"></canvas>
    <div class="cap" id="cap">masked voxels</div></div>
</div>
<div class="stats" id="stats"></div>
</div><script>
let PTS=null, VAL=null, N=0, T=1, ROT={x:-0.35,y:0.6}, ZOOM=1.25, DRAG=null;
let MODE="mag", FRAME=0, VMAX=1, BUSY=false, SCALE=16, PLAY=false, SUB0=0;
// Advances by a stride so the whole run plays in a few hundred steps, and awaits
// each frame rather than firing on a timer, so it self-throttles to the network.
async function run(){
  while(PLAY){
    // Skip frames rather than wait for them. Each frame is a fresh chunk read
    // from the bucket at roughly half a second, so the loop is bound by the
    // network and speed can only come from taking bigger steps: 10 steps across
    // the run, every ~790th frame. The field is rank 2 over the whole recording,
    // so the frames in between carry almost nothing the neighbours do not.
    const stride=Math.max(1, Math.round(T/10));
    FRAME=(FRAME+stride)%T; rng.value=FRAME; val.textContent=FRAME;
    await loadFrame(FRAME);
  }
}

const cv=document.getElementById("view"), g=cv.getContext("2d");
const img=g.createImageData(cv.width, cv.height);

function seg(name, opts, cur, cb){
  const C=document.getElementById("controls");
  const gp=document.createElement("div"); gp.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(([txt,val])=>{
    const b=document.createElement("button"); b.textContent=txt;
    b.setAttribute("aria-pressed", cur===val);
    b.onclick=()=>{ [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b));
                    cb(val); };
    s.appendChild(b); });
  gp.append(l,s); C.appendChild(gp);
}
seg("colour by", [["|u|","mag"],["direction","dir"],["c0","0"],["c1","1"],["c2","2"]], "mag",
    v=>{ MODE=v; loadFrame(FRAME); });
seg("scale (voxels)", [["4",4],["8",8],["16",16],["32",32]], 16,
    v=>{ SCALE=v; loadFrame(FRAME); });
seg("baseline", [["none",0],["subtract frame 0",1]], 0,
    v=>{ SUB0=v; loadFrame(FRAME); });
seg("", [["play",1],["pause",0]], 0, v=>{ PLAY=!!v; if(PLAY) run(); });

const K=document.getElementById("knobs");
K.innerHTML='<div class="title">time</div>';
const kn=document.createElement("div"); kn.className="knob";
kn.style.minWidth="100%";
const lab=document.createElement("div"); lab.className="kl";
const val=document.createElement("b"); val.textContent="0";
lab.innerHTML="<span>frame</span>"; lab.appendChild(val);
const rng=document.createElement("input"); rng.type="range"; rng.min=0; rng.value=0;
const ends=document.createElement("div"); ends.className="ends";
kn.append(lab,rng,ends); K.appendChild(kn);
let pend=null;
rng.oninput=()=>{ val.textContent=rng.value;
  clearTimeout(pend); pend=setTimeout(()=>loadFrame(+rng.value), 90); };

// Rotation is recomputed from the drag origin on every move, not accumulated,
// so a dropped mouseup cannot leave the view spinning.
cv.addEventListener("mousedown", e=>{ DRAG={sx:e.clientX, sy:e.clientY,
                                            rx:ROT.x, ry:ROT.y}; });
window.addEventListener("mouseup", ()=>{ DRAG=null; });
window.addEventListener("mousemove", e=>{
  if(!DRAG) return;
  ROT.y = DRAG.ry + (e.clientX-DRAG.sx)*0.006;
  ROT.x = DRAG.rx + (e.clientY-DRAG.sy)*0.006;
  draw();
});
cv.addEventListener("wheel", e=>{ e.preventDefault();
  ZOOM = Math.min(6, Math.max(0.4, ZOOM*(e.deltaY<0?1.12:0.9))); draw(); },
  {passive:false});

function ramp(t){   // the repo's level ramp: bright at both ends
  const S=[[77,163,255],[64,224,208],[124,255,90],[255,210,77],[255,107,107]];
  const x=Math.max(0,Math.min(1,t))*(S.length-1), i=Math.floor(x), f=x-i;
  const a=S[i], b=S[Math.min(S.length-1,i+1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
function hue(t){   // cyclic, for an angle: no seam except where the angle wraps
  const h=(t%1)*6, i=Math.floor(h), f=h-i;
  const q=[[255,60,60],[255,215,60],[60,255,90],[60,230,255],[110,110,255],[255,80,220]];
  const a=q[i%6], b=q[(i+1)%6];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
function diverge(v){  // signed: blue negative, white zero, red positive
  const t=Math.max(-1,Math.min(1,v));
  return t<0 ? [77+178*(1+t), 163+92*(1+t), 255] : [255, 255-148*t, 255-148*t];
}

function draw(){
  if(!PTS) return;
  const W=cv.width, H=cv.height, d=img.data;
  d.fill(0);
  for(let i=3;i<d.length;i+=4) d[i]=255;
  const cy=Math.cos(ROT.y), sy=Math.sin(ROT.y);
  const cx=Math.cos(ROT.x), sx=Math.sin(ROT.x);
  const sc=Math.min(W,H)*0.42*ZOOM;
  const depth=new Float32Array(W*H).fill(-1e9);
  for(let i=0;i<N;i++){
    const X=PTS[3*i], Y=PTS[3*i+1], Z=PTS[3*i+2];
    const x1=X*cy - Z*sy,  z1=X*sy + Z*cy;
    const y1=Y*cx - z1*sx, z2=Y*sx + z1*cx;
    const px=(W/2 + x1*sc)|0, py=(H/2 + y1*sc)|0;
    if(px<0||px>=W||py<0||py>=H) continue;
    const o=py*W+px;
    if(z2 <= depth[o]) continue;          // painter's algorithm on a z-buffer
    depth[o]=z2;
    const v=VAL[i]/255;
    const c = MODE==="mag" ? ramp(v) : MODE==="dir" ? hue(v) : diverge(v*2-1);
    const k=o*4;
    d[k]=c[0]; d[k+1]=c[1]; d[k+2]=c[2];
  }
  g.putImageData(img,0,0);
}

async function loadFrame(t){
  if(BUSY) return; BUSY=true;
  FRAME=t;
  const r=await (await fetch(`/api/frame?t=${t}&mode=${MODE}&vmax=${SCALE}&sub0=${SUB0}`)).json();
  VAL=new Uint8Array(atob(r.val).split("").map(c=>c.charCodeAt(0)));
  VMAX=r.vmax;
  let bar="";
  for(let i=0;i<=20;i++){
    const c = MODE==="mag" ? ramp(i/20) : MODE==="dir" ? hue(i/20) : diverge(i/10-1);
    bar+=`<span style="display:inline-block;width:13px;height:9px;`
        +`background:rgb(${c[0]|0},${c[1]|0},${c[2]|0})"></span>`;
  }
  const ends = MODE==="dir" ? `0 &rarr; 360&deg; in the c1&ndash;c2 plane`
             : MODE==="mag" ? `0 &rarr; ${SCALE} voxels`
                            : `&minus;${SCALE} &rarr; +${SCALE} voxels`;
  document.getElementById("stats").innerHTML=
    `frame <b>${t}</b> of ${T-1}`
    +(SUB0 ? ` &nbsp;&middot;&nbsp; <b>minus frame 0</b>` : "")
    +` &nbsp;&middot;&nbsp; `
    +(MODE==="dir"
        ? `mean direction <b>${r.mean.toFixed(0)}&deg;</b>, in-plane |u| up to `
          +`<b>${r.data_max.toFixed(1)}</b>`
        : `mean <b>${r.mean.toFixed(2)}</b> vox, largest <b>${r.data_max.toFixed(1)}</b>`)
    +(r.clipped>0 ? ` &nbsp;&middot;&nbsp; <span class="bad">`
                    +`${(r.clipped*100).toFixed(1)}% beyond the scale</span>` : "")
    +`<br><span style="line-height:0">${bar}</span> `
    +`<span style="color:var(--dim)"> ${ends} (fixed)</span>`;
  BUSY=false; draw();
}

async function boot(){
  for(;;){
    const s=await (await fetch("/api/state")).json();
    document.getElementById("note").textContent=s.note;
    if(s.ready){ T=s.T; rng.max=T-1;
      ends.innerHTML=`<span>0</span><span>${T-1}</span>`;
      const geo=await (await fetch("/api/geometry")).json();
      const raw=atob(geo.pts); const buf=new Uint8Array(raw.length);
      for(let i=0;i<raw.length;i++) buf[i]=raw.charCodeAt(i);
      PTS=new Float32Array(buf.buffer); N=geo.n;
      document.getElementById("cap").textContent =
        `${geo.n.toLocaleString()} masked voxels of ${s.n_total.toLocaleString()}`;
      loadFrame(0); return; }
    await new Promise(r=>setTimeout(r, 1000));
  }
}
boot();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            return self._send(PAGE.replace("__CSS__", CSS), "text/html; charset=utf-8")
        if u.path == "/api/state":
            with LOCK:
                return self._send(json.dumps(STATE), "application/json")
        if u.path == "/api/geometry":
            with LOCK:
                pts = CACHE["pts"]
            return self._send(json.dumps({"pts": b64(pts), "n": int(len(pts))}),
                              "application/json")
        if u.path == "/api/frame":
            t = int(float(q.get("t", 0)))
            mode = q.get("mode", "mag")
            a, mag = frame(t)
            # SUBTRACT FRAME 0. The field is dominated by a near-constant offset
            # -- one component sits at about -9 for the entire run -- so the
            # drift, which is the part that varies and the part a registration
            # has to track, is a small signal riding on a large constant. Taking
            # frame 0 as the baseline shows u(t) - u(0) and nothing else.
            if q.get("sub0") == "1" and int(t) != 0:
                a = a - frame(0)[0]
                mag = np.linalg.norm(a, axis=0)
            elif q.get("sub0") == "1":
                a = np.zeros_like(a)
                mag = np.zeros_like(mag)
            # A FIXED scale by default. Rescaling to each frame's own percentile
            # makes a drifting field look static: the colours stay put while the
            # displacement grows underneath them.
            fixed = q.get("vmax")
            if mode == "dir":
                # In-plane direction, in the plane orthogonal to component 0.
                # An angle wraps, so it gets a cyclic ramp and a fixed 0-360
                # scale; a linear one would put a seam through the middle of it.
                ang = np.arctan2(a[1], a[2]) % (2 * np.pi)
                val = ang / (2 * np.pi)
                vmax, d = 360.0, np.hypot(a[1], a[2])
                return self._send(json.dumps(
                    {"val": b64((val * 255).astype(np.uint8)), "vmax": vmax,
                     "mean": float(np.degrees(np.arctan2(a[1].mean(), a[2].mean())) % 360),
                     "data_max": float(d.max()), "clipped": 0.0}), "application/json")
            if mode == "mag":
                d = mag
                vmax = float(fixed) if fixed else float(np.percentile(mag, 99)) or 1.0
                val = np.clip(mag / vmax, 0, 1)
            else:
                d = a[int(mode)]
                vmax = float(fixed) if fixed else float(np.percentile(np.abs(d), 99)) or 1.0
                val = np.clip(d / vmax, -1, 1) * 0.5 + 0.5
            over = float((np.abs(d) > vmax).mean())
            return self._send(json.dumps({"val": b64((val * 255).astype(np.uint8)),
                                          "vmax": vmax, "mean": float(d.mean()),
                                          "data_max": float(np.abs(d).max()),
                                          "clipped": over}),
                              "application/json")
        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8023)
    p.add_argument("--stride", type=int, default=1,
                   help="show every Nth masked voxel; 1 shows all ~83,000")
    a = p.parse_args()
    try:
        import tensorstore                                   # noqa: F401
    except ImportError:
        sys.exit("this viewer needs tensorstore:  pip install tensorstore")
    threading.Thread(target=build, args=(a.stride,), daemon=True).start()
    print(f"http://localhost:{a.port}   (loading the mask takes ~30 s)")
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except OSError as e:
        if e.errno == 98:
            sys.exit(f"port {a.port} is already in use -- pass --port with a free one")
        raise
    srv.serve_forever()


if __name__ == "__main__":
    main()
