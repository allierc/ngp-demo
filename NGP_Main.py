#!/usr/bin/env python
"""Entry point for a yaml-specified NGP fit, in the shape connectome-gnn uses.

    python NGP_Main.py -o bench zapbench_bench       # time it, write bench_<gpu>.json
    python NGP_Main.py -o fit   zapbench_bench       # train it, write metrics.json
    python NGP_Main.py -o table zapbench_bench       # the table across every gpu so far

`-o <task> <config>` mirrors GNN_Main.py: the config is `config/<name>.yaml` and
everything the run needs is in it, so the same file can be handed to a cluster
job and to a laptop and produce comparable numbers.

The benchmark measures the TRAINING STEP and nothing else -- no rendering, no
panels, no png -- because that is what a bigger GPU is supposed to change, and
mixing the display into it is how a benchmark ends up measuring matplotlib.  Data
is loaded and resident before the clock starts; the reported peak memory is the
allocator's, which includes the resident volume, and both are printed separately
so a memory-bound result cannot be read as a compute-bound one.

Results land in `log/<config>/bench_<gpu>.json`, one file per GPU, so runs from
different machines collect into one directory and `-o table` reads them all.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ngp import NGPField                                              # noqa: E402
from ngp.utils import psnr                                            # noqa: E402
# The dataset loading and the encoder construction live with the page that was
# written against them; importing keeps one implementation rather than a second
# that drifts.
import gui_scalar_time as page                                        # noqa: E402


def load_config(name):
    path = name if os.path.isabs(name) or name.endswith(".yaml") else \
        os.path.join(ROOT, "config", f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f), path


def gpu_tag(device):
    if device.type != "cuda":
        return "cpu"
    n = torch.cuda.get_device_name(device)
    return n.replace("NVIDIA ", "").replace(" ", "_").replace("-", "_")


def build_from_config(cfg, shape, n_frames, device):
    """The encoder the yaml asks for.  Same construction the pages use, so a
    benchmark number and a panel come from the same model."""
    h, w = shape
    enc, tr = cfg["encoder"], cfg["training"]
    p = dict(page.DEFAULTS)
    p.update(n_levels=enc["n_levels"],
             log2_hashmap_size=enc["log2_hashmap_size"],
             px_per_finest_cell=enc["px_per_finest_cell"],
             frames_per_finest_cell=enc["frames_per_finest_cell"],
             lr=tr["lr"], steps=tr["steps"], batch=tr["batch"])
    torch.manual_seed(tr.get("seed", 0))
    return page.build(p, w, h, n_frames).to(device), p


def get_data(cfg, device):
    stores = page.datasets(cfg["dataset"].get("glob") or None)
    field = cfg["dataset"]["field"]
    if field not in stores:
        sys.exit(f"dataset {field!r} not found; on disk: {sorted(stores)}")
    return page.load_field(field, int(cfg["dataset"]["downsample"]), device,
                           stores=stores)


def one_step(model, opt, vol, batch, device, T):
    xy = torch.rand(batch, 2, device=device)
    ti = torch.randint(T, (batch,), device=device).float() / max(1, T - 1)
    xyt = torch.cat([xy, ti[:, None]], 1)
    loss = ((model(xyt)[:, 0] - page.sample_field(vol, xyt)) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def run_bench(cfg, name, device):
    b = cfg["benchmark"]
    tr = cfg["training"]
    vol, T, h, w, src = get_data(cfg, device)
    resident = vol.numel() * vol.element_size()
    model, p = build_from_config(cfg, (h, w), T, device)
    n_enc, n_mlp = model.n_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    batch = int(tr["batch"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    print(f"[bench ] {gpu_tag(device)}  {name}  {T} frames of {w}x{h}, "
          f"{n_enc + n_mlp:,} parameters, batch {batch:,}", flush=True)

    for _ in range(int(b["warmup_steps"])):
        one_step(model, opt, vol, batch, device, T)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    reps = []
    n = int(b["timed_steps"])
    for r in range(int(b["repeats"])):
        t0 = time.perf_counter()
        for _ in range(n):
            one_step(model, opt, vol, batch, device, T)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / n
        reps.append(dt)
        print(f"  repeat {r + 1}/{b['repeats']}: {dt * 1e3:7.2f} ms/step, "
              f"{batch / dt / 1e6:6.2f} M samples/s", flush=True)

    ms = statistics.median(reps) * 1e3
    peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    out = {
        "config": name, "gpu": gpu_tag(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else platform.processor() or "cpu"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "dataset": cfg["dataset"]["field"], "source": src,
        "frames": int(T), "height": int(h), "width": int(w),
        "n_parameters": int(n_enc + n_mlp), "n_table": int(n_enc),
        "batch": batch, "ms_per_step": ms,
        "ms_per_step_all": [x * 1e3 for x in reps],
        "samples_per_s": batch / (ms / 1e3),
        "steps_per_s": 1e3 / ms,
        "peak_alloc_gb": peak / 1e9,
        "resident_data_gb": resident / 1e9,
        "model_state_gb": (peak - resident) / 1e9,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[result] {out['gpu']}: {ms:.2f} ms/step, "
          f"{out['samples_per_s'] / 1e6:.2f} M samples/s, peak "
          f"{out['peak_alloc_gb']:.2f} GB of which {out['resident_data_gb']:.2f} "
          f"is the resident volume", flush=True)
    return out


def run_fit(cfg, name, device):
    """The same step, run to `training.steps`, scored on the whole volume."""
    tr = cfg["training"]
    vol, T, h, w, src = get_data(cfg, device)
    model, p = build_from_config(cfg, (h, w), T, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    batch, steps = int(tr["batch"]), int(tr["steps"])
    t0 = time.perf_counter()
    for step in range(steps):
        loss = one_step(model, opt, vol, batch, device, T)
        if step % max(1, steps // 10) == 0:
            print(f"  step {step:5d}  loss {loss.detach().item():.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_s = time.perf_counter() - t0

    # Scored frame by frame so the whole volume is covered without holding a
    # second copy of it.
    from ngp.utils import render
    se, n = 0.0, 0
    with torch.no_grad():
        for i in range(T):
            c = page.frame_coords(h, w, i / max(1, T - 1), device)
            se += float(((render(model, c, (h, w)) - vol[i]) ** 2).sum())
            n += h * w
    mse = se / n
    peak = 2 * float(vol.abs().max())
    db = 10 * np.log10(peak ** 2 / max(mse, 1e-12))
    n_enc, n_mlp = model.n_parameters()
    print(f"[fit   ] {steps} steps in {train_s:.1f} s -> {db:.2f} dB over all "
          f"{T} frames, {n_enc + n_mlp:,} parameters", flush=True)
    return {"config": name, "gpu": gpu_tag(device), "psnr_db": db,
            "train_s": train_s, "steps": steps,
            "n_parameters": int(n_enc + n_mlp), "source": src,
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_table(name, log_dir):
    rows = []
    for f in sorted(os.listdir(log_dir)) if os.path.isdir(log_dir) else []:
        if f.startswith("bench_") and f.endswith(".json"):
            rows.append(json.load(open(os.path.join(log_dir, f))))
    if not rows:
        sys.exit(f"no bench_*.json under {log_dir}")
    rows.sort(key=lambda r: r["ms_per_step"])
    base = rows[-1]["ms_per_step"]
    print(f"\n{name}: {rows[0]['dataset']}, {rows[0]['frames']} frames of "
          f"{rows[0]['width']}x{rows[0]['height']}, "
          f"{rows[0]['n_parameters']:,} parameters, batch {rows[0]['batch']:,}\n")
    print(f"  {'gpu':22s} {'ms/step':>9s} {'M samples/s':>12s} {'peak GB':>9s} "
          f"{'data GB':>8s} {'vs slowest':>11s}")
    for r in rows:
        print(f"  {r['device_name'][:22]:22s} {r['ms_per_step']:9.2f} "
              f"{r['samples_per_s'] / 1e6:12.2f} {r['peak_alloc_gb']:9.2f} "
              f"{r['resident_data_gb']:8.2f} {base / r['ms_per_step']:10.2f}x")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--option", nargs="+", required=True,
                    help="<task> <config>, task in {bench, fit, table}")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=None, help="override training.steps")
    ap.add_argument("--batch", type=int, default=None, help="override training.batch")
    a = ap.parse_args()
    task, name = a.option[0], a.option[1]
    cfg, path = load_config(name)
    if a.steps:
        cfg["training"]["steps"] = a.steps
    if a.batch:
        cfg["training"]["batch"] = a.batch
    log_dir = os.path.join(ROOT, "log", cfg.get("name", name))
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device(a.device)
    print(f"config {path}\ndevice {a.device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)

    if task == "bench":
        out = run_bench(cfg, cfg.get("name", name), device)
        f = os.path.join(log_dir, f"bench_{out['gpu']}.json")
        json.dump(out, open(f, "w"), indent=1)
        print(f"wrote {f}")
        run_table(cfg.get("name", name), log_dir)
    elif task == "fit":
        out = run_fit(cfg, cfg.get("name", name), device)
        f = os.path.join(log_dir, f"fit_{out['gpu']}.json")
        json.dump(out, open(f, "w"), indent=1)
        print(f"wrote {f}")
    elif task == "table":
        run_table(cfg.get("name", name), log_dir)
    else:
        sys.exit(f"unknown task {task!r}; expected bench, fit or table")


if __name__ == "__main__":
    main()
