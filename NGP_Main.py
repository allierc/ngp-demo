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
    model = page.build(p, w, h, n_frames).to(device)
    # THE IMPLEMENTATION AXIS, as Plexus puts it: same equations, different
    # machinery, picked by name in the spec. "default" is the pure-PyTorch
    # encoder; "compile" is the same graph handed to inductor; "warp" will be
    # the hand-written kernel. Swapping one for another asserts the maths did
    # not change, which tests/impl_gate.py checks.
    impl = str(enc.get("implementation", "default"))
    if impl == "compile":
        model.encoding.forward = torch.compile(model.encoding.forward,
                                               dynamic=False)
    elif impl != "default":
        raise SystemExit(f"unknown encoder.implementation {impl!r}")
    return model, p


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
    wall0 = time.perf_counter()
    vol, T, h, w, src = get_data(cfg, device)
    resident = vol.numel() * vol.element_size()
    model, p = build_from_config(cfg, (h, w), T, device)
    n_enc, n_mlp = model.n_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    batch = int(tr["batch"])

    print(f"[bench ] {gpu_tag(device)}  {name}  {T} frames of {w}x{h}, "
          f"{n_enc + n_mlp:,} parameters, batch {batch:,}", flush=True)

    def peak_at(bs, iters=6):
        """Peak allocation for one training step at this batch, data excluded.

        Reserved as well as allocated: reserved is what the allocator holds and
        therefore what decides whether the next run OOMs, and the two differ by
        the fragmentation that a large transient causes.  The resident volume is
        subtracted so what is left is the STEP's own cost, which is the thing a
        kernel rewrite can change.
        """
        if device.type != "cuda":
            return 0.0, 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iters):
            one_step(model, opt, vol, bs, device, T)
        torch.cuda.synchronize(device)
        return (torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

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
    reserved = (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)

    # THE MARGINAL COST PER SAMPLE, measured two-point rather than divided out of
    # one number: the step holds a fixed part (the table, Adam's moments, the
    # resident volume) and a part that scales with the batch (the gather's
    # intermediates), and only the second is what a batch-size or kernel change
    # moves.  Same method as the MPM note's bytes-per-particle.
    sweep = []
    for bs in sorted({max(4096, batch // 4), max(4096, batch // 2), batch}):
        a, r = peak_at(bs)
        sweep.append({"batch": bs, "peak_alloc_gb": a, "peak_reserved_gb": r})
        print(f"  batch {bs:>9,}: peak {a:6.2f} GB allocated, {r:6.2f} reserved",
              flush=True)
    per_sample = 0.0
    if len(sweep) >= 2:
        d_gb = sweep[-1]["peak_alloc_gb"] - sweep[0]["peak_alloc_gb"]
        d_n = sweep[-1]["batch"] - sweep[0]["batch"]
        per_sample = d_gb * 1e9 / max(1, d_n)
    fixed_gb = (sweep[-1]["peak_alloc_gb"] - per_sample * batch / 1e9) if sweep else 0.0
    out = {
        "config": name, "gpu": gpu_tag(device),
        "impl": str(cfg["encoder"].get("implementation", "default")),
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
        # What the configured fit would COST on this card: the number anyone
        # actually plans around, and the one a 5x on ms/step is worth reading as.
        "train_minutes": int(tr["steps"]) * ms / 60_000.0,
        "train_steps": int(tr["steps"]),
        "bench_wall_min": (time.perf_counter() - wall0) / 60.0,
        "peak_alloc_gb": peak / 1e9,
        "peak_reserved_gb": reserved / 1e9,
        "resident_data_gb": resident / 1e9,
        "model_state_gb": (peak - resident) / 1e9,
        "memory_sweep": sweep,
        "bytes_per_sample": per_sample,
        "fixed_gb": fixed_gb,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[result] {out['gpu']}: {ms:.2f} ms/step, "
          f"{out['samples_per_s'] / 1e6:.2f} M samples/s, peak "
          f"{out['peak_alloc_gb']:.2f} GB allocated / {out['peak_reserved_gb']:.2f} "
          f"reserved, of which {out['resident_data_gb']:.2f} is the resident volume",
          flush=True)
    print(f"[time  ] {out['train_steps']:,} steps would take "
          f"{out['train_minutes']:.1f} min on this card; the benchmark itself "
          f"took {out['bench_wall_min']:.1f} min", flush=True)
    if per_sample:
        card = float(cfg.get("benchmark", {}).get("card_gb", 0) or 0)
        print(f"[memory] {per_sample:.0f} B per sample marginal, {fixed_gb:.2f} GB "
              f"fixed", flush=True)
        if card:
            room = card * 0.9 - fixed_gb
            print(f"[memory] on a {card:.0f} GB card that leaves room for a batch of "
                  f"{room * 1e9 / per_sample / 1e6:.1f} M samples at this model size",
                  flush=True)
    return out


def run_profile(cfg, name, device, iters=30):
    """Where the step's time and memory go, stage by stage.

    A kernel rewrite can only pay where the time is, so this is the measurement
    that decides whether one is worth writing.  Each stage is timed in isolation
    with the device synchronised around it, and its peak allocation is the
    allocator's high-water mark for that stage alone -- so a stage that is cheap
    in time and expensive in memory (a big transient) shows up as such.

    The backward is split: the encoder's own backward is the scatter-add into
    the table, which is the part a Warp kernel would replace, and it is reported
    apart from the decoder's.
    """
    tr = cfg["training"]
    vol, T, h, w, src = get_data(cfg, device)
    model, p = build_from_config(cfg, (h, w), T, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    batch = int(tr["batch"])
    enc = model.encoding

    def timed(fn, n=iters):
        fn()                                        # warm the kernels
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / n * 1e3
        pk = (torch.cuda.max_memory_allocated(device) / 1e9
              if device.type == "cuda" else 0.0)
        return dt, pk

    xy = torch.rand(batch, 2, device=device)
    ti = torch.randint(T, (batch,), device=device).float() / max(1, T - 1)
    xyt = torch.cat([xy, ti[:, None]], 1)
    tgt = page.sample_field(vol, xyt)

    stages = []

    def add(label, fn, note=""):
        ms, gb = timed(fn)
        stages.append({"stage": label, "ms": ms, "peak_gb": gb, "note": note})
        print(f"  {label:26s} {ms:8.2f} ms   peak {gb:6.2f} GB   {note}", flush=True)

    add("coords (rand + cat)",
        lambda: torch.cat([torch.rand(batch, 2, device=device),
                           (torch.randint(T, (batch,), device=device).float()
                            / max(1, T - 1))[:, None]], 1))
    add("target (trilinear read)", lambda: page.sample_field(vol, xyt))
    add("encode forward", lambda: enc(xyt), f"{enc.n_levels} levels, 2^D corners each")
    feat = enc(xyt).detach()
    add("decoder forward", lambda: model.mlp(feat))
    add("model forward (both)", lambda: model(xyt))

    def enc_fwd_bwd():
        f = enc(xyt)
        f.sum().backward()
        model.zero_grad(set_to_none=True)
    add("encode fwd+bwd", enc_fwd_bwd, "the backward is the scatter-add")
    # The isolated backward cannot be run without its forward, so it is reported
    # as the difference rather than measured directly -- and said to be so,
    # because a table of stages that sums past 100% has quietly done this.
    fb = next(x["ms"] for x in stages if x["stage"] == "encode fwd+bwd")
    ff = next(x["ms"] for x in stages if x["stage"] == "encode forward")
    stages.append({"stage": "encode backward (by difference)", "ms": fb - ff,
                   "peak_gb": 0.0, "note": "fwd+bwd minus fwd"})
    print(f"  {'encode backward (diff)':26s} {fb - ff:8.2f} ms"
          f"                        scatter-add alone", flush=True)

    def full_step():
        pred = model(xyt)[:, 0]
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    add("full step (fwd+bwd+adam)", full_step)

    def adam_only():
        opt.step()
    add("adam step alone", adam_only, f"{sum(q.numel() for q in model.parameters()):,} params")

    total = next(x["ms"] for x in stages if x["stage"].startswith("full step"))
    print(f"\n  the step is {total:.2f} ms; as a share of it:")
    for x in stages:
        if not x["stage"].startswith("full step"):
            print(f"    {x['stage']:26s} {100 * x['ms'] / total:5.1f}%")
    out = {"config": name, "gpu": gpu_tag(device),
           "device_name": torch.cuda.get_device_name(device)
           if device.type == "cuda" else "cpu",
           "batch": batch, "stages": stages, "step_ms": total,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
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
    steps = rows[0].get("train_steps", 0)
    print(f"  {'gpu':22s} {'impl':>8s} {'ms/step':>9s} {'M smp/s':>9s} "
          f"{f'{steps} steps (min)':>17s} {'peak GB':>9s} {'B/sample':>9s} "
          f"{'vs slowest':>11s}")
    for r in rows:
        print(f"  {r['device_name'][:22]:22s} {r.get('impl','default'):>8s} "
              f"{r['ms_per_step']:9.2f} "
              f"{r['samples_per_s'] / 1e6:9.2f} {r.get('train_minutes', 0):17.1f} "
              f"{r['peak_alloc_gb']:9.2f} {r.get('bytes_per_sample', 0):9.0f} "
              f"{base / r['ms_per_step']:10.2f}x")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--option", nargs="+", required=True,
                    help="<task> <config>, task in {bench, profile, fit, table}")
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
        suffix = "" if out["impl"] == "default" else f"_{out['impl']}"
        f = os.path.join(log_dir, f"bench_{out['gpu']}{suffix}.json")
        json.dump(out, open(f, "w"), indent=1)
        print(f"wrote {f}")
        run_table(cfg.get("name", name), log_dir)
    elif task == "fit":
        out = run_fit(cfg, cfg.get("name", name), device)
        f = os.path.join(log_dir, f"fit_{out['gpu']}.json")
        json.dump(out, open(f, "w"), indent=1)
        print(f"wrote {f}")
    elif task == "profile":
        out = run_profile(cfg, cfg.get("name", name), device)
        f = os.path.join(log_dir, f"profile_{out['gpu']}.json")
        json.dump(out, open(f, "w"), indent=1)
        print(f"wrote {f}")
    elif task == "table":
        run_table(cfg.get("name", name), log_dir)
    else:
        sys.exit(f"unknown task {task!r}; expected bench, profile, fit or table")


if __name__ == "__main__":
    main()
