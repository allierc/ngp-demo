#!/usr/bin/env python
"""default against warp: the same encoding, or the comparison means nothing.

    python tests/impl_gate.py

Swapping an implementation asserts the maths did not change.  This is that
assertion, run rather than stated: same seed, same table copied across, and the
forward and the table gradient compared against the pure-PyTorch encoder over
the shapes the pages actually use.

The tolerances are not tight by accident.  The forward is deterministic in both,
so it agrees to float32 rounding.  The gradient is a scatter-add of thousands of
contributions per row and the order differs between an atomic kernel and
index_put_, so it agrees to a relative tolerance and not an absolute one -- a
gate that demanded bitwise equality of a float32 atomic sum would be testing the
summation order, not the maths.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp.hashgrid import MultiResHashGrid
from ngp.hashgrid_warp import WarpHashGrid

CASES = [
    dict(n_input_dims=2, n_levels=8, log2_hashmap_size=12, base_resolution=4,
         per_level_scale=1.6, max_resolution=128, interpolation="linear"),
    dict(n_input_dims=3, n_levels=6, log2_hashmap_size=14,
         base_resolution=(8, 8, 2), per_level_scale=1.7,
         max_resolution=(64, 64, 16),
         interpolation=("smoothstep", "smoothstep", "linear")),
    dict(n_input_dims=3, n_levels=16, log2_hashmap_size=18,
         base_resolution=(8, 8, 2), per_level_scale=1.5,
         max_resolution=(512, 512, 128),
         interpolation=("smoothstep", "smoothstep", "linear")),
    # FOUR DIMENSIONS, because a volume that moves is (x, y, z, t) and not a
    # stack of independent planes. Sixteen corners per level instead of eight,
    # which is the whole difference: twice the gather, twice the atomics, and
    # the same equations. Untested until now -- the kernel was written for it
    # (vec4i, `1 << ndim`) but nothing had asked.
    dict(n_input_dims=4, n_levels=8, log2_hashmap_size=16,
         base_resolution=(8, 8, 2, 4), per_level_scale=1.5,
         max_resolution=(256, 256, 32, 128),
         interpolation=("smoothstep", "smoothstep", "smoothstep", "linear")),
]


def check(kw, n=8192, device="cuda:0"):
    dev = torch.device(device)
    torch.manual_seed(0)
    a = MultiResHashGrid(n_features_per_level=2, **kw).to(dev)
    torch.manual_seed(0)
    b = WarpHashGrid(n_features_per_level=2, **kw).to(dev)
    assert a.resolutions == b.resolutions, "ladders differ"
    assert a.n_entries == b.n_entries, "table sizes differ"
    assert a.dense == b.dense, "dense/hashed split differs"
    with torch.no_grad():
        b.table.copy_(a.table)

    x = torch.rand(n, kw["n_input_dims"], device=dev)
    ya, yb = a(x), b(x)
    fwd = float((ya - yb).abs().max())
    scale = float(ya.detach().abs().max())

    g = torch.randn_like(ya)
    a.zero_grad(set_to_none=True)
    (ya * g).sum().backward()
    b.zero_grad(set_to_none=True)
    (b(x) * g).sum().backward()
    ga, gb = a.table.grad, b.table.grad
    rel = float((ga - gb).abs().max() / ga.abs().max().clamp(min=1e-12))

    # the level window has to gate both the same way, or a coarse-to-fine run
    # would silently differ between implementations
    a.set_level_window(3.5)
    b.set_level_window(3.5)
    win = float((a(x) - b(x)).abs().max())

    ok = fwd < 1e-6 * max(scale, 1e-6) + 1e-9 and rel < 1e-4 and win < 1e-6 * max(scale, 1e-6) + 1e-9
    print(f"  D={kw['n_input_dims']} L={kw['n_levels']:2d} "
          f"T=2^{kw['log2_hashmap_size']}: forward {fwd:.2e} (values {scale:.2e}), "
          f"table grad rel {rel:.2e}, windowed {win:.2e}   {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("the warp encoder is CUDA only")
    print("default against warp, same seed and same table:")
    good = all([check(kw) for kw in CASES])
    print("\nimplementations agree" if good else "\nFAILED")
    sys.exit(0 if good else 1)
