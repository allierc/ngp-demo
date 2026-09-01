"""The hash encoding as two Warp kernels, with the backward written by hand.

Same equations as `ngp/hashgrid.py`, different machinery: this sits on the
IMPLEMENTATION axis and not the model axis, so swapping one for the other
asserts the encoding did not change -- which `tests/impl_gate.py` tests rather
than assumes.

Why it is worth writing at all, measured before it was:

    A6000, batch 262,144, 16 levels, 32.6 M parameters
      encode forward       21.44 ms   28% of the step
      encode backward      45.66 ms   60% of the step   <- the scatter-add
      everything else       9.13 ms   12%
    torch.compile on the same graph: 1.9%.

Inductor fuses the forward's elementwise chain and cannot touch an atomic
scatter-add, so the 60% is untouched by anything short of a kernel.

TWO THINGS THE DEFAULT PATH DOES THAT THIS ONE DOES NOT.

  * It materialises the gather.  Per level, `table[idx]` is a (B, 2^D, F)
    tensor and the corner weights another (B, 2^D), to produce (B, F).  At batch
    262,144 with 16 levels that is most of the measured 2,438 B per sample, for
    128 B of features.  Here the 2^D corners are accumulated in registers and
    none of it is written.

  * It builds an autograd graph.  Every one of those intermediates is kept alive
    for the backward.  This is one `torch.autograd.Function` whose backward
    RECOMPUTES the weights and indices -- a few flops per corner -- and scatters
    straight into the table's gradient.  Recompute is cheaper than storage when
    the stored thing is 2 kB per sample and the recompute is a dozen multiplies.

LIMITATION, stated rather than discovered: the backward returns a gradient for
the table and NOT for the input coordinates.  The scalar-field fits never need
one; the registration pages do, and they use the default implementation.  Asking
for it here raises rather than silently returning zeros.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

import warp as wp

from .hashgrid import _PRIMES, _per_axis

wp.init()

F_FIXED = 2          # features per level; the paper's default and the only one
                     # this kernel is written for -- registers are not dynamic


@wp.func
def _corner_index(cx: wp.int32, cy: wp.int32, cz: wp.int32, cw: wp.int32,
                  rx: wp.int32, ry: wp.int32, rz: wp.int32, rw: wp.int32,
                  ndim: wp.int32, dense: wp.int32, tsize: wp.int32) -> wp.int32:
    """Row of the table this corner reads, dense or hashed.

    The same two rules as the reference: a level whose nodes fit in the table is
    indexed by its raster number, and one that does not is indexed by the XOR of
    its coordinates times the instant-NGP primes.
    """
    if dense == 1:
        idx = cx
        if ndim > 1:
            idx = idx * (ry + 1) + cy
        if ndim > 2:
            idx = idx * (rz + 1) + cz
        if ndim > 3:
            idx = idx * (rw + 1) + cw
        return idx
    h = cx * 1
    if ndim > 1:
        h = h ^ (cy * 2654435761)
    if ndim > 2:
        h = h ^ (cz * 805459861)
    if ndim > 3:
        h = h ^ (cw * 3674653429)
    h = h % tsize
    if h < 0:
        h = h + tsize
    return h


@wp.kernel
def hash_forward(x: wp.array2d(dtype=wp.float32),        # (B, D)
                 table: wp.array2d(dtype=wp.float32),    # (E, F)
                 res: wp.array2d(dtype=wp.int32),        # (L, 4)
                 off: wp.array(dtype=wp.int32),          # (L,)
                 tsz: wp.array(dtype=wp.int32),          # (L,)
                 dense: wp.array(dtype=wp.int32),        # (L,)
                 smooth: wp.array(dtype=wp.int32),       # (4,)
                 gain: wp.array(dtype=wp.float32),       # (L,)
                 ndim: wp.int32,
                 out: wp.array2d(dtype=wp.float32)):     # (B, L*F)
    b, l = wp.tid()
    a0 = float(0.0)
    a1 = float(0.0)
    g = gain[l]
    if g != 0.0:
        base = wp.vec4i(0, 0, 0, 0)
        w = wp.vec4(0.0, 0.0, 0.0, 0.0)
        for d in range(ndim):
            r = float(res[l, d])
            p = x[b, d] * r
            fl = wp.floor(p)
            if fl < 0.0:
                fl = 0.0
            if fl > r - 1.0:
                fl = r - 1.0
            t = p - fl
            if smooth[d] == 1:
                t = t * t * (3.0 - 2.0 * t)
            base[d] = int(fl)
            w[d] = t
        for c in range(1 << ndim):
            ww = float(1.0)
            cc = wp.vec4i(0, 0, 0, 0)
            for d in range(ndim):
                if ((c >> d) & 1) == 1:
                    ww = ww * w[d]
                    cc[d] = base[d] + 1
                else:
                    ww = ww * (1.0 - w[d])
                    cc[d] = base[d]
            idx = _corner_index(cc[0], cc[1], cc[2], cc[3],
                                res[l, 0], res[l, 1], res[l, 2], res[l, 3],
                                ndim, dense[l], tsz[l]) + off[l]
            a0 = a0 + ww * table[idx, 0]
            a1 = a1 + ww * table[idx, 1]
    out[b, l * 2 + 0] = a0 * g
    out[b, l * 2 + 1] = a1 * g


@wp.kernel
def hash_backward(x: wp.array2d(dtype=wp.float32),
                  grad_out: wp.array2d(dtype=wp.float32),   # (B, L*F)
                  res: wp.array2d(dtype=wp.int32),
                  off: wp.array(dtype=wp.int32),
                  tsz: wp.array(dtype=wp.int32),
                  dense: wp.array(dtype=wp.int32),
                  smooth: wp.array(dtype=wp.int32),
                  gain: wp.array(dtype=wp.float32),
                  ndim: wp.int32,
                  grad_table: wp.array2d(dtype=wp.float32)):
    """The weights and indices are RECOMPUTED, not read back.

    Nothing was stored by the forward, so there is nothing to load: a dozen
    multiplies per corner replaces 2 kB per sample of traffic.
    """
    b, l = wp.tid()
    g = gain[l]
    if g == 0.0:
        return
    g0 = grad_out[b, l * 2 + 0] * g
    g1 = grad_out[b, l * 2 + 1] * g
    base = wp.vec4i(0, 0, 0, 0)
    w = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for d in range(ndim):
        r = float(res[l, d])
        p = x[b, d] * r
        fl = wp.floor(p)
        if fl < 0.0:
            fl = 0.0
        if fl > r - 1.0:
            fl = r - 1.0
        t = p - fl
        if smooth[d] == 1:
            t = t * t * (3.0 - 2.0 * t)
        base[d] = int(fl)
        w[d] = t
    for c in range(1 << ndim):
        ww = float(1.0)
        cc = wp.vec4i(0, 0, 0, 0)
        for d in range(ndim):
            if ((c >> d) & 1) == 1:
                ww = ww * w[d]
                cc[d] = base[d] + 1
            else:
                ww = ww * (1.0 - w[d])
                cc[d] = base[d]
        idx = _corner_index(cc[0], cc[1], cc[2], cc[3],
                            res[l, 0], res[l, 1], res[l, 2], res[l, 3],
                            ndim, dense[l], tsz[l]) + off[l]
        wp.atomic_add(grad_table, idx, 0, ww * g0)
        wp.atomic_add(grad_table, idx, 1, ww * g1)


class _WarpEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, meta):
        B = x.shape[0]
        L, ndim = meta["n_levels"], meta["ndim"]
        out = torch.empty(B, L * F_FIXED, device=x.device, dtype=torch.float32)
        wp.launch(hash_forward, dim=(B, L), inputs=[
            wp.from_torch(x.contiguous()), wp.from_torch(table),
            meta["res"], meta["off"], meta["tsz"], meta["dense"],
            meta["smooth"], meta["gain"], ndim, wp.from_torch(out)])
        ctx.save_for_backward(x, table)
        ctx.meta = meta
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, table = ctx.saved_tensors
        meta = ctx.meta
        gt = None
        if table.requires_grad:
            gt = torch.zeros_like(table)
            wp.launch(hash_backward, dim=(x.shape[0], meta["n_levels"]), inputs=[
                wp.from_torch(x.contiguous()), wp.from_torch(grad_out.contiguous()),
                meta["res"], meta["off"], meta["tsz"], meta["dense"],
                meta["smooth"], meta["gain"], meta["ndim"], wp.from_torch(gt)])
        if x.requires_grad:
            raise NotImplementedError(
                "the warp encoder has no input gradient; the registration pages "
                "need one and should use encoder.implementation: default")
        return None, gt, None


class WarpHashGrid(nn.Module):
    """Drop-in for MultiResHashGrid, F fixed at 2, no input gradient.

    Carries the same attributes the pages read -- resolutions, dense,
    level_offsets, n_entries, table, level_gain -- so a page or a benchmark
    cannot tell which one it holds except by timing it.
    """

    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, per_level_scale=1.5,
                 max_resolution=None, interpolation="linear", hash_shuffle=True):
        super().__init__()
        if n_features_per_level != F_FIXED:
            raise ValueError(f"the warp kernel is written for F={F_FIXED}")
        if not hash_shuffle:
            raise ValueError("the warp kernel implements the xor-primes hash only")
        D = n_input_dims
        interp = ([interpolation] * D if isinstance(interpolation, str)
                  else list(interpolation))
        base = _per_axis(base_resolution, D, "base_resolution")
        scale = _per_axis(per_level_scale, D, "per_level_scale")
        cap = _per_axis(2 ** 24 if max_resolution is None else max_resolution, D,
                        "max_resolution")
        table_size = 2 ** log2_hashmap_size

        resolutions, offsets, dense = [], [], []
        offset = 0
        for lvl in range(n_levels):
            r = tuple(max(1, min(int(round(base[d] * scale[d] ** lvl)), int(cap[d])))
                      for d in range(D))
            nodes = 1
            for q in r:
                nodes *= q + 1
            is_dense = nodes <= table_size
            resolutions.append(r)
            dense.append(is_dense)
            offsets.append(offset)
            offset += nodes if is_dense else table_size
        self.n_input_dims, self.n_levels = D, n_levels
        self.n_features_per_level = F_FIXED
        self.n_output_dims = n_levels * F_FIXED
        self.resolutions, self.dense, self.level_offsets = resolutions, dense, offsets
        self.n_entries = offset
        self.per_level_scale = scale
        self.interpolation = interpolation
        self.hash_shuffle = True

        self.table = nn.Parameter(torch.empty(offset, F_FIXED))
        nn.init.uniform_(self.table, -1e-4, 1e-4)
        self.register_buffer("level_gain", torch.ones(n_levels), persistent=False)
        self._interp = interp
        self._meta = None

    def _build_meta(self, device):
        r = np.zeros((self.n_levels, 4), np.int32)
        for l, res in enumerate(self.resolutions):
            r[l, :len(res)] = res
        sm = np.zeros(4, np.int32)
        for d, i in enumerate(self._interp):
            sm[d] = 1 if i == "smoothstep" else 0
        tsz = np.array([(self.level_offsets[l + 1] if l + 1 < self.n_levels
                         else self.n_entries) - self.level_offsets[l]
                        for l in range(self.n_levels)], np.int32)
        dev = wp.device_from_torch(device)
        return {
            "n_levels": self.n_levels, "ndim": self.n_input_dims,
            "res": wp.array(r, dtype=wp.int32, device=dev, ndim=2),
            "off": wp.array(np.array(self.level_offsets, np.int32), dtype=wp.int32,
                            device=dev),
            "tsz": wp.array(tsz, dtype=wp.int32, device=dev),
            "dense": wp.array(np.array(self.dense, np.int32), dtype=wp.int32,
                              device=dev),
            "smooth": wp.array(sm, dtype=wp.int32, device=dev),
        }

    def set_level_window(self, alpha: float) -> None:
        l = torch.arange(self.n_levels, dtype=torch.float32,
                         device=self.level_gain.device)
        self.level_gain.copy_((alpha - l).clamp(0.0, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, self.n_input_dims)
        if self._meta is None or self._meta.get("dev") != str(x.device):
            self._meta = self._build_meta(x.device)
            self._meta["dev"] = str(x.device)
        meta = dict(self._meta)
        meta["gain"] = wp.from_torch(self.level_gain.contiguous())
        return _WarpEncode.apply(x, self.table, meta).reshape(
            *lead, self.n_output_dims)

    def extra_repr(self) -> str:
        n_hash = sum(1 for d in self.dense if not d)
        return (f"warp, D={self.n_input_dims}, L={self.n_levels}, F={F_FIXED}, "
                f"{n_hash}/{self.n_levels} levels hashed, "
                f"{self.n_entries * F_FIXED / 1e6:.2f}M table params")
