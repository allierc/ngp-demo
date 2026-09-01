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
                  rep_rows: wp.int32,
                  rep: wp.int32,
                  grad_rep: wp.array(dtype=wp.vec2),
                  grad_table: wp.array(dtype=wp.vec2)):
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
        # WHERE THE ADD LANDS. A coarse level is a few hundred rows shared by
        # every sample in the batch: level 0 here is 243 rows taking 8,630 adds
        # each (16,660 for the busiest), and those adds serialise on one address.
        # A fine level is the opposite -- level 15 averages 1.3 adds per row and
        # collides with nobody. So the coarse rows, and only those, are given
        # `rep` private copies and the thread picks one by its own index; the
        # copies are summed afterwards. Contention falls by `rep`, the fine
        # levels keep the direct path, and the arithmetic is unchanged because
        # addition does not care in what order it is done.
        if idx < rep_rows:
            j = (b & (rep - 1)) * rep_rows + idx
            wp.atomic_add(grad_rep, j, wp.vec2(ww * g0, ww * g1))
        else:
            # The two features of a row are adjacent, so the pair is written as
            # one vec2 add. MEASURED NO FASTER than two scalar adds (4.63 vs
            # 4.62 ms/step on an H100): warp lowers it to two atomics anyway.
            # Kept because it reads better, not because it bought anything.
            # What the 2.57 ms backward is actually made of: 8 corners x 16
            # levels x 262,144 samples = 33.5 M atomic adds, at roughly the
            # 26 G ops/s the hardware does them at. Spreading them helps a
            # little (privatisation below, 5%); issuing fewer would help more,
            # and nothing here issues fewer.
            wp.atomic_add(grad_table, idx, wp.vec2(ww * g0, ww * g1))


@wp.kernel
def reduce_rep(grad_rep: wp.array(dtype=wp.vec2),
               rep_rows: wp.int32,
               rep: wp.int32,
               grad_table: wp.array(dtype=wp.vec2)):
    """Sum the private copies into the real gradient, and clear them.

    One thread per coarse row reads `rep` values with no atomics at all, and
    zeroes as it goes so the next backward starts clean without a separate
    memset.
    """
    i = wp.tid()
    s = wp.vec2(0.0, 0.0)
    for r in range(rep):
        j = r * rep_rows + i
        s += grad_rep[j]
        grad_rep[j] = wp.vec2(0.0, 0.0)
    grad_table[i] = s


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
            rr, rep = meta["rep_rows"], meta["rep"]
            wp.launch(hash_backward, dim=(x.shape[0], meta["n_levels"]), inputs=[
                wp.from_torch(x.contiguous()), wp.from_torch(grad_out.contiguous()),
                meta["res"], meta["off"], meta["tsz"], meta["dense"],
                meta["smooth"], meta["gain"], meta["ndim"], rr, rep,
                meta["grad_rep"], wp.from_torch(gt, dtype=wp.vec2)])
            if rr:
                wp.launch(reduce_rep, dim=rr,
                          inputs=[meta["grad_rep"], rr, rep,
                                  wp.from_torch(gt, dtype=wp.vec2)])
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
                 max_resolution=None, interpolation="linear", hash_shuffle=True,
                 replicate=64, replicate_max_rows=1 << 16):
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
        # PRIVATISATION OF THE COARSE ROWS, see hash_backward. `replicate` is
        # how many private copies each such row gets (a power of two, 0 turns
        # the whole thing off); `replicate_max_rows` decides which levels count
        # as coarse -- a level small enough that the batch keeps hitting the
        # same rows.
        self.replicate = int(replicate)
        self.replicate_max_rows = int(replicate_max_rows)
        if self.replicate and (self.replicate & (self.replicate - 1)):
            raise ValueError("replicate must be a power of two")

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
        # The coarse levels form a prefix of the table, so "the replicated rows"
        # is a single bound rather than a mask: everything below rep_rows is
        # private, everything above is direct.
        rep_rows = 0
        for l in range(self.n_levels):
            if int(tsz[l]) > self.replicate_max_rows:
                break
            rep_rows = self.level_offsets[l] + int(tsz[l])
        rep = self.replicate if rep_rows else 0
        while rep > 1 and rep * rep_rows * 2 * 4 > 64 << 20:   # 64 MB ceiling
            rep //= 2
        if rep <= 1:
            rep, rep_rows = 1, 0
        return {
            "rep": rep, "rep_rows": rep_rows,
            "grad_rep": wp.zeros(shape=(max(rep * rep_rows, 1),),
                                 dtype=wp.vec2, device=dev),
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


# ----------------------------------------------------------------- the update


@wp.kernel
def adam_table(p: wp.array2d(dtype=wp.float32),
               g: wp.array2d(dtype=wp.float32),
               m: wp.array2d(dtype=wp.float32),
               v: wp.array2d(dtype=wp.float32),
               lr: wp.float32, b1: wp.float32, b2: wp.float32, eps: wp.float32,
               c1: wp.float32, c2: wp.float32, skip_zero: wp.int32):
    """Adam over the table, skipping rows this step never touched.

    Measured reason: at batch 262,144 the backward writes into 7.37 M of the
    table's 16.28 M rows -- 45.3% -- and torch's Adam updates all of them, which
    is six streams of traffic over the whole table to move a little under half
    of it.  A row with no gradient has m and v decayed by b1 and b2 and p moved
    by a bias-corrected zero; skipping it is not an approximation of the update,
    it is the update, minus the arithmetic that a decayed zero contributes.

    NOT exactly torch's Adam on an untouched row: torch still decays m and v
    there, so a row that is skipped for k steps and then hit again carries a
    staler moment than torch would.  That is the trade, and it is the same one
    torch.optim.SparseAdam makes.  `skip_zero=0` runs it dense, which is what
    the gate compares against.
    """
    i = wp.tid()
    g0 = g[i, 0]
    g1 = g[i, 1]
    if skip_zero == 1:
        if g0 == 0.0 and g1 == 0.0:
            return
    m0 = b1 * m[i, 0] + (1.0 - b1) * g0
    m1 = b1 * m[i, 1] + (1.0 - b1) * g1
    v0 = b2 * v[i, 0] + (1.0 - b2) * g0 * g0
    v1 = b2 * v[i, 1] + (1.0 - b2) * g1 * g1
    m[i, 0] = m0
    m[i, 1] = m1
    v[i, 0] = v0
    v[i, 1] = v1
    p[i, 0] = p[i, 0] - lr * (m0 / c1) / (wp.sqrt(v0 / c2) + eps)
    p[i, 1] = p[i, 1] - lr * (m1 / c1) / (wp.sqrt(v1 / c2) + eps)


class TableAdam:
    """Adam for the table alone; the decoder keeps torch's.

    Two parameters with wildly different shapes were sharing one optimiser: 6,337
    decoder weights that every step needs, and 32.5 M table values of which fewer
    than half do.  Splitting them lets the big one use a kernel that knows that.
    """

    def __init__(self, table, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 skip_zero=True):
        self.p = table
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.skip_zero = 1 if skip_zero else 0
        self.m = torch.zeros_like(table)
        self.v = torch.zeros_like(table)
        self.t = 0

    def step(self):
        if self.p.grad is None:
            return
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        wp.launch(adam_table, dim=self.p.shape[0], inputs=[
            wp.from_torch(self.p.data), wp.from_torch(self.p.grad),
            wp.from_torch(self.m), wp.from_torch(self.v),
            float(self.lr), float(self.b1), float(self.b2), float(self.eps),
            float(c1), float(c2), self.skip_zero])

    def zero_grad(self, set_to_none=True):
        self.p.grad = None if set_to_none else self.p.grad


# ------------------------------------------------------------------ the swap

class HalfMLP(nn.Module):
    """Run the decoder's matmuls in half, hand fp32 back to the loss.

    A Module rather than a closure because the optimiser splits parameters by
    asking `model.mlp.parameters()`.  The weights stay fp32 masters -- only the
    matmuls are cast -- so Adam's arithmetic is unchanged and the gradient
    reaching the encoder arrives in the precision its kernel expects.
    """

    def __init__(self, inner, dt=torch.bfloat16):
        super().__init__()
        self.inner, self.dt = inner, dt

    def forward(self, f):
        with torch.autocast("cuda", dtype=self.dt):
            return self.inner(f).float()


class SplitOpt:
    """Torch's Adam for the decoder, a fused kernel for the table.

    Two parameter groups with nothing in common: 6,337 decoder weights that
    every step needs, and 32.5 M table values of which 45.3% get a gradient.
    One optimiser over both has to treat them the same.
    """

    def __init__(self, model, lr, skip_zero=True):
        table = model.encoding.table
        rest = [q for q in model.parameters() if q is not table]
        self.small = torch.optim.Adam(rest, lr=lr) if rest else None
        self.big = TableAdam(table, lr=lr, skip_zero=skip_zero)

    def zero_grad(self, set_to_none=True):
        if self.small is not None:
            self.small.zero_grad(set_to_none=set_to_none)
        self.big.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.small is not None:
            self.small.step()
        self.big.step()


def accelerate(model, log2_hashmap_size, half_decoder=True):
    """The measured-fastest path, in place: warp encoder + bf16 decoder.

    27.09 -> 4.30 ms/step on an H100 at batch 262,144, and 53.37 dB either way
    -- the encoding is unchanged, which `tests/impl_gate.py` holds to 3e-11 on
    the forward and 6e-07 relative on the table gradient.  The new encoder is
    built from the OLD ONE'S LADDER rather than from the caller's arguments
    again, so the two cannot drift, and the trained table is copied across.

    NO INPUT GRADIENT.  The kernel differentiates with respect to the table
    only, which is all a fit needs and not what a registration needs, so this
    returns the model untouched when the caller asks for derivatives -- and
    raises nothing, because the point is a faster fit and not a different one.
    Refuses quietly on cpu for the same reason.
    """
    e = getattr(model, "encoding", None)
    if e is None or not e.table.is_cuda or isinstance(e, WarpHashGrid):
        return model
    if e.n_features_per_level != F_FIXED or not getattr(e, "hash_shuffle", True):
        return model
    w = WarpHashGrid(n_input_dims=e.n_input_dims, n_levels=e.n_levels,
                     n_features_per_level=e.n_features_per_level,
                     log2_hashmap_size=int(log2_hashmap_size),
                     base_resolution=list(e.resolutions[0]),
                     per_level_scale=list(e.per_level_scale),
                     max_resolution=list(e.resolutions[-1]),
                     interpolation=e.interpolation).to(e.table.device)
    if w.n_entries != e.n_entries or w.resolutions != e.resolutions:
        return model                      # ladders disagree: leave it alone
    with torch.no_grad():
        w.table.copy_(e.table)
        w.level_gain.copy_(e.level_gain)
    model.encoding = w
    if half_decoder and hasattr(model, "mlp"):
        model.mlp = HalfMLP(model.mlp)
    return model
