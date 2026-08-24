"""Multiresolution hash encoding (Müller et al. 2022, "Instant NGP"), pure PyTorch.

One learnable feature table per level, indexed either densely (when the level's
grid fits in the table) or through the instant-NGP spatial hash (when it does
not).  A query reads the 2**D corners of its cell at every level, interpolates
them, and concatenates the levels.

Differences from tiny-cuda-nn that matter here:
  * everything is plain autograd ops, so gradients flow to the *inputs* as well
    as the table, and second derivatives (double backward) are available;
  * `interpolation="smoothstep"` replaces the multilinear weight w by
    3w^2-2w^3, which makes the encoding C^1 -- with plain linear weights the
    encoding is only C^0 and its second derivative is zero almost everywhere,
    which silently kills any Laplacian computed through it;
  * `base_resolution`, `per_level_scale` and `max_resolution` may be given per
    axis.  For an (x, y, t) field the time axis must usually be capped at the
    frame spacing, otherwise the finest levels resolve each observed frame
    separately and the fit stops interpolating between them.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

# instant-NGP's spatial hash primes (pi_1 = 1 by construction).
_PRIMES = (1, 2654435761, 805459861, 3674653429)


def _per_axis(v, D: int, name: str) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * D
    v = list(v)
    if len(v) != D:
        raise ValueError(f"{name} must be a scalar or a sequence of {D} values, got {v}")
    return [float(x) for x in v]


def _hash(corners: torch.Tensor, table_size: int) -> torch.Tensor:
    """corners: (..., D) integer lattice coordinates -> (...) table indices."""
    h = torch.zeros(corners.shape[:-1], dtype=torch.int64, device=corners.device)
    for d in range(corners.shape[-1]):
        h = h ^ (corners[..., d] * _PRIMES[d])
    return h % table_size


class MultiResHashGrid(nn.Module):
    """(B, D) coordinates in [0, 1]^D -> (B, n_levels * n_features_per_level).

    Args:
        n_input_dims: D, up to 4 (2 = image, 3 = 2D+t field).
        n_levels: number of resolution levels L.
        n_features_per_level: F, learnable features per grid node.
        log2_hashmap_size: log2 of the per-level table capacity T.
        base_resolution: N_min, cells per axis at the coarsest level.
            Scalar or one value per axis.
        per_level_scale: b, geometric growth of resolution per level.
            Scalar or one value per axis.
        max_resolution: cells per axis the refinement stops at.
            Scalar or one value per axis; None means unbounded.
        interpolation: "linear" (C^0) or "smoothstep" (C^1, needed for
            second derivatives through the encoding), or one such string per
            axis.  Smoothstep is not free: its weight derivative 6w(1-w)
            vanishes at every cell boundary and peaks mid-cell, so on an axis
            whose cells are aligned with your samples -- a time axis capped at
            the frame spacing, say -- it forces the first derivative to zero at
            the sample times and inflates it in between.  Differentiate that
            axis linearly and reserve smoothstep for the axes whose curvature
            you need.
    """

    def __init__(
        self,
        n_input_dims: int = 2,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int | Sequence[float] = 16,
        per_level_scale: float | Sequence[float] = 1.5,
        max_resolution: int | Sequence[float] | None = None,
        interpolation: str | Sequence[str] = "linear",
    ):
        super().__init__()
        if not 1 <= n_input_dims <= 4:
            raise ValueError(f"n_input_dims must be in 1..4, got {n_input_dims}")
        interp = ([interpolation] * n_input_dims if isinstance(interpolation, str)
                  else list(interpolation))
        if len(interp) != n_input_dims or any(
                i not in ("linear", "smoothstep") for i in interp):
            raise ValueError(f"bad interpolation {interpolation!r}")

        D = n_input_dims
        self.n_input_dims = D
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.n_output_dims = n_levels * n_features_per_level
        self.interpolation = interpolation
        self._interp = interp
        self._smooth_dims = [d for d, i in enumerate(interp) if i == "smoothstep"]

        base = _per_axis(base_resolution, D, "base_resolution")
        scale = _per_axis(per_level_scale, D, "per_level_scale")
        cap = _per_axis(2**24 if max_resolution is None else max_resolution, D,
                        "max_resolution")
        table_size = 2**log2_hashmap_size

        # Per level: cells per axis, whether the level is dense, and where its
        # slice of the shared table starts.
        resolutions: list[tuple[int, ...]] = []
        offsets: list[int] = []
        dense: list[bool] = []
        offset = 0
        for lvl in range(n_levels):
            res = tuple(max(1, min(int(round(base[d] * scale[d] ** lvl)), int(cap[d])))
                        for d in range(D))
            n_nodes = 1
            for r in res:
                n_nodes *= r + 1
            is_dense = n_nodes <= table_size
            resolutions.append(res)
            dense.append(is_dense)
            offsets.append(offset)
            offset += n_nodes if is_dense else table_size
        self.resolutions = resolutions
        self.dense = dense
        self.level_offsets = offsets
        self.n_entries = offset

        # One flat table for all levels: (sum_l entries_l, F).
        self.table = nn.Parameter(torch.empty(offset, n_features_per_level))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

        # (L, D) resolutions as floats, and the (2**D, D) corners of a unit cell.
        self.register_buffer("res_f", torch.tensor(resolutions, dtype=torch.float32),
                             persistent=False)
        corners = torch.stack(
            torch.meshgrid(*[torch.arange(2) for _ in range(D)], indexing="ij"), dim=-1
        ).reshape(-1, D)
        self.register_buffer("corner_offsets", corners, persistent=False)
        self.register_buffer("_smooth_mask",
                             torch.tensor([[i == "smoothstep" for i in interp]]),
                             persistent=False)
        # Coarse-to-fine window (BARF/Nerfies): per-level gains, all on by default.
        self.register_buffer("level_gain", torch.ones(n_levels), persistent=False)

    def set_level_window(self, alpha: float) -> None:
        """Enable levels up to `alpha`, with the fractional level faded in.

        alpha = 4.0 gives levels 0-3 at full weight and everything above off;
        alpha = 4.5 fades level 4 in at half.  Registration needs this: releasing
        the fine levels only once the coarse ones have converged is what keeps an
        intensity loss out of its nearest local minimum.
        """
        l = torch.arange(self.n_levels, dtype=torch.float32, device=self.level_gain.device)
        self.level_gain.copy_((alpha - l).clamp(0.0, 1.0))

    def extra_repr(self) -> str:
        n_hash = sum(1 for d in self.dense if not d)
        rng = ", ".join(f"{self.resolutions[0][d]}..{self.resolutions[-1][d]}"
                        for d in range(self.n_input_dims))
        return (
            f"D={self.n_input_dims}, L={self.n_levels}, F={self.n_features_per_level}, "
            f"res per axis ({rng}), {n_hash}/{self.n_levels} levels hashed, "
            f"{self.n_entries * self.n_features_per_level / 1e6:.2f}M table params, "
            f"interp={'/'.join(self._interp)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_input_dims:
            raise ValueError(f"expected (..., {self.n_input_dims}), got {tuple(x.shape)}")
        lead, D = x.shape[:-1], self.n_input_dims
        x = x.reshape(-1, D)
        res_f = self.res_f.to(x.dtype)

        feats = []
        for lvl, (res, off, is_dense) in enumerate(
            zip(self.resolutions, self.level_offsets, self.dense)
        ):
            r = res_f[lvl]                             # (D,) cells per axis
            pos = x * r                                # cell coordinates
            # Clamp the cell *before* taking the offset, so that x = 1 lands in
            # the last cell with w = 1 instead of the last cell with w = 0.
            base = torch.minimum(torch.floor(pos).clamp(min=0), r - 1)  # zero grad
            w = pos - base                             # (B, D) in [0, 1], grad = r
            if self._smooth_dims:
                ws = w * w * (3.0 - 2.0 * w)
                if len(self._smooth_dims) == D:
                    w = ws
                else:
                    w = torch.where(self._smooth_mask, ws, w)

            corners = base.long().unsqueeze(1) + self.corner_offsets  # (B, C, D)
            if is_dense:
                idx = torch.zeros(corners.shape[:-1], dtype=torch.int64, device=x.device)
                for d in range(D):
                    idx = idx * (res[d] + 1) + corners[..., d]
            else:
                idx = _hash(corners, self._table_size(lvl))

            # Multilinear weight of each corner: prod_d (w_d or 1-w_d).  Built
            # as an explicit chain of multiplies rather than Tensor.prod: prod's
            # double backward goes through a division and leaks ~1e-2 of
            # spurious curvature in float32, which would swamp the (exactly
            # zero) second derivative of a linear interpolant.
            wc = None
            for d in range(D):
                wd = w[:, d : d + 1]                                 # (B, 1)
                sel = self.corner_offsets[:, d].unsqueeze(0).bool()  # (1, C)
                term = torch.where(sel, wd, 1.0 - wd)                # (B, C)
                wc = term if wc is None else wc * term

            f = self.table[idx + off]                                # (B, C, F)
            out_l = (wc.unsqueeze(-1) * f).sum(dim=1)                # (B, F)
            if self.level_gain[lvl] != 1.0:
                out_l = out_l * self.level_gain[lvl]
            feats.append(out_l)

        return torch.cat(feats, dim=-1).reshape(*lead, self.n_output_dims)

    def _table_size(self, lvl: int) -> int:
        """Capacity of level `lvl`'s slice of the shared table."""
        end = self.level_offsets[lvl + 1] if lvl + 1 < self.n_levels else self.n_entries
        return end - self.level_offsets[lvl]
