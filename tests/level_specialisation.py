"""Does a hash grid put its fine levels where the fine structure is?

    python tests/level_specialisation.py

It does not, and this is the measurement that says so. The claim is common
enough to be worth a standing check: fit an NGPField by plain regression -- no
registration loss, no regulariser, nothing to confound it -- to a field that is
smooth on its left half and ten times finer on its right at equal amplitude,
then ask where each level's contribution lives.

Every level contributes about equally to both halves. The level with 344 cells
per axis does as much work on the smooth side as on the fine side. Nothing in
the architecture would do otherwise: all levels are queried everywhere, summed
into one feature vector, and a fine level represents a smooth function perfectly
well by varying its entries slowly.

The consequence is practical. A level map separates regions that carry signal
from regions that do not; it does not report the local scale of the structure.
If capacity should sit at a particular scale, cap the finest level there rather
than expecting the hierarchy to discover it.
"""
import sys; sys.path.insert(0, ".")
import torch
from ngp import NGPField
from ngp.deform import pixel_grid

dev = torch.device("cuda")
H = W = 384

torch.manual_seed(0)
c_coarse = torch.rand(6, 2, generator=torch.Generator().manual_seed(1)).to(dev)
c_coarse[:, 0] *= 0.5
c_fine = torch.rand(120, 2, generator=torch.Generator().manual_seed(2)).to(dev)
c_fine[:, 0] = c_fine[:, 0] * 0.5 + 0.5
s_coarse, s_fine = 0.12, 0.012

def f(xy):
    a = torch.exp(-0.5 * (((xy[:, None] - c_coarse[None]) / s_coarse) ** 2).sum(-1)).sum(1)
    b = torch.exp(-0.5 * (((xy[:, None] - c_fine[None]) / s_fine) ** 2).sum(-1)).sum(1)
    return (a / a.max() + b / b.max()).unsqueeze(1)

m = NGPField(n_input_dims=2, n_output_dims=1, n_neurons=64, n_hidden_layers=2,
             activation="gelu", output_activation="none", n_levels=12,
             n_features_per_level=2, log2_hashmap_size=17, base_resolution=8,
             per_level_scale=1.6, interpolation="smoothstep").to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
for step in range(1500):
    xy = torch.rand(1 << 15, 2, device=dev)
    loss = ((m(xy) - f(xy)) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"final regression loss {loss.item():.3e}")

xy = pixel_grid(H, W, dev)
enc = m.encoding
prev, deltas = None, []
with torch.no_grad():
    for k in range(enc.n_levels + 1):
        enc.set_level_window(float(k))
        out = m(xy).reshape(H, W)
        if prev is not None:
            deltas.append((out - prev).abs())
        prev = out
    enc.set_level_window(float(enc.n_levels))
D = torch.stack(deltas)                                   # (L, H, W)
left, right = D[:, :, :W // 2], D[:, :, W // 2:]
print(f"\n  cells/axis   contribution left (smooth)   right (fine)   right/left")
for l in range(enc.n_levels):
    a, b = float(left[l].mean()), float(right[l].mean())
    print(f"   {enc.resolutions[l][0]:6d}      {a:14.5f}          {b:9.5f}     {b/max(a,1e-9):7.2f}")

ratios = [float(right[l].mean()) / max(float(left[l].mean()), 1e-9)
          for l in range(enc.n_levels)]
spread = max(ratios) / min(ratios)
print(f"\nfine/smooth contribution ratio spans {min(ratios):.2f} to {max(ratios):.2f} "
      f"({spread:.1f}x). Specialisation by frequency would put the fine levels' ratio "
      f"orders of magnitude above the coarse levels'.")
assert spread < 5.0, "levels appear to specialise -- the claim in the docstring is stale"
print("no level specialisation, as documented")
