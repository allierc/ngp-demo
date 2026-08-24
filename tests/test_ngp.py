"""Correctness checks for the encoder and the analytic field.

Run:  python -m pytest tests/ -q      (or just: python tests/test_ngp.py)
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngp import AdvDiffField, MultiResHashGrid, NGPField, laplacian  # noqa: E402


def test_shapes_and_levels():
    g = MultiResHashGrid(n_input_dims=2, n_levels=6, n_features_per_level=2,
                         log2_hashmap_size=10, base_resolution=8, per_level_scale=2.0)
    x = torch.rand(97, 2)
    assert g(x).shape == (97, 12)
    # 2^10 = 1024 entries; nodes are (res+1)^2 = 81, 289, 1089, ... so the
    # res-32 level (1089 > 1024) is the first that has to hash.
    assert g.dense == [True, True, False, False, False, False], g.dense
    assert g(torch.rand(4, 5, 2)).shape == (4, 5, 12)


def test_per_axis_resolution_and_cap():
    """The time axis can be given its own growth rate and a hard cap."""
    g = MultiResHashGrid(n_input_dims=3, n_levels=8, n_features_per_level=2,
                         log2_hashmap_size=16, base_resolution=(8, 8, 2),
                         per_level_scale=(1.5, 1.5, 1.5), max_resolution=(None or 10**6, 10**6, 16))
    res_t = [r[2] for r in g.resolutions]
    assert max(res_t) == 16, res_t                      # cap respected
    assert res_t == sorted(res_t), res_t                # still monotone
    assert g.resolutions[-1][0] > 16                    # x/y kept refining
    assert g(torch.rand(32, 3)).shape == (32, 16)


def test_partition_of_unity():
    """Corner weights sum to 1, so a constant table gives back that constant."""
    g = MultiResHashGrid(n_input_dims=3, n_levels=3, n_features_per_level=1,
                         log2_hashmap_size=16, base_resolution=4, per_level_scale=1.5)
    with torch.no_grad():
        g.table.fill_(0.7)
    out = g(torch.rand(64, 3))
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-6)


def test_endpoints_are_covered():
    """x = 1 must read the last node, not fall back into the previous cell."""
    g = MultiResHashGrid(n_input_dims=1, n_levels=1, n_features_per_level=1,
                         log2_hashmap_size=16, base_resolution=4, per_level_scale=2.0)
    with torch.no_grad():
        g.table.zero_()
        g.table[4] = 1.0          # last node of the res-4 grid (nodes 0..4)
    assert torch.allclose(g(torch.ones(1, 1)), torch.ones(1, 1))
    assert torch.allclose(g(torch.zeros(1, 1)), torch.zeros(1, 1))


def test_gradcheck_inputs_and_table():
    """Double-precision gradcheck w.r.t. both the coordinates and the table."""
    torch.manual_seed(0)
    g = MultiResHashGrid(n_input_dims=2, n_levels=3, n_features_per_level=2,
                         log2_hashmap_size=8, base_resolution=4,
                         per_level_scale=2.0).double()
    with torch.no_grad():
        g.table.uniform_(-1.0, 1.0)
    # Points kept away from cell boundaries: the linear interpolant is only
    # piecewise smooth, and finite differences straddle the kinks otherwise.
    x = (torch.rand(8, 2, dtype=torch.float64) * 0.5 + 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a: g(a).sum(), (x,), eps=1e-6, atol=1e-6)

    xd = x.detach()
    f_table = lambda w: torch.func.functional_call(g, {"table": w}, (xd,)).sum()  # noqa: E731
    w0 = g.table.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f_table, (w0,), eps=1e-6, atol=1e-6)
    # Table gradient: non-zero and confined to the corners actually touched.
    g.zero_grad()
    g(x.detach()).sum().backward()
    touched = (g.table.grad.abs().sum(1) > 0).sum().item()
    assert 0 < touched <= 8 * 4 * 3


def test_smoothstep_is_c1_linear_is_not():
    """Second derivative through the encoding: zero for linear, non-zero for smoothstep."""
    for interp, expect_zero in (("linear", True), ("smoothstep", False)):
        torch.manual_seed(0)
        m = NGPField(n_input_dims=2, n_output_dims=1, n_neurons=16, n_hidden_layers=1,
                     activation="gelu", output_activation="none", n_levels=4,
                     n_features_per_level=2, log2_hashmap_size=12, base_resolution=4,
                     per_level_scale=2.0, interpolation=interp)
        with torch.no_grad():
            m.encoding.table.uniform_(-1.0, 1.0)
        x = torch.rand(256, 2) * 0.8 + 0.1
        # Pure encoding curvature: linearise the decoder away by summing features.
        enc = lambda a: m.encoding(a).sum(1, keepdim=True)  # noqa: E731
        a = x.clone().requires_grad_(True)
        (g1,) = torch.autograd.grad(enc(a).sum(), a, create_graph=True)
        (g2,) = torch.autograd.grad(g1[:, 0].sum(), a, create_graph=True)
        curv = g2[:, 0].abs().max().item()
        if expect_zero:
            assert curv == 0.0, f"linear interp should have zero d2/dx2, got {curv}"
        else:
            assert curv > 1e-3, f"smoothstep should have non-zero d2/dx2, got {curv}"
        # The full model's Laplacian helper must run either way.
        assert laplacian(m, x, dims=(0, 1)).shape == (256, 1)


def test_per_axis_interpolation():
    """Smoothstep on x only: zero curvature along y, non-zero along x."""
    torch.manual_seed(0)
    g = MultiResHashGrid(n_input_dims=2, n_levels=3, n_features_per_level=2,
                         log2_hashmap_size=12, base_resolution=4, per_level_scale=2.0,
                         interpolation=("smoothstep", "linear"))
    with torch.no_grad():
        g.table.uniform_(-1.0, 1.0)
    x = (torch.rand(256, 2) * 0.8 + 0.1).requires_grad_(True)
    (g1,) = torch.autograd.grad(g(x).sum(), x, create_graph=True)
    curv = []
    for d in (0, 1):
        (g2,) = torch.autograd.grad(g1[:, d].sum(), x, create_graph=True)
        curv.append(g2[:, d].abs().max().item())
    assert curv[0] > 1e-3, curv
    assert curv[1] == 0.0, curv


def test_field_matches_its_own_pde():
    """The analytic field satisfies u_t + c.grad u - nu lap u = 0 to machine precision."""
    f = AdvDiffField(n_modes=12, k_max=6, nu=2e-3, seed=1)
    xyt = torch.rand(1000, 3, dtype=torch.float32)
    g = f.grad(xyt)
    res = g[:, 2:3] + (g[:, :2] * f.c).sum(1, keepdim=True) - f.nu * f.laplacian(xyt)
    rms_u = f(xyt).std().item()
    assert res.abs().max().item() < 1e-3 * max(rms_u, 1.0), res.abs().max().item()


def test_field_analytic_grad_matches_autograd():
    f = AdvDiffField(n_modes=12, k_max=6, nu=2e-3, seed=1)
    xyt = torch.rand(64, 3, dtype=torch.float64).requires_grad_(True)
    f.k, f._amp, f.phase, f.decay, f.c = (t.double() for t in
                                          (f.k, f._amp, f.phase, f.decay, f.c))
    (auto,) = torch.autograd.grad(f(xyt).sum(), xyt)
    assert torch.allclose(auto, f.grad(xyt.detach()), atol=1e-8)
    lap_auto = torch.stack([
        torch.autograd.grad(torch.autograd.grad(f(xyt).sum(), xyt, create_graph=True)[0][:, d].sum(),
                            xyt, retain_graph=True)[0][:, d] for d in (0, 1)]).sum(0)
    assert torch.allclose(lap_auto.unsqueeze(1), f.laplacian(xyt.detach()), atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
