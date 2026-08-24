"""Hash encoding + MLP decoder, with autograd helpers for input derivatives."""

from __future__ import annotations

import torch
import torch.nn as nn

from .hashgrid import MultiResHashGrid

_ACTIVATIONS = {
    "relu": nn.ReLU,          # instant-NGP default; second derivative is 0 a.e.
    "gelu": nn.GELU,          # smooth, use when you need d2f/dx2
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
}


class NGPField(nn.Module):
    """x in [0, 1]^D -> (B, n_output).

    Args:
        n_input_dims / n_output_dims: D and the number of channels (3 = RGB).
        n_neurons / n_hidden_layers: decoder MLP shape.
        activation: hidden activation; use a smooth one ("gelu") together with
            interpolation="smoothstep" if you intend to take second derivatives.
        output_activation: "sigmoid" bounds the output to [0, 1], "none" leaves
            it free.
        **grid_kwargs: forwarded to MultiResHashGrid.
    """

    def __init__(
        self,
        n_input_dims: int = 2,
        n_output_dims: int = 3,
        n_neurons: int = 64,
        n_hidden_layers: int = 2,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        **grid_kwargs,
    ):
        super().__init__()
        self.encoding = MultiResHashGrid(n_input_dims=n_input_dims, **grid_kwargs)

        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = [nn.Linear(self.encoding.n_output_dims, n_neurons), act()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act()]
        layers += [nn.Linear(n_neurons, n_output_dims)]
        if output_activation == "sigmoid":
            layers += [nn.Sigmoid()]
        elif output_activation != "none":
            raise ValueError(f"unknown output_activation {output_activation!r}")
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(x))

    def n_parameters(self) -> tuple[int, int]:
        """(encoding params, decoder params)."""
        enc = self.encoding.table.numel()
        return enc, sum(p.numel() for p in self.mlp.parameters())


def save_checkpoint(path: str, model: NGPField, model_kwargs: dict, **extra) -> None:
    torch.save({"model_kwargs": model_kwargs, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str, device="cpu") -> tuple[NGPField, dict]:
    """-> (model in eval mode, the full checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = NGPField(**ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def jacobian(model, x: torch.Tensor, channel: int = 0, create_graph: bool = False):
    """df_c/dx at x. Returns (values (B, 1), grad (B, D)).

    x need not require grad on entry -- a leaf copy is made.
    """
    x = x.detach().requires_grad_(True)
    y = model(x)[:, channel : channel + 1]
    (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    return y.detach(), g


def laplacian(model, x: torch.Tensor, channel: int = 0, dims=None) -> torch.Tensor:
    """sum_d d2f_c/dx_d2 over `dims` (default: all input dims). Returns (B, 1).

    Needs a C^1 encoding (interpolation="smoothstep") and a smooth activation;
    with linear interpolation + ReLU the result is identically zero.
    """
    x = x.detach().requires_grad_(True)
    y = model(x)[:, channel : channel + 1]
    (g,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    dims = range(x.shape[1]) if dims is None else dims
    lap = torch.zeros_like(y)
    for d in dims:
        (gg,) = torch.autograd.grad(g[:, d].sum(), x, create_graph=True)
        lap = lap + gg[:, d : d + 1]
    return lap
