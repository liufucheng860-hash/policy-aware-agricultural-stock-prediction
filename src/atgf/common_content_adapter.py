"""Shared frozen content adapter and classification head."""

from __future__ import annotations

import torch
from torch import nn


MARKET_CHANNELS = 6
SEQUENCE_LENGTH = 60
MACRO_DIM = 9
MICRO_DIM = 23
CONTENT_INPUT_DIM = MACRO_DIM + MICRO_DIM
CONTENT_EMBED_DIM = 16
MARKET_EMBED_DIM = 40
FUSION_DIM = MARKET_EMBED_DIM + CONTENT_EMBED_DIM


def initialize_frozen_module(module: nn.Module) -> None:
    """Apply the initialization frozen by the final benchmark protocol."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def initialize_attention(module: nn.Module) -> None:
    """Initialize attention projections explicitly and reproducibly."""
    if isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight, gain=1.0)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)


def validate_frozen_inputs(
    x_market: torch.Tensor,
    x_macro: torch.Tensor,
    x_micro: torch.Tensor,
) -> None:
    """Reject any tensor shape outside the frozen three-input interface."""
    if x_market.ndim != 3 or tuple(x_market.shape[1:]) != (MARKET_CHANNELS, SEQUENCE_LENGTH):
        raise ValueError(f"x_market must have shape [B,{MARKET_CHANNELS},{SEQUENCE_LENGTH}]")
    batch = x_market.shape[0]
    if x_macro.ndim != 2 or tuple(x_macro.shape) != (batch, MACRO_DIM):
        raise ValueError(f"x_macro must have shape [B,{MACRO_DIM}]")
    if x_micro.ndim != 2 or tuple(x_micro.shape) != (batch, MICRO_DIM):
        raise ValueError(f"x_micro must have shape [B,{MICRO_DIM}]")
    if not (x_market.is_floating_point() and x_macro.is_floating_point() and x_micro.is_floating_point()):
        raise TypeError("all model inputs must be floating-point tensors")


class CommonContentAdapter(nn.Module):
    """Frozen 32 -> 32 -> 16 joint macro/micro adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CONTENT_INPUT_DIM, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, CONTENT_EMBED_DIM, bias=True),
            nn.ReLU(),
        )

    def forward(self, x_macro: torch.Tensor, x_micro: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((x_macro, x_micro), dim=1))


class FrozenClassifierHead(nn.Module):
    """Frozen 56 -> 24 -> 2 binary-classification head."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FUSION_DIM, 24, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 2, bias=True),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.network(fused)

