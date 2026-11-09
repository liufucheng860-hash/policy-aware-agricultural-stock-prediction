"""Frozen Compact PatchTST classifier.

The market backbone preserves PatchTST's channel-independent subseries
patching and shared Transformer encoder. The content adapter and two-logit
head are the minimal multimodal classification adaptation required by the
final benchmark protocol.
"""

from __future__ import annotations

import torch
from torch import nn

from .common_content_adapter import (
    CommonContentAdapter,
    FrozenClassifierHead,
    MARKET_CHANNELS,
    MARKET_EMBED_DIM,
    SEQUENCE_LENGTH,
    initialize_attention,
    initialize_frozen_module,
    validate_frozen_inputs,
)


class CompactPatchTSTClassifier(nn.Module):
    patch_length = 12
    stride = 6
    d_model = MARKET_EMBED_DIM
    n_heads = 4
    n_layers = 2
    d_ff = 80
    dropout = 0.1
    number_of_patches = (SEQUENCE_LENGTH - patch_length) // stride + 1
    expected_trainable_parameters = 30_442

    def __init__(self) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(self.patch_length, self.d_model, bias=True)
        self.positional_embedding = nn.Parameter(torch.empty(1, self.number_of_patches, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5, elementwise_affine=True)
        self.content_adapter = CommonContentAdapter()
        self.classifier = FrozenClassifierHead()
        self.apply(initialize_frozen_module)
        self.apply(initialize_attention)
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def extract_patches(self, x_market: torch.Tensor) -> torch.Tensor:
        """Return channel-independent patches as [B, 6, 9, 12]."""
        return x_market.unfold(dimension=2, size=self.patch_length, step=self.stride)

    def encode_market(self, x_market: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x_market.shape[0]
        patches = self.extract_patches(x_market)
        tokens = self.patch_projection(patches.reshape(batch * MARKET_CHANNELS, self.number_of_patches, self.patch_length))
        tokens = tokens + self.positional_embedding
        encoded = self.final_norm(self.encoder(tokens))
        per_channel = encoded.mean(dim=1).reshape(batch, MARKET_CHANNELS, self.d_model)
        market_embedding = per_channel.mean(dim=1)
        return patches, encoded, market_embedding

    def forward_features(
        self,
        x_market: torch.Tensor,
        x_macro: torch.Tensor,
        x_micro: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_frozen_inputs(x_market, x_macro, x_micro)
        patches, encoded_patches, market_embedding = self.encode_market(x_market)
        content_embedding = self.content_adapter(x_macro, x_micro)
        fusion = torch.cat((market_embedding, content_embedding), dim=1)
        logits = self.classifier(fusion)
        return {
            "patches": patches,
            "encoded_patches": encoded_patches,
            "market_embedding": market_embedding,
            "content_embedding": content_embedding,
            "fusion": fusion,
            "logits": logits,
        }

    def forward(
        self,
        x_market: torch.Tensor,
        x_macro: torch.Tensor,
        x_micro: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_features(x_market, x_macro, x_micro)["logits"]

