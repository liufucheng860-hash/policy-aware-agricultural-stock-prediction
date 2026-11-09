"""Frozen Compact iTransformer classifier.

Each complete 60-step variate history becomes one token. Self-attention is
therefore applied across six variates, preserving the inverted-token core.
The shared content adapter and two-logit head provide the frozen multimodal
classification adaptation.
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


class CompactITransformerClassifier(nn.Module):
    token_count = MARKET_CHANNELS
    token_input_dimension = SEQUENCE_LENGTH
    d_model = MARKET_EMBED_DIM
    n_heads = 4
    n_layers = 2
    d_ff = 80
    dropout = 0.1
    expected_trainable_parameters = 32_002

    def __init__(self) -> None:
        super().__init__()
        self.inverted_embedding = nn.Linear(self.token_input_dimension, self.d_model, bias=True)
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

    def encode_market(self, x_market: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inverted_tokens = self.inverted_embedding(x_market)
        encoded_tokens = self.final_norm(self.encoder(inverted_tokens))
        return inverted_tokens, encoded_tokens, encoded_tokens.mean(dim=1)

    def forward_features(
        self,
        x_market: torch.Tensor,
        x_macro: torch.Tensor,
        x_micro: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        validate_frozen_inputs(x_market, x_macro, x_micro)
        inverted_tokens, encoded_tokens, market_embedding = self.encode_market(x_market)
        content_embedding = self.content_adapter(x_macro, x_micro)
        fusion = torch.cat((market_embedding, content_embedding), dim=1)
        logits = self.classifier(fusion)
        return {
            "inverted_tokens": inverted_tokens,
            "encoded_tokens": encoded_tokens,
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
