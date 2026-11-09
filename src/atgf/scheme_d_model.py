"""Frozen Scheme D / Temporal Context Encoder v2.0.

This module preserves the canonical static branches and changes only the gate
input for the frozen temporal-context ablations. Temporal metadata is limited
to as-of information ages and the historical-event availability indicator.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


CORE_ABLATION_MODES = (
    "ordinary",
    "macro_context",
    "micro_context",
    "full",
)
SECONDARY_DIAGNOSTIC_MODE = "secondary_time_only"
ALLOWED_MODES = CORE_ABLATION_MODES + (SECONDARY_DIAGNOSTIC_MODE,)


def _as_column(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1]")
    if not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    return value


def encode_macro_age(days_since_policy: torch.Tensor) -> torch.Tensor:
    """Encode nonnegative policy information age in real calendar days."""
    age = _as_column(days_since_policy, "days_since_policy")
    if not torch.isfinite(age).all():
        raise ValueError("days_since_policy must be finite")
    if (age < 0).any():
        raise ValueError("days_since_policy must be nonnegative")
    return torch.log1p(age) / math.log(181.0)


def encode_micro_age(
    days_since_event: torch.Tensor,
    has_available_event: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode event age and return it with a distinct availability indicator."""
    age = _as_column(days_since_event, "days_since_event")
    indicator = _as_column(has_available_event, "has_available_event").to(
        device=age.device,
        dtype=age.dtype,
    )
    if age.shape[0] != indicator.shape[0]:
        raise ValueError("event age and availability indicator batch sizes differ")
    if not torch.isfinite(indicator).all():
        raise ValueError("has_available_event must be finite")
    if not torch.logical_or(indicator == 0, indicator == 1).all():
        raise ValueError("has_available_event must contain only 0 or 1")

    available = indicator.to(dtype=torch.bool)
    available_ages = age[available]
    if available_ages.numel() and not torch.isfinite(available_ages).all():
        raise ValueError("available-event ages must be finite")
    if available_ages.numel() and (available_ages < 0).any():
        raise ValueError("available-event ages must be nonnegative")

    safe_age = torch.where(available, age, torch.zeros_like(age))
    encoded = torch.log1p(safe_age) / math.log(61.0)
    encoded = torch.where(available, encoded, torch.zeros_like(encoded))
    return encoded, indicator


class MacroContextEncoder(nn.Module):
    """Frozen 17-to-8-to-4 macro context encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(17, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, h_macro: torch.Tensor, encoded_age: torch.Tensor) -> torch.Tensor:
        if h_macro.ndim != 2 or h_macro.shape[1] != 16:
            raise ValueError("h_macro must have shape [batch, 16]")
        encoded_age = _as_column(encoded_age, "encoded_macro_age").to(
            device=h_macro.device,
            dtype=h_macro.dtype,
        )
        if h_macro.shape[0] != encoded_age.shape[0]:
            raise ValueError("macro representation and age batch sizes differ")
        return self.network(torch.cat([h_macro, encoded_age], dim=1))


class MicroContextEncoder(nn.Module):
    """Frozen 18-to-8-to-4 micro context encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(18, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    @staticmethod
    def build_input(
        h_micro: torch.Tensor,
        encoded_age: torch.Tensor,
        event_indicator: torch.Tensor,
    ) -> torch.Tensor:
        if h_micro.ndim != 2 or h_micro.shape[1] != 16:
            raise ValueError("h_micro must have shape [batch, 16]")
        encoded_age = _as_column(encoded_age, "encoded_micro_age").to(
            device=h_micro.device,
            dtype=h_micro.dtype,
        )
        event_indicator = _as_column(event_indicator, "event_indicator").to(
            device=h_micro.device,
            dtype=h_micro.dtype,
        )
        if not (h_micro.shape[0] == encoded_age.shape[0] == event_indicator.shape[0]):
            raise ValueError("micro representation and temporal input batch sizes differ")
        return torch.cat([h_micro, encoded_age, event_indicator], dim=1)

    def forward(
        self,
        h_micro: torch.Tensor,
        encoded_age: torch.Tensor,
        event_indicator: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(self.build_input(h_micro, encoded_age, event_indicator))


class TemporalContextGate(nn.Module):
    """Single gate for one frozen core ablation or secondary diagnostic."""

    INPUT_DIMS = {
        "ordinary": 64,
        "macro_context": 68,
        "micro_context": 68,
        "full": 72,
        "secondary_time_only": 67,
    }

    def __init__(self, ablation_mode: str) -> None:
        super().__init__()
        if ablation_mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        self.ablation_mode = ablation_mode
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIMS[ablation_mode], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, gate_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = self.INPUT_DIMS[self.ablation_mode]
        if gate_input.ndim != 2 or gate_input.shape[1] != expected:
            raise ValueError(f"gate_input must have shape [batch, {expected}]")
        gate_logits = self.gate_mlp(gate_input)
        return gate_logits, torch.sigmoid(gate_logits)


class SchemeDModel(nn.Module):
    """Canonical multimodal model with the frozen Scheme D gate variants."""

    def __init__(
        self,
        n_macro_features: int,
        n_micro_features: int,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        if ablation_mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        self.ablation_mode = ablation_mode

        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.macro_branch = nn.Sequential(
            nn.Linear(n_macro_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.micro_branch = nn.Sequential(
            nn.Linear(n_micro_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.fusion_macro = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fusion_micro = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.phi_M: Optional[MacroContextEncoder]
        self.phi_E: Optional[MicroContextEncoder]
        self.phi_M = MacroContextEncoder() if ablation_mode in ("macro_context", "full") else None
        self.phi_E = MicroContextEncoder() if ablation_mode in ("micro_context", "full") else None
        self.gate_mlp = TemporalContextGate(ablation_mode).gate_mlp

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    @staticmethod
    def fuse_representations(
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if z_macro.shape != z_micro.shape or z_macro.ndim != 2 or z_macro.shape[1] != 32:
            raise ValueError("z_macro and z_micro must both have shape [batch, 32]")
        alpha = _as_column(alpha, "alpha").to(device=z_macro.device, dtype=z_macro.dtype)
        if alpha.shape[0] != z_macro.shape[0]:
            raise ValueError("alpha and representation batch sizes differ")
        return alpha * z_macro + (1.0 - alpha) * z_micro

    def forward_from_representations(
        self,
        h_macro: torch.Tensor,
        h_micro: torch.Tensor,
        z_macro: torch.Tensor,
        z_micro: torch.Tensor,
        days_since_policy: Optional[torch.Tensor] = None,
        days_since_event: Optional[torch.Tensor] = None,
        has_available_event: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if h_macro.ndim != 2 or h_macro.shape[1] != 16:
            raise ValueError("h_macro must have shape [batch, 16]")
        if h_micro.ndim != 2 or h_micro.shape[1] != 16:
            raise ValueError("h_micro must have shape [batch, 16]")
        if z_macro.shape != z_micro.shape or z_macro.ndim != 2 or z_macro.shape[1] != 32:
            raise ValueError("z_macro and z_micro must both have shape [batch, 32]")
        batch_size = z_macro.shape[0]
        if h_macro.shape[0] != batch_size or h_micro.shape[0] != batch_size:
            raise ValueError("all representation batch sizes must match")

        encoded_macro = None
        encoded_micro = None
        event_indicator = None
        macro_context = None
        micro_context = None

        gate_parts = [z_macro, z_micro]
        if self.ablation_mode in ("macro_context", "full", SECONDARY_DIAGNOSTIC_MODE):
            if days_since_policy is None:
                raise ValueError("days_since_policy is required for this mode")
            encoded_macro = encode_macro_age(days_since_policy).to(
                device=z_macro.device,
                dtype=z_macro.dtype,
            )
        if self.ablation_mode in ("micro_context", "full", SECONDARY_DIAGNOSTIC_MODE):
            if days_since_event is None or has_available_event is None:
                raise ValueError("micro temporal inputs are required for this mode")
            encoded_micro, event_indicator = encode_micro_age(
                days_since_event,
                has_available_event,
            )
            encoded_micro = encoded_micro.to(device=z_micro.device, dtype=z_micro.dtype)
            event_indicator = event_indicator.to(device=z_micro.device, dtype=z_micro.dtype)

        if self.ablation_mode in ("macro_context", "full"):
            if self.phi_M is None or encoded_macro is None:
                raise RuntimeError("macro context encoder is unavailable")
            macro_context = self.phi_M(h_macro, encoded_macro)
            gate_parts.append(macro_context)
        if self.ablation_mode in ("micro_context", "full"):
            if self.phi_E is None or encoded_micro is None or event_indicator is None:
                raise RuntimeError("micro context encoder is unavailable")
            micro_context = self.phi_E(h_micro, encoded_micro, event_indicator)
            gate_parts.append(micro_context)
        if self.ablation_mode == SECONDARY_DIAGNOSTIC_MODE:
            if encoded_macro is None or encoded_micro is None or event_indicator is None:
                raise RuntimeError("secondary diagnostic temporal inputs are unavailable")
            gate_parts.extend([encoded_macro, encoded_micro, event_indicator])

        gate_input = torch.cat(gate_parts, dim=1)
        gate_logits = self.gate_mlp(gate_input)
        alpha = torch.sigmoid(gate_logits)
        z_final = self.fuse_representations(z_macro, z_micro, alpha)
        logits = self.classifier(z_final)
        return {
            "logits": logits,
            "alpha": alpha,
            "gate_logits": gate_logits,
            "z_final": z_final,
            "z_macro": z_macro,
            "z_micro": z_micro,
            "e_macro": encoded_macro,
            "e_micro": encoded_micro,
            "event_indicator": event_indicator,
            "u_macro": macro_context,
            "u_micro": micro_context,
        }

    def forward(
        self,
        x_img: torch.Tensor,
        x_macro: torch.Tensor,
        x_micro: torch.Tensor,
        days_since_policy: Optional[torch.Tensor] = None,
        days_since_event: Optional[torch.Tensor] = None,
        has_available_event: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        h_img = self.image_head(self.image_branch(x_img))
        h_macro = self.macro_branch(x_macro)
        h_micro = self.micro_branch(x_micro)
        z_macro = self.fusion_macro(torch.cat([h_img, h_macro], dim=1))
        z_micro = self.fusion_micro(torch.cat([h_img, h_micro], dim=1))
        return self.forward_from_representations(
            h_macro,
            h_micro,
            z_macro,
            z_micro,
            days_since_policy,
            days_since_event,
            has_available_event,
        )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
