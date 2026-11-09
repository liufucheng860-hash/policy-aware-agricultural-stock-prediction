import torch
from torch import nn

from .scheme_d_model import SchemeDModel, encode_macro_age, encode_micro_age


class AgeControl(SchemeDModel):
    def __init__(self, n_macro_features=9, n_micro_features=23, *, variant, seed):
        if variant not in ('noage', 'direct'):
            raise ValueError(variant)
        super().__init__(n_macro_features, n_micro_features, ablation_mode='full')
        self.variant = variant
        if variant == 'direct':
            self.phi_M = None
            self.phi_E = None
            # Preserve the ATGF shared initialization and training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1000003 + seed)
                self.gate_mlp = nn.Sequential(nn.Linear(99, 15), nn.ReLU(), nn.Linear(15, 1))

    def forward_from_representations(self, h_macro, h_micro, z_macro, z_micro,
                                     days_since_policy=None, days_since_event=None,
                                     has_available_event=None):
        e_macro = encode_macro_age(days_since_policy).to(z_macro)
        e_micro, indicator = encode_micro_age(days_since_event, has_available_event)
        e_micro, indicator = e_micro.to(z_micro), indicator.to(z_micro)
        u_macro = u_micro = None
        if self.variant == 'noage':
            e_macro = torch.zeros_like(e_macro)
            e_micro = torch.zeros_like(e_micro)
            u_macro = self.phi_M(h_macro, e_macro)
            u_micro = self.phi_E(h_micro, e_micro, indicator)
            gate_input = torch.cat([z_macro, z_micro, u_macro, u_micro], dim=1)
        else:
            gate_input = torch.cat([z_macro, z_micro, h_macro, h_micro,
                                    e_macro, e_micro, indicator], dim=1)
        gate_logits = self.gate_mlp(gate_input)
        alpha = torch.sigmoid(gate_logits)
        z_final = self.fuse_representations(z_macro, z_micro, alpha)
        return dict(logits=self.classifier(z_final), alpha=alpha, gate_logits=gate_logits,
                    z_final=z_final, z_macro=z_macro, z_micro=z_micro,
                    e_macro=e_macro, e_micro=e_micro, event_indicator=indicator,
                    u_macro=u_macro, u_micro=u_micro)
