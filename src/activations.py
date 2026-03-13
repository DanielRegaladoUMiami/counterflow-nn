"""ChemE-inspired activation functions derived from reaction kinetics."""

import torch
import torch.nn as nn


class MichaelisMenten(nn.Module):
    """f(x) = V_max * x / (K_m + |x|). Smooth, bounded, passes through origin."""
    def __init__(self, dim=1):
        super().__init__()
        self.log_vmax = nn.Parameter(torch.zeros(dim))
        self.log_km = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        vmax = self.log_vmax.exp()
        km = self.log_km.exp()
        return vmax * x / (km + x.abs() + 1e-8)


class Arrhenius(nn.Module):
    """f(x) = x * exp(-E_a / (|x| + eps)). Near-zero for small inputs, then rapidly increasing."""
    def __init__(self, dim=1, eps=0.1):
        super().__init__()
        self.log_ea = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        ea = self.log_ea.exp()
        return x * torch.exp(-ea / (x.abs() + self.eps))


class HillActivation(nn.Module):
    """f(x) = x^n / (K^n + x^n). Sigmoidal with learnable steepness n."""
    def __init__(self, dim=1):
        super().__init__()
        self.log_n = nn.Parameter(torch.zeros(dim))  # n > 0
        self.log_k = nn.Parameter(torch.zeros(dim))  # K > 0

    def forward(self, x):
        n = self.log_n.exp() + 1.0  # ensure n >= 1
        k = self.log_k.exp()
        x_abs = x.abs() + 1e-8
        return x.sign() * (x_abs.pow(n) / (k.pow(n) + x_abs.pow(n)))


class Autocatalytic(nn.Module):
    """f(x) = k * x * (1 - x/C_max). Logistic growth, self-limiting."""
    def __init__(self, dim=1):
        super().__init__()
        self.log_k = nn.Parameter(torch.zeros(dim))
        self.log_cmax = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        k = self.log_k.exp()
        cmax = self.log_cmax.exp()
        return k * x * (1.0 - x / (cmax + 1e-8))
