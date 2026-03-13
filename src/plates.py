"""
CounterFlow exchange plates — the core building blocks of CFNN.

Each plate performs bidirectional information transfer between
the gas (ascending) and liquid (descending) streams, analogous
to a tray in an absorption or distillation tower.

Chemical Engineering Reference:
    At each differential element dZ of an absorption tower:
        Gas phase:    G_s dY = K_Ya (Y - Y*) dZ
        Liquid phase: L_s dX = K_Xa (X* - X) dZ

    Where Y* = f(X) is the equilibrium relationship.

    Neural analog:
        δ = g - E(l)           # driving force (Y - Y*)
        Δ = α · tanh(T(δ))    # mass transfer rate
        g_new = g - Δ          # gas loses solute
        l_new = l + Δ          # liquid gains solute
"""

import torch
import torch.nn as nn


class CounterFlowPlate(nn.Module):
    """
    Single exchange plate in an absorption tower.

    Performs unidirectional transfer from gas → liquid, driven by
    the difference between the gas state and the equilibrium mapping
    of the liquid state.

    Args:
        d_gas: Dimension of the gas stream
        d_liquid: Dimension of the liquid stream
        alpha_init: Initial value for the transfer coefficient (default: 0.1)
            Analogous to K_Ya — the overall mass transfer coefficient.
    """

    def __init__(self, d_gas: int, d_liquid: int, alpha_init: float = 0.1):
        super().__init__()
        self.d_gas = d_gas
        self.d_liquid = d_liquid

        # Equilibrium function: E(l) maps liquid state → gas-equivalent
        # Analogous to Y* = f(X) — the equilibrium curve
        self.equilibrium = nn.Linear(d_liquid, d_gas)

        # Transfer function: T(δ) determines how much transfers given the driving force
        # Analogous to the mass transfer rate equation N_A = K · (Y - Y*)
        self.transfer = nn.Sequential(
            nn.Linear(d_gas, d_gas),
            nn.Tanh(),
        )

        # Transfer coefficient α > 0 (learnable)
        # Analogous to K_Ya — controls the "efficiency" of each plate
        self.log_alpha = nn.Parameter(torch.full((d_gas,), torch.tensor(alpha_init).log()))

    @property
    def alpha(self) -> torch.Tensor:
        """Transfer coefficient, guaranteed positive via exp(log_alpha)."""
        return self.log_alpha.exp()

    def forward(
        self, g: torch.Tensor, l: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single plate exchange step.

        Args:
            g: Gas stream state (batch, d_gas) — ascending, carries raw features
            l: Liquid stream state (batch, d_liquid) — descending, accumulates extracted info

        Returns:
            g_new: Updated gas state (batch, d_gas) — gas after losing "solute"
            l_new: Updated liquid state (batch, d_liquid) — liquid after gaining "solute"
            delta: Transfer amount (batch, d_gas) — what was transferred this plate
        """
        # Step 1: Equilibrium mapping — what the gas "should be" given the liquid
        e = torch.sigmoid(self.equilibrium(l))  # Y* = f(X)

        # Step 2: Driving force — difference between actual gas and equilibrium
        driving_force = g - e  # δ = Y - Y*

        # Step 3: Transfer amount — bounded by tanh, scaled by α
        delta = self.alpha * self.transfer(driving_force)  # Δ = α · tanh(T(δ))

        # Step 4: Conservation — what gas loses, liquid gains
        g_new = g - delta  # gas loses solute
        l_new = l + delta  # liquid gains solute (NOTE: requires d_gas == d_liquid for strict conservation)

        return g_new, l_new, delta

    def extra_repr(self) -> str:
        return (
            f"d_gas={self.d_gas}, d_liquid={self.d_liquid}, "
            f"alpha_mean={self.alpha.mean().item():.4f}"
        )
