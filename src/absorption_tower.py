"""
Physically-exact countercurrent gas absorption tower as a PyTorch module.

This module implements the textbook Kremser analytical solution for a
countercurrent multistage absorber with:

  - Linear equilibrium (Henry's law):  y* = m·x + b
  - Murphree plate efficiency per stage: y_n = y_{n+1} + E·(y_n* - y_{n+1})
  - Countercurrent mass balance:  G·(y_{n+1} - y_1) = L·(x_n - x_0)

All physical parameters (m, L/G, E, b) are learnable per-channel tensors,
so a batch of feature channels behaves as a batch of independent towers —
each one solved exactly and differentiably.

Reference derivation
--------------------
Define stripping factor S = m·G/L = 1/A, where A is the absorption factor.
Substituting the operating line into the Murphree relation yields the
linear recurrence:

    y_n = β·y_{n+1} + γ·y_1 + δ
    β   = (1-E) + E·S
    γ   = -E·S
    δ   = E·(m·x_0 + b)

Applying it N times from n=N down to n=1 with boundary y_{N+1} = y_feed
gives a single equation for y_1:

    y_1 = (β^N · y_feed + δ · S_N) / (1 - γ · S_N)

with S_N = Σ_{k=0}^{N-1} β^k = (β^N - 1)/(β - 1)  (= N when β = 1).

This collapses the whole tower into a closed form — no iteration needed,
and every operation is differentiable.

References
----------
Treybal, R.E. "Mass Transfer Operations", 3rd Ed., McGraw-Hill, 1980.
    Ch. 8 — Gas Absorption. Kremser equation, Eqs. 8.44–8.50.
Seader, Henley, Roper. "Separation Process Principles", 3rd Ed., Wiley, 2011.
    Ch. 6 — Absorption and Stripping.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class AbsorptionTower(nn.Module):
    """
    Countercurrent N-stage gas absorption tower with closed-form Kremser solve.

    Inputs
    ------
        y_feed : (B, d)  gas composition entering the bottom (stage N+1).
        x_top  : (B, d)  liquid composition entering the top (stage 0).
                         Defaults to zeros (clean solvent).

    Outputs
    -------
        y_top : (B, d)   gas leaving the top (stage 1)  = "lean gas".
        x_bot : (B, d)   liquid leaving the bottom (stage N) = "rich liquid".

    The d feature channels behave as d independent species, each with its
    own Henry's constant m, absorption ratio L/G, Murphree efficiency E,
    and equilibrium intercept b. All four are learnable.

    Parameter encoding
    ------------------
    To keep physics valid during training:
        L/G > 0       via log parametrisation
        m   > 0       via log parametrisation
        E ∈ (0, 1)    via sigmoid parametrisation
        b   free

    Args:
        d: feature dimension (number of parallel "species").
        n_stages: number of equilibrium trays N. Must be >= 1.
        L_over_G_init, m_init, E_init, b_init: initial values for the
            learnable parameters. Defaults picked so A = L/(mG) ≈ 2.14,
            a typical well-designed absorber.
    """

    def __init__(
        self,
        d: int,
        n_stages: int = 5,
        L_over_G_init: float = 1.5,
        m_init: float = 0.7,
        E_init: float = 0.8,
        b_init: float = 0.0,
    ):
        super().__init__()
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {n_stages}")
        if not (0.0 < E_init < 1.0):
            raise ValueError(f"E_init must be in (0, 1), got {E_init}")
        if L_over_G_init <= 0 or m_init <= 0:
            raise ValueError("L_over_G_init and m_init must be positive")

        self.d = d
        self.n_stages = n_stages

        self.log_L_over_G = nn.Parameter(
            torch.full((d,), math.log(L_over_G_init))
        )
        self.log_m = nn.Parameter(torch.full((d,), math.log(m_init)))
        self.logit_E = nn.Parameter(
            torch.full((d,), math.log(E_init / (1.0 - E_init)))
        )
        self.b = nn.Parameter(torch.full((d,), float(b_init)))

    # ------------------------------------------------------------------
    # Physical parameter views
    # ------------------------------------------------------------------
    @property
    def L_over_G(self) -> torch.Tensor:
        return self.log_L_over_G.exp()

    @property
    def m(self) -> torch.Tensor:
        return self.log_m.exp()

    @property
    def E(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_E)

    @property
    def A(self) -> torch.Tensor:
        """Absorption factor A = L/(m·G). A > 1 ⇒ absorption feasible."""
        return self.L_over_G / self.m

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------
    def _kremser_coefficients(self, x_top: torch.Tensor):
        """Compute β, γ, δ, S_N, β^N used by the closed-form solution."""
        m = self.m
        E = self.E
        S = m / self.L_over_G                # stripping factor = 1/A
        beta = (1.0 - E) + E * S             # (d,)
        gamma = -E * S                       # (d,)
        delta = E * (m * x_top + self.b)     # (B, d)  — broadcasts with x_top

        N = self.n_stages
        beta_N = beta.pow(N)                 # (d,)

        denom = beta - 1.0
        # Stable S_N: the series sum has a removable singularity at β = 1.
        # Where β ≈ 1 we fall back to S_N = N.
        near_one = denom.abs() < 1e-6
        safe_denom = torch.where(
            near_one, torch.ones_like(denom), denom
        )
        S_N = torch.where(
            near_one,
            torch.full_like(beta, float(N)),
            (beta_N - 1.0) / safe_denom,
        )
        return beta, gamma, delta, S_N, beta_N

    def forward(
        self, y_feed: torch.Tensor, x_top: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form solve for (y_top, x_bot).

        Args:
            y_feed: (B, d) gas feed composition at the bottom.
            x_top:  (B, d) liquid feed at the top (defaults to zeros).

        Returns:
            y_top: (B, d) gas composition leaving the top (= y_1).
            x_bot: (B, d) liquid composition leaving the bottom (= x_N).
        """
        if x_top is None:
            x_top = torch.zeros_like(y_feed)

        beta, gamma, delta, S_N, beta_N = self._kremser_coefficients(x_top)

        # Solve for y_1:  y_1 · (1 - γ·S_N) = β^N · y_feed + δ · S_N
        y_top = (beta_N * y_feed + delta * S_N) / (1.0 - gamma * S_N)

        # Overall mass balance ⇒ x_bot
        #   G·(y_feed - y_top) = L·(x_bot - x_top)
        x_bot = x_top + (y_feed - y_top) / self.L_over_G

        return y_top, x_bot

    # ------------------------------------------------------------------
    # Full stage-by-stage profiles (for diagnostics / visualisation)
    # ------------------------------------------------------------------
    def profiles(
        self, y_feed: torch.Tensor, x_top: torch.Tensor | None = None
    ) -> dict:
        """
        Return every stage composition y_1..y_{N+1} and x_0..x_N.

        Shapes:
            y_stages: (B, N+1, d)   y_stages[:, 0]  = y_top (= y_1)
                                    y_stages[:, N]  = y_feed (= y_{N+1})
            x_stages: (B, N+1, d)   x_stages[:, 0]  = x_top (= x_0)
                                    x_stages[:, N]  = x_bot (= x_N)
        """
        if x_top is None:
            x_top = torch.zeros_like(y_feed)

        beta, gamma, delta, _, _ = self._kremser_coefficients(x_top)
        y_top, x_bot = self.forward(y_feed, x_top)

        # March from y_1 upward using the inverted recurrence:
        #   y_{n+1} = (y_n - γ·y_1 - δ) / β
        y_stages = [y_top]
        y_prev = y_top
        for _ in range(self.n_stages):
            y_next = (y_prev - gamma * y_top - delta) / beta
            y_stages.append(y_next)
            y_prev = y_next
        y_stages_t = torch.stack(y_stages, dim=1)  # (B, N+1, d)

        # x_n from the operating line:
        #   x_n = x_0 + (y_{n+1} - y_1) / (L/G)
        # (degenerates to x_0 when n=0 since we use y_1)
        x_stages = [x_top]
        for n in range(1, self.n_stages + 1):
            y_n_plus_1 = y_stages_t[:, n, :]
            x_n = x_top + (y_n_plus_1 - y_top) / self.L_over_G
            x_stages.append(x_n)
        x_stages_t = torch.stack(x_stages, dim=1)  # (B, N+1, d)

        return {
            "y_stages": y_stages_t,
            "x_stages": x_stages_t,
            "y_top": y_top,
            "x_bot": x_bot,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def fraction_absorbed(
        self, y_feed: torch.Tensor, x_top: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Fraction of solute removed from the gas stream:
            φ = (y_feed - y_top) / (y_feed - m·x_top - b)

        This is the standard absorption efficiency used in design;
        the denominator is the maximum possible removal (gas in
        equilibrium with the inlet solvent).
        """
        if x_top is None:
            x_top = torch.zeros_like(y_feed)
        y_top, _ = self.forward(y_feed, x_top)
        y_star_top = self.m * x_top + self.b
        num = y_feed - y_top
        den = y_feed - y_star_top
        # Avoid division by zero in degenerate case y_feed == y*_top.
        return num / den.clamp_min(1e-12)

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, n_stages={self.n_stages}, "
            f"A_mean={self.A.mean().item():.3f}, "
            f"E_mean={self.E.mean().item():.3f}"
        )


class AbsorptionNetwork(nn.Module):
    """
    Thin ML wrapper around AbsorptionTower.

    Encodes the input into (y_feed, x_top) pairs, runs the tower, and
    decodes the concatenated [y_top || x_bot] into predictions.

    Purpose: let the exact-physics tower act as a drop-in layer for
    classification / regression tasks and compare against CFNN-A / MLP.
    """

    def __init__(
        self,
        d_in: int,
        d_tower: int,
        d_out: int,
        n_stages: int = 5,
        **tower_kwargs,
    ):
        super().__init__()
        self.gas_encoder = nn.Linear(d_in, d_tower)
        self.liquid_encoder = nn.Linear(d_in, d_tower)
        self.tower = AbsorptionTower(d_tower, n_stages=n_stages, **tower_kwargs)
        self.head = nn.Sequential(
            nn.Linear(2 * d_tower, d_tower),
            nn.ReLU(),
            nn.Linear(d_tower, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_feed = torch.sigmoid(self.gas_encoder(x))     # keep in [0,1] like mole fraction
        x_top = torch.sigmoid(self.liquid_encoder(x))
        y_top, x_bot = self.tower(y_feed, x_top)
        return self.head(torch.cat([y_top, x_bot], dim=-1))
