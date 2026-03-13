"""
Full CounterFlow Neural Network architectures.

CounterFlowNetwork (CFNN-A):
    Absorption-inspired architecture with unidirectional transfer.
    Uses Approach C (unrolled bidirectional sweeps) to resolve
    the backward dependency between gas and liquid streams.

    Approach C:
        1. Sweep 1 (top-down): Liquid descends using initial gas guess (g₀)
        2. Sweep 2 (bottom-up): Gas ascends using computed liquid values
        3. Optional additional sweeps for iterative refinement

    This is analogous to solving a real absorption tower by alternating
    tray-by-tray calculations until convergence.
"""

import torch
import torch.nn as nn

from src.plates import CounterFlowPlate


class CounterFlowNetwork(nn.Module):
    """
    Full countercurrent absorption network (CFNN-A).

    Two streams flow in opposite directions through N exchange plates:
        - Gas stream (ascending): carries raw features from input
        - Liquid stream (descending): accumulates extracted representation

    The output concatenates both streams: [g_N || l_1], giving the model
    access to both "cleaned" features and "extracted" representation.

    Args:
        d_in: Input dimension
        d_gas: Gas stream dimension (feature processing width)
        d_liquid: Liquid stream dimension (context/extraction width)
        n_plates: Number of exchange plates (analogous to NTU — network depth)
        d_out: Output dimension (number of classes or regression targets)
        n_sweeps: Number of convergence sweeps per forward pass (default: 2)
        share_plates: If True, all plates share weights (uniform packing).
            If False, each plate has unique parameters (sectioned column).
        alpha_init: Initial transfer coefficient for all plates.
    """

    def __init__(
        self,
        d_in: int,
        d_gas: int,
        d_liquid: int,
        n_plates: int,
        d_out: int,
        n_sweeps: int = 2,
        share_plates: bool = True,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_gas = d_gas
        self.d_liquid = d_liquid
        self.n_plates = n_plates
        self.d_out = d_out
        self.n_sweeps = n_sweeps
        self.share_plates = share_plates

        # Gas encoder: maps input to gas stream initial state
        # Analogous to the feed gas entering the bottom of the tower
        self.gas_encoder = nn.Sequential(
            nn.Linear(d_in, d_gas),
            nn.ReLU(),
        )

        # Exchange plates
        if share_plates:
            # Uniform packing: one plate module reused N times
            self._shared_plate = CounterFlowPlate(d_gas, d_liquid, alpha_init)
        else:
            # Sectioned column: unique parameters per plate
            self.plates = nn.ModuleList(
                [CounterFlowPlate(d_gas, d_liquid, alpha_init) for _ in range(n_plates)]
            )

        # Output head: combines cleaned gas + loaded liquid → prediction
        # Analogous to analyzing both the exit gas and exit liquid
        self.output_head = nn.Sequential(
            nn.Linear(d_gas + d_liquid, d_gas),
            nn.ReLU(),
            nn.Linear(d_gas, d_out),
        )

    def get_plate(self, n: int) -> CounterFlowPlate:
        """Get plate n (0-indexed). Handles shared vs unique plates."""
        if self.share_plates:
            return self._shared_plate
        return self.plates[n]

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass using Approach C (unrolled bidirectional sweeps).

        Args:
            x: Input tensor (batch, d_in)
            context: Optional initial liquid context (batch, d_liquid).
                If None, liquid starts as zeros (clean solvent entering tower top).

        Returns:
            output: Predictions (batch, d_out)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode input → initial gas state (feed gas entering tower bottom)
        g0 = self.gas_encoder(x)  # (batch, d_gas)

        # Initialize liquid stream: clean solvent from top
        if context is not None:
            l_init = context
        else:
            l_init = torch.zeros(batch_size, self.d_liquid, device=device)

        # Initialize gas and liquid arrays
        # g[n] = gas state after plate n (g[0] = encoded input)
        # l[n] = liquid state at plate n (l[N+1] = initial clean liquid from top)
        g = [g0] + [None] * self.n_plates
        l = [None] * (self.n_plates + 1) + [l_init]

        # Approach C: Alternating sweeps
        for sweep in range(self.n_sweeps):
            # --- Sweep down: liquid descends (top → bottom) ---
            # Uses current gas values (g[0] on first sweep, updated gas on later sweeps)
            for n in range(self.n_plates, 0, -1):
                plate = self.get_plate(n - 1)
                # Use gas from below this plate (or g0 if not yet computed)
                g_input = g[n - 1] if g[n - 1] is not None else g0
                _, l[n], _ = plate(g_input, l[n + 1])

            # --- Sweep up: gas ascends (bottom → top) ---
            # Uses freshly computed liquid values
            g[0] = g0  # reset gas input (feed gas doesn't change)
            for n in range(1, self.n_plates + 1):
                plate = self.get_plate(n - 1)
                g[n], _, _ = plate(g[n - 1], l[n])

        # If l[1] is still None (shouldn't happen, but safety), use zeros
        if l[1] is None:
            l[1] = torch.zeros(batch_size, self.d_liquid, device=device)

        # Concatenate cleaned gas (tower top exit) + loaded liquid (tower bottom exit)
        output_features = torch.cat([g[self.n_plates], l[1]], dim=-1)

        return self.output_head(output_features)

    def forward_with_intermediates(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> dict:
        """
        Forward pass that also returns intermediate states for visualization.

        Returns dict with:
            - output: final predictions
            - gas_states: list of g[0]...g[N] tensors
            - liquid_states: list of l[1]...l[N+1] tensors
            - deltas: list of transfer amounts per plate
            - driving_forces: list of driving forces per plate
        """
        batch_size = x.shape[0]
        device = x.device

        g0 = self.gas_encoder(x)

        if context is not None:
            l_init = context
        else:
            l_init = torch.zeros(batch_size, self.d_liquid, device=device)

        g = [g0] + [None] * self.n_plates
        l = [None] * (self.n_plates + 1) + [l_init]

        for sweep in range(self.n_sweeps):
            for n in range(self.n_plates, 0, -1):
                plate = self.get_plate(n - 1)
                g_input = g[n - 1] if g[n - 1] is not None else g0
                _, l[n], _ = plate(g_input, l[n + 1])

            g[0] = g0
            for n in range(1, self.n_plates + 1):
                plate = self.get_plate(n - 1)
                g[n], _, _ = plate(g[n - 1], l[n])

        # Final pass to collect deltas and driving forces
        deltas = []
        driving_forces = []
        for n in range(1, self.n_plates + 1):
            plate = self.get_plate(n - 1)
            e = torch.sigmoid(plate.equilibrium(l[n]))
            df = g[n - 1] - e
            delta = plate.alpha * plate.transfer(df)
            driving_forces.append(df.detach())
            deltas.append(delta.detach())

        if l[1] is None:
            l[1] = torch.zeros(batch_size, self.d_liquid, device=device)

        output_features = torch.cat([g[self.n_plates], l[1]], dim=-1)
        output = self.output_head(output_features)

        return {
            "output": output,
            "gas_states": [gs.detach() for gs in g if gs is not None],
            "liquid_states": [ls.detach() for ls in l if ls is not None],
            "deltas": deltas,
            "driving_forces": driving_forces,
        }

    def count_parameters(self) -> int:
        """Total learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_gas={self.d_gas}, d_liquid={self.d_liquid}, "
            f"n_plates={self.n_plates}, d_out={self.d_out}, "
            f"n_sweeps={self.n_sweeps}, share_plates={self.share_plates}, "
            f"params={self.count_parameters()}"
        )
