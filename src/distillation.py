"""
CFNN-D: Distillation-inspired CounterFlow Neural Network.

Extends the absorption model (CFNN-A) with concepts from distillation:
    - Feed plate: Input enters at a specific plate, splitting the tower into
      a rectifying section (above) and a stripping section (below)
    - Bidirectional transfer: Unlike absorption, distillation has
      vapor-liquid equilibrium with transfer in both directions
    - Reflux: A fraction of the top product is recycled back down,
      analogous to reflux in a distillation column
    - Reboil: A fraction of the bottom product is recycled back up,
      analogous to the reboiler in a distillation column

Chemical Engineering Reference:
    Distillation column sections:
        Rectifying (above feed): enriches the light component going up
        Stripping (below feed): strips the light component going down

    McCabe-Thiele:
        Operating line slopes differ in rectifying vs stripping sections
        Feed condition (q-line) determines where they intersect

    Reflux ratio R = L/D controls separation quality vs energy cost.
    Higher R → better separation but more computation.
"""

import torch
import torch.nn as nn
from src.plates import CounterFlowPlate


class DistillationPlate(CounterFlowPlate):
    """
    Enhanced exchange plate for distillation with bidirectional transfer.

    Unlike the absorption plate (gas→liquid only), the distillation plate
    allows transfer in both directions: the gas stream can gain components
    from the liquid (stripping/vaporization) AND the liquid can gain
    components from the gas (condensation/absorption).

    The net transfer direction is determined by the driving force and
    learned equilibrium, but the capacity to go both ways makes this
    more expressive.

    Args:
        d_gas: Dimension of the gas (vapor) stream
        d_liquid: Dimension of the liquid stream
        alpha_init: Initial forward transfer coefficient (gas→liquid)
        beta_init: Initial reverse transfer coefficient (liquid→gas)
    """

    def __init__(
        self,
        d_gas: int,
        d_liquid: int,
        alpha_init: float = 0.1,
        beta_init: float = 0.05,
    ):
        super().__init__(d_gas, d_liquid, alpha_init)

        # Reverse transfer: liquid → gas (stripping/vaporization)
        # In distillation, liquid can also evaporate back into vapor
        self.reverse_equilibrium = nn.Linear(d_gas, d_liquid)
        self.reverse_transfer = nn.Sequential(
            nn.Linear(d_liquid, d_liquid),
            nn.Tanh(),
        )
        self.log_beta = nn.Parameter(
            torch.full((d_liquid,), torch.tensor(beta_init).log())
        )

    @property
    def beta(self) -> torch.Tensor:
        """Reverse transfer coefficient (liquid→gas), guaranteed positive."""
        return self.log_beta.exp()

    def forward(
        self, g: torch.Tensor, l: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bidirectional exchange at a distillation plate.

        Transfer occurs in two channels:
        1. Forward (gas→liquid): condensation — heavy components drop into liquid
        2. Reverse (liquid→gas): vaporization — light components rise into gas

        Conservation is maintained for each channel separately:
            g_new = g - delta_forward + delta_reverse_proj
            l_new = l + delta_forward_proj - delta_reverse

        where _proj indicates projection when d_gas != d_liquid.

        Returns:
            g_new: Updated gas state
            l_new: Updated liquid state
            net_delta: Net transfer amount (for diagnostics)
        """
        # --- Forward channel: gas → liquid (condensation) ---
        e_fwd = torch.sigmoid(self.equilibrium(l))  # Y* = f(X)
        driving_fwd = g - e_fwd  # δ_fwd = Y - Y*
        delta_fwd = self.alpha * self.transfer(driving_fwd)  # Δ_fwd

        # --- Reverse channel: liquid → gas (vaporization) ---
        e_rev = torch.sigmoid(self.reverse_equilibrium(g))  # X* = g(Y)
        driving_rev = l - e_rev  # δ_rev = X - X*
        delta_rev = self.beta * self.reverse_transfer(driving_rev)  # Δ_rev

        # --- Apply transfers with conservation ---
        # Gas loses via forward, gains via reverse
        g_new = g - delta_fwd + delta_rev  # (when d_gas == d_liquid)
        # Liquid gains via forward, loses via reverse
        l_new = l + delta_fwd - delta_rev

        # Net delta for diagnostics
        net_delta = delta_fwd - delta_rev

        return g_new, l_new, net_delta

    def extra_repr(self) -> str:
        return (
            f"d_gas={self.d_gas}, d_liquid={self.d_liquid}, "
            f"alpha_mean={self.alpha.mean().item():.4f}, "
            f"beta_mean={self.beta.mean().item():.4f}"
        )


class DistillationNetwork(nn.Module):
    """
    Full distillation-inspired neural network (CFNN-D).

    The column has three zones:
        1. Rectifying section (plates above feed): enriches the ascending gas
        2. Feed plate: where the input enters the column
        3. Stripping section (plates below feed): strips the descending liquid

    Feed enters at feed_plate, splitting the column into rectifying (above)
    and stripping (below) sections. Optionally, reflux recycles a fraction
    of the top product back down, and reboil recycles bottom product up.

    Args:
        d_in: Input dimension
        d_gas: Gas/vapor stream dimension
        d_liquid: Liquid stream dimension
        n_plates_rect: Number of plates in rectifying section (above feed)
        n_plates_strip: Number of plates in stripping section (below feed)
        d_out: Output dimension
        n_sweeps: Number of convergence sweeps
        share_plates_per_section: If True, share weights within each section
        reflux_ratio: Fraction of top product recycled (0 = no reflux)
        reboil_ratio: Fraction of bottom product recycled (0 = no reboil)
        alpha_init: Forward transfer coefficient init
        beta_init: Reverse transfer coefficient init
    """

    def __init__(
        self,
        d_in: int,
        d_gas: int,
        d_liquid: int,
        n_plates_rect: int,
        n_plates_strip: int,
        d_out: int,
        n_sweeps: int = 2,
        share_plates_per_section: bool = True,
        reflux_ratio: float = 0.3,
        reboil_ratio: float = 0.2,
        alpha_init: float = 0.1,
        beta_init: float = 0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_gas = d_gas
        self.d_liquid = d_liquid
        self.n_plates_rect = n_plates_rect
        self.n_plates_strip = n_plates_strip
        self.n_plates_total = n_plates_rect + n_plates_strip
        self.d_out = d_out
        self.n_sweeps = n_sweeps
        self.share_plates_per_section = share_plates_per_section

        # Learnable reflux and reboil ratios, clamped to [0, 1] via sigmoid
        self._reflux_logit = nn.Parameter(
            torch.tensor(reflux_ratio).logit()
        )
        self._reboil_logit = nn.Parameter(
            torch.tensor(reboil_ratio).logit()
        )

        # Feed encoder: maps input to feed state that gets injected at feed plate
        self.feed_encoder = nn.Sequential(
            nn.Linear(d_in, d_gas),
            nn.ReLU(),
        )

        # Feed condition network: determines how the feed splits between
        # gas and liquid streams (analogous to q-line in McCabe-Thiele)
        # q=1 → all liquid (subcooled feed), q=0 → all vapor (superheated)
        self.feed_q = nn.Sequential(
            nn.Linear(d_gas, 1),
            nn.Sigmoid(),  # q ∈ (0, 1)
        )

        # Rectifying section plates (above feed — enriching)
        if share_plates_per_section:
            self._rect_plate = DistillationPlate(d_gas, d_liquid, alpha_init, beta_init)
        else:
            self.rect_plates = nn.ModuleList([
                DistillationPlate(d_gas, d_liquid, alpha_init, beta_init)
                for _ in range(n_plates_rect)
            ])

        # Stripping section plates (below feed — stripping)
        if share_plates_per_section:
            self._strip_plate = DistillationPlate(d_gas, d_liquid, alpha_init, beta_init)
        else:
            self.strip_plates = nn.ModuleList([
                DistillationPlate(d_gas, d_liquid, alpha_init, beta_init)
                for _ in range(n_plates_strip)
            ])

        # Output head: distillate (top product) + bottoms (bottom product)
        # In distillation, you get two products — we concatenate both
        self.output_head = nn.Sequential(
            nn.Linear(d_gas + d_liquid, d_gas),
            nn.ReLU(),
            nn.Linear(d_gas, d_out),
        )

    @property
    def reflux_ratio(self) -> torch.Tensor:
        """Reflux ratio R ∈ (0, 1), learnable."""
        return torch.sigmoid(self._reflux_logit)

    @property
    def reboil_ratio(self) -> torch.Tensor:
        """Reboil ratio ∈ (0, 1), learnable."""
        return torch.sigmoid(self._reboil_logit)

    def get_rect_plate(self, n: int) -> DistillationPlate:
        """Get rectifying section plate n (0-indexed)."""
        if self.share_plates_per_section:
            return self._rect_plate
        return self.rect_plates[n]

    def get_strip_plate(self, n: int) -> DistillationPlate:
        """Get stripping section plate n (0-indexed)."""
        if self.share_plates_per_section:
            return self._strip_plate
        return self.strip_plates[n]

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the distillation column.

        Flow:
        1. Encode input → feed state
        2. Split feed into gas and liquid fractions (q-line)
        3. Run alternating sweeps through rectifying + stripping sections
        4. Apply reflux (top recycle) and reboil (bottom recycle)
        5. Output from distillate + bottoms

        Args:
            x: Input tensor (batch, d_in)
            context: Optional initial liquid context (batch, d_liquid)

        Returns:
            output: Predictions (batch, d_out)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode feed
        feed = self.feed_encoder(x)  # (batch, d_gas)

        # Feed condition: determine gas/liquid split
        q = self.feed_q(feed)  # (batch, 1) — fraction that enters as liquid

        # Feed splits into vapor (1-q) fraction going up, liquid (q) fraction going down
        feed_gas = feed * (1.0 - q)  # vapor portion of feed
        feed_liquid = feed * q  # liquid portion of feed

        # Initialize streams
        # Gas starts from bottom of stripping section (reboiler output)
        g_bottom = torch.zeros(batch_size, self.d_gas, device=device)

        # Liquid starts from top of rectifying section (condenser output)
        if context is not None:
            l_top = context
        else:
            l_top = torch.zeros(batch_size, self.d_liquid, device=device)

        # Index convention:
        # Rectifying section: plates 0..n_rect-1 (top to bottom, plate 0 = top)
        # Stripping section: plates 0..n_strip-1 (top to bottom, plate 0 = just below feed)
        # Gas flows bottom→top, Liquid flows top→bottom

        # Arrays for rectifying section
        # g_rect[i] = gas leaving plate i (going up), i=0..n_rect-1
        # l_rect[i] = liquid leaving plate i (going down), i=0..n_rect-1
        n_r = self.n_plates_rect
        n_s = self.n_plates_strip
        g_rect = [None] * n_r
        l_rect = [None] * n_r
        g_strip = [None] * n_s
        l_strip = [None] * n_s

        for sweep in range(self.n_sweeps):
            # === STRIPPING SECTION: bottom → feed plate ===
            # Gas ascends through stripping section
            g_in = g_bottom
            for i in range(n_s - 1, -1, -1):
                # Plate i: gas comes from below (i+1 or reboiler)
                plate = self.get_strip_plate(i)
                l_in = l_strip[i] if l_strip[i] is not None else feed_liquid
                g_strip[i], l_strip[i], _ = plate(g_in, l_in)
                if i > 0:
                    g_in = g_strip[i]

            # Gas leaving stripping top + feed vapor → enters rectifying bottom
            g_at_feed = (g_strip[0] if g_strip[0] is not None else g_bottom) + feed_gas

            # === RECTIFYING SECTION: feed plate → top ===
            # Gas continues ascending through rectifying section
            g_in = g_at_feed
            for i in range(n_r - 1, -1, -1):
                plate = self.get_rect_plate(i)
                l_in = l_rect[i] if l_rect[i] is not None else l_top
                g_rect[i], l_rect[i], _ = plate(g_in, l_in)
                if i > 0:
                    g_in = g_rect[i]

            # --- Liquid descends: top → bottom ---
            # Liquid descends through rectifying section
            l_in = l_top
            for i in range(n_r):
                plate = self.get_rect_plate(i)
                g_in = g_rect[i] if g_rect[i] is not None else g_at_feed
                _, l_rect[i], _ = plate(g_in, l_in)
                l_in = l_rect[i]

            # Liquid leaving rectifying bottom + feed liquid → enters stripping top
            l_at_feed = l_rect[n_r - 1] + feed_liquid if n_r > 0 else feed_liquid

            # Liquid descends through stripping section
            l_in = l_at_feed
            for i in range(n_s):
                plate = self.get_strip_plate(i)
                g_in = g_strip[i] if g_strip[i] is not None else g_bottom
                _, l_strip[i], _ = plate(g_in, l_in)
                l_in = l_strip[i]

            # === REFLUX & REBOIL ===
            # Reflux: fraction of top gas recycled as liquid
            distillate_gas = g_rect[0] if g_rect[0] is not None else g_at_feed
            R = self.reflux_ratio
            l_top = R * distillate_gas  # reflux → next sweep's top liquid

            # Reboil: fraction of bottom liquid recycled as gas
            bottoms_liquid = l_strip[n_s - 1] if n_s > 0 and l_strip[n_s - 1] is not None else l_at_feed
            Rb = self.reboil_ratio
            g_bottom = Rb * bottoms_liquid  # reboil → next sweep's bottom gas

        # Output: combine distillate (top gas) and bottoms (bottom liquid)
        distillate = g_rect[0] if g_rect[0] is not None else g_at_feed
        bottoms = l_strip[n_s - 1] if n_s > 0 and l_strip[n_s - 1] is not None else l_at_feed

        output_features = torch.cat([distillate, bottoms], dim=-1)
        return self.output_head(output_features)

    def forward_with_intermediates(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> dict:
        """
        Forward pass returning intermediate states for visualization.

        Returns dict with:
            - output: final predictions
            - gas_rect: gas states in rectifying section
            - gas_strip: gas states in stripping section
            - liquid_rect: liquid states in rectifying section
            - liquid_strip: liquid states in stripping section
            - feed_q: feed condition values
            - reflux_ratio: current reflux ratio
            - reboil_ratio: current reboil ratio
            - deltas_rect: transfer amounts in rectifying section
            - deltas_strip: transfer amounts in stripping section
        """
        batch_size = x.shape[0]
        device = x.device

        feed = self.feed_encoder(x)
        q = self.feed_q(feed)
        feed_gas = feed * (1.0 - q)
        feed_liquid = feed * q

        g_bottom = torch.zeros(batch_size, self.d_gas, device=device)
        l_top = context if context is not None else torch.zeros(batch_size, self.d_liquid, device=device)

        n_r = self.n_plates_rect
        n_s = self.n_plates_strip
        g_rect = [None] * n_r
        l_rect = [None] * n_r
        g_strip = [None] * n_s
        l_strip = [None] * n_s
        deltas_rect = []
        deltas_strip = []

        for sweep in range(self.n_sweeps):
            # Stripping section (gas ascending)
            g_in = g_bottom
            for i in range(n_s - 1, -1, -1):
                plate = self.get_strip_plate(i)
                l_in = l_strip[i] if l_strip[i] is not None else feed_liquid
                g_strip[i], l_strip[i], _ = plate(g_in, l_in)
                if i > 0:
                    g_in = g_strip[i]

            g_at_feed = (g_strip[0] if g_strip[0] is not None else g_bottom) + feed_gas

            # Rectifying section (gas ascending)
            g_in = g_at_feed
            for i in range(n_r - 1, -1, -1):
                plate = self.get_rect_plate(i)
                l_in = l_rect[i] if l_rect[i] is not None else l_top
                g_rect[i], l_rect[i], _ = plate(g_in, l_in)
                if i > 0:
                    g_in = g_rect[i]

            # Liquid descending through rectifying
            l_in = l_top
            for i in range(n_r):
                plate = self.get_rect_plate(i)
                g_in = g_rect[i] if g_rect[i] is not None else g_at_feed
                _, l_rect[i], d = plate(g_in, l_in)
                l_in = l_rect[i]
                if sweep == self.n_sweeps - 1:
                    deltas_rect.append(d.detach())

            l_at_feed = l_rect[n_r - 1] + feed_liquid if n_r > 0 else feed_liquid

            # Liquid descending through stripping
            l_in = l_at_feed
            for i in range(n_s):
                plate = self.get_strip_plate(i)
                g_in = g_strip[i] if g_strip[i] is not None else g_bottom
                _, l_strip[i], d = plate(g_in, l_in)
                l_in = l_strip[i]
                if sweep == self.n_sweeps - 1:
                    deltas_strip.append(d.detach())

            # Reflux & Reboil
            distillate_gas = g_rect[0] if g_rect[0] is not None else g_at_feed
            R = self.reflux_ratio
            l_top = R * distillate_gas

            bottoms_liquid = l_strip[n_s - 1] if n_s > 0 and l_strip[n_s - 1] is not None else l_at_feed
            Rb = self.reboil_ratio
            g_bottom = Rb * bottoms_liquid

        distillate = g_rect[0] if g_rect[0] is not None else g_at_feed
        bottoms = l_strip[n_s - 1] if n_s > 0 and l_strip[n_s - 1] is not None else l_at_feed
        output_features = torch.cat([distillate, bottoms], dim=-1)
        output = self.output_head(output_features)

        return {
            "output": output,
            "gas_rect": [g.detach() for g in g_rect if g is not None],
            "gas_strip": [g.detach() for g in g_strip if g is not None],
            "liquid_rect": [li.detach() for li in l_rect if li is not None],
            "liquid_strip": [li.detach() for li in l_strip if li is not None],
            "feed_q": q.detach(),
            "reflux_ratio": self.reflux_ratio.item(),
            "reboil_ratio": self.reboil_ratio.item(),
            "deltas_rect": deltas_rect,
            "deltas_strip": deltas_strip,
        }

    def count_parameters(self) -> int:
        """Total learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_gas={self.d_gas}, d_liquid={self.d_liquid}, "
            f"n_rect={self.n_plates_rect}, n_strip={self.n_plates_strip}, "
            f"d_out={self.d_out}, n_sweeps={self.n_sweeps}, "
            f"share_per_section={self.share_plates_per_section}, "
            f"reflux={self.reflux_ratio.item():.3f}, "
            f"reboil={self.reboil_ratio.item():.3f}, "
            f"params={self.count_parameters()}"
        )
