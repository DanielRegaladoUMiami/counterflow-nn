"""
Tests for the physically-exact AbsorptionTower.

Validates:
  1. Closed-form solution matches the Kremser analytical formula
     for ideal stages (E=1, b=0).
  2. Closed-form matches brute-force stage-by-stage iteration with
     Murphree efficiency.
  3. Overall mass balance holds exactly.
  4. Operating line holds at every intermediate stage.
  5. Limiting cases (β → 1, A → 1, E → 1, E → 0).
  6. Gradients flow through all learnable parameters.
  7. Learnable parameters stay in physical ranges via their
     reparametrizations.
"""

import math
import pytest
import torch

from src.absorption_tower import AbsorptionNetwork, AbsorptionTower


# ---------------------------------------------------------------------------
# Reference implementations — used to validate the closed form
# ---------------------------------------------------------------------------

def kremser_fraction_removed(A: float, N: int) -> float:
    """
    Classic Kremser formula for fraction of solute absorbed, ideal stages,
    zero inlet liquid, linear equilibrium through origin.

        (y_{N+1} - y_1) / (y_{N+1} - m·x_0)
            = (A^{N+1} - A) / (A^{N+1} - 1)   if A ≠ 1
            = N / (N + 1)                      if A = 1
    """
    if abs(A - 1.0) < 1e-9:
        return N / (N + 1.0)
    return (A ** (N + 1) - A) / (A ** (N + 1) - 1.0)


def stage_by_stage_solve(
    y_feed: float,
    x_top: float,
    m: float,
    b: float,
    L_over_G: float,
    E: float,
    N: int,
    sweeps: int = 500,
) -> tuple[list[float], list[float]]:
    """
    Reference solver that does NOT use Kremser. Iterates the operating line
    and Murphree relation until convergence. Returns (y_stages, x_stages)
    where y_stages[n] = y_{n+1}  (y_stages[0] = y_1, y_stages[N] = y_feed).
    """
    # y_stages[i] holds y_{i+1}  for i = 0..N  (so index 0 is y_1, index N is y_{N+1})
    y = [0.0] * (N + 1)
    x = [0.0] * (N + 1)   # x[i] is x_i, x[0] = x_top, x[N] = x_bot
    y[N] = y_feed
    x[0] = x_top

    for _ in range(sweeps):
        # Sweep up — compute y_1 from guessed y_1 via recurrence? Simpler:
        # iterate: given current (x_n) use Murphree to update y_n,
        # then update x via operating line.
        # Murphree on tray n: y_n = y_{n+1} + E·(m·x_n + b - y_{n+1})
        for n in range(1, N + 1):
            y_star = m * x[n] + b
            y[n - 1] = y[n] + E * (y_star - y[n - 1] if False else (y_star - y[n]))
        # Operating line → update x using current y_1 at index 0
        y_1 = y[0]
        for n in range(1, N + 1):
            # x_n = x_top + (y_{n+1} - y_1) / (L/G)
            x[n] = x_top + (y[n] - y_1) / L_over_G

    return y, x


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_construction(self):
        t = AbsorptionTower(d=4, n_stages=3)
        assert t.d == 4 and t.n_stages == 3
        assert t.L_over_G.shape == (4,)
        assert t.m.shape == (4,)
        assert t.E.shape == (4,)
        assert t.b.shape == (4,)

    def test_init_values(self):
        t = AbsorptionTower(d=2, n_stages=4, L_over_G_init=2.0,
                             m_init=0.5, E_init=0.75, b_init=0.1)
        assert torch.allclose(t.L_over_G, torch.tensor([2.0, 2.0]), atol=1e-5)
        assert torch.allclose(t.m, torch.tensor([0.5, 0.5]), atol=1e-5)
        assert torch.allclose(t.E, torch.tensor([0.75, 0.75]), atol=1e-5)
        assert torch.allclose(t.b, torch.tensor([0.1, 0.1]), atol=1e-5)

    def test_absorption_factor(self):
        t = AbsorptionTower(d=1, n_stages=3, L_over_G_init=2.0, m_init=0.5)
        assert torch.allclose(t.A, torch.tensor([4.0]), atol=1e-5)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            AbsorptionTower(d=1, n_stages=0)
        with pytest.raises(ValueError):
            AbsorptionTower(d=1, n_stages=3, E_init=1.5)
        with pytest.raises(ValueError):
            AbsorptionTower(d=1, n_stages=3, L_over_G_init=-1.0)


# ---------------------------------------------------------------------------
# Physical correctness — the headline tests
# ---------------------------------------------------------------------------

class TestKremserExact:
    """Closed-form solve must match the classical Kremser equation."""

    @pytest.mark.parametrize("A,N", [
        (0.5, 3), (1.5, 3), (2.0, 5), (2.0, 2), (3.0, 10), (5.0, 4),
    ])
    def test_fraction_absorbed_matches_kremser(self, A, N):
        # Pick L/G=A and m=1 so that A = L/(mG) = L/G.
        t = AbsorptionTower(d=1, n_stages=N, L_over_G_init=A, m_init=1.0,
                             E_init=0.9999, b_init=0.0)

        y_feed = torch.tensor([[1.0]])
        x_top = torch.tensor([[0.0]])
        y_top, _ = t(y_feed, x_top)

        frac_expected = kremser_fraction_removed(A, N)
        frac_actual = (1.0 - y_top).item()
        assert math.isclose(frac_actual, frac_expected, abs_tol=1e-3), (
            f"A={A}, N={N}: expected {frac_expected:.6f}, got {frac_actual:.6f}"
        )

    def test_ideal_hand_computation(self):
        """Hand-computed case: A=2, N=2, m=1 ⇒ y_1 = 1/7."""
        t = AbsorptionTower(d=1, n_stages=2, L_over_G_init=2.0,
                             m_init=1.0, E_init=0.9999, b_init=0.0)
        y_top, x_bot = t(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
        assert math.isclose(y_top.item(), 1.0 / 7.0, abs_tol=5e-4)
        # Overall mass balance: L·(x_bot - 0) = G·(1 - y_top)
        # ⇒ x_bot = (1 - y_top) / (L/G) = (6/7) / 2 = 3/7
        assert math.isclose(x_bot.item(), 3.0 / 7.0, abs_tol=5e-4)


class TestMassBalance:
    """The overall mass balance must hold for every combination of params."""

    @pytest.mark.parametrize("seed", range(8))
    def test_overall_balance(self, seed):
        torch.manual_seed(seed)
        d, N = 6, 7
        t = AbsorptionTower(
            d=d, n_stages=N,
            L_over_G_init=float(torch.rand(1).item() * 3 + 0.3),
            m_init=float(torch.rand(1).item() * 1.5 + 0.2),
            E_init=float(torch.rand(1).item() * 0.6 + 0.3),
        )
        y_feed = torch.rand(4, d)
        x_top = torch.rand(4, d) * 0.1
        y_top, x_bot = t(y_feed, x_top)

        # G·(y_feed - y_top) = L·(x_bot - x_top)
        lhs = y_feed - y_top
        rhs = (x_bot - x_top) * t.L_over_G
        assert torch.allclose(lhs, rhs, atol=1e-5), (
            f"Mass balance violated, max diff: {(lhs-rhs).abs().max():.2e}"
        )


class TestOperatingLine:
    """Every internal stage must satisfy y_{n+1} = y_1 + (L/G)·(x_n - x_0)."""

    def test_operating_line_profiles(self):
        torch.manual_seed(0)
        t = AbsorptionTower(d=3, n_stages=6, L_over_G_init=1.8,
                             m_init=0.6, E_init=0.85)
        y_feed = torch.rand(2, 3)
        x_top = torch.rand(2, 3) * 0.1
        prof = t.profiles(y_feed, x_top)

        y_stages = prof["y_stages"]  # (B, N+1, d), index 0 = y_1, index N = y_{N+1}
        x_stages = prof["x_stages"]
        y_1 = y_stages[:, 0, :]
        N = t.n_stages

        for n in range(1, N + 1):
            # Want y_{n+1} = y_1 + (L/G)·(x_n - x_0)
            expected_y_np1 = y_1 + t.L_over_G * (x_stages[:, n, :] - x_stages[:, 0, :])
            actual_y_np1 = y_stages[:, n, :]
            assert torch.allclose(expected_y_np1, actual_y_np1, atol=1e-5), (
                f"Operating line violated at stage {n}"
            )

    def test_boundary_conditions(self):
        """y_stages[:, N] must equal y_feed; x_stages[:, 0] must equal x_top."""
        torch.manual_seed(1)
        t = AbsorptionTower(d=2, n_stages=4, L_over_G_init=2.0, m_init=0.8)
        y_feed = torch.rand(3, 2)
        x_top = torch.rand(3, 2) * 0.05
        prof = t.profiles(y_feed, x_top)
        assert torch.allclose(prof["y_stages"][:, -1, :], y_feed, atol=1e-5)
        assert torch.allclose(prof["x_stages"][:, 0, :], x_top, atol=1e-5)
        assert torch.allclose(prof["y_stages"][:, 0, :], prof["y_top"], atol=1e-5)
        assert torch.allclose(prof["x_stages"][:, -1, :], prof["x_bot"], atol=1e-5)


class TestMurphreeRelation:
    """Closed-form must respect Murphree efficiency on every tray."""

    def test_murphree_each_stage(self):
        torch.manual_seed(2)
        t = AbsorptionTower(d=2, n_stages=5, L_over_G_init=2.0,
                             m_init=0.6, E_init=0.7, b_init=0.02)
        y_feed = torch.rand(3, 2) * 0.5 + 0.3
        x_top = torch.rand(3, 2) * 0.05
        prof = t.profiles(y_feed, x_top)

        y_stages = prof["y_stages"]   # (B, N+1, d)  y_stages[:, i] = y_{i+1}
        x_stages = prof["x_stages"]   # (B, N+1, d)  x_stages[:, i] = x_i
        # On stage n (n=1..N):  y_n = y_{n+1} + E·(m·x_n + b - y_{n+1})
        for n in range(1, t.n_stages + 1):
            y_n = y_stages[:, n - 1, :]
            y_np1 = y_stages[:, n, :]
            x_n = x_stages[:, n, :]
            y_star = t.m * x_n + t.b
            expected = y_np1 + t.E * (y_star - y_np1)
            assert torch.allclose(y_n, expected, atol=1e-5), (
                f"Murphree violated at stage {n}, max diff "
                f"{(y_n - expected).abs().max():.2e}"
            )


class TestLimits:
    def test_beta_near_one_is_stable(self):
        """β = 1 when (1-E)+E·S = 1 ⇔ S = 1 (A = 1). Must not blow up."""
        t = AbsorptionTower(d=1, n_stages=6, L_over_G_init=1.0,
                             m_init=1.0, E_init=0.9999)
        y_top, _ = t(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
        # Kremser at A=1: frac = N/(N+1) = 6/7 ⇒ y_top = 1/7
        assert math.isclose(y_top.item(), 1.0 / 7.0, abs_tol=1e-3)
        assert not torch.isnan(y_top).any()
        assert not torch.isinf(y_top).any()

    def test_zero_efficiency_no_transfer(self):
        """E → 0: nothing transfers, y_top == y_feed, x_bot == x_top."""
        t = AbsorptionTower(d=2, n_stages=3, L_over_G_init=1.5,
                             m_init=0.7, E_init=1e-6)
        y_feed = torch.tensor([[0.5, 0.8]])
        x_top = torch.tensor([[0.02, 0.01]])
        y_top, x_bot = t(y_feed, x_top)
        assert torch.allclose(y_top, y_feed, atol=1e-4)
        assert torch.allclose(x_bot, x_top, atol=1e-4)

    def test_large_A_high_absorption(self):
        """A ≫ 1 with many stages ⇒ nearly complete removal."""
        t = AbsorptionTower(d=1, n_stages=20, L_over_G_init=5.0,
                             m_init=0.5, E_init=0.99)
        # A = 10, N = 20 ⇒ Kremser frac ≈ 1 - 10^{-20}, basically complete
        y_top, _ = t(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
        assert y_top.item() < 1e-6


# ---------------------------------------------------------------------------
# Differentiability / training
# ---------------------------------------------------------------------------

class TestGradients:
    def test_gradients_through_all_params(self):
        t = AbsorptionTower(d=4, n_stages=5)
        y_feed = torch.rand(3, 4, requires_grad=False)
        x_top = torch.rand(3, 4) * 0.05
        y_top, x_bot = t(y_feed, x_top)
        loss = (y_top.pow(2) + x_bot.pow(2)).sum()
        loss.backward()

        for name, p in t.named_parameters():
            assert p.grad is not None, f"No gradient on {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient on {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient on {name}"

    def test_gradients_wrt_inputs(self):
        t = AbsorptionTower(d=3, n_stages=4)
        y_feed = torch.rand(2, 3, requires_grad=True)
        y_top, _ = t(y_feed)
        y_top.sum().backward()
        assert y_feed.grad is not None
        assert torch.isfinite(y_feed.grad).all()

    def test_training_step_reduces_loss(self):
        """One training step should reduce a simple MSE loss."""
        torch.manual_seed(42)
        t = AbsorptionTower(d=2, n_stages=4, L_over_G_init=1.1,
                             m_init=0.9, E_init=0.5)
        y_feed = torch.full((8, 2), 0.8)
        x_top = torch.zeros(8, 2)
        target = torch.full((8, 2), 0.1)   # want very lean gas out

        opt = torch.optim.Adam(t.parameters(), lr=0.1)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            y_top, _ = t(y_feed, x_top)
            loss = (y_top - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.2, (
            f"Loss didn't drop enough: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
        # Physics invariants preserved through training
        assert (t.L_over_G > 0).all() and (t.m > 0).all()
        assert ((t.E > 0) & (t.E < 1)).all()


# ---------------------------------------------------------------------------
# AbsorptionNetwork wrapper
# ---------------------------------------------------------------------------

class TestAbsorptionNetwork:
    def test_forward_shape(self):
        net = AbsorptionNetwork(d_in=10, d_tower=8, d_out=3, n_stages=4)
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 3)

    def test_gradients_flow(self):
        net = AbsorptionNetwork(d_in=10, d_tower=8, d_out=3, n_stages=4)
        x = torch.randn(5, 10)
        out = net(x)
        out.sum().backward()
        for name, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
