"""
Unit tests for CounterFlow plates and network.

Tests conservation laws, dimension consistency, gradient flow,
and basic training behavior.
"""

import torch
import pytest

from src.plates import CounterFlowPlate
from src.network import CounterFlowNetwork


# ========== Plate Tests ==========


class TestCounterFlowPlate:
    """Tests for a single exchange plate."""

    def setup_method(self):
        self.d_gas = 16
        self.d_liquid = 16
        self.batch = 8
        self.plate = CounterFlowPlate(self.d_gas, self.d_liquid)

    def test_output_dimensions(self):
        """Plate output dimensions should match input dimensions."""
        g = torch.randn(self.batch, self.d_gas)
        l = torch.randn(self.batch, self.d_liquid)
        g_new, l_new, delta = self.plate(g, l)

        assert g_new.shape == (self.batch, self.d_gas)
        assert l_new.shape == (self.batch, self.d_liquid)
        assert delta.shape == (self.batch, self.d_gas)

    def test_conservation_law(self):
        """Conservation: what gas loses, liquid gains. g_new + l_new ≈ g + l."""
        g = torch.randn(self.batch, self.d_gas)
        l = torch.randn(self.batch, self.d_liquid)
        g_new, l_new, delta = self.plate(g, l)

        # g_new = g - delta, l_new = l + delta
        # So: g_new + l_new = g + l (when d_gas == d_liquid)
        total_before = g + l
        total_after = g_new + l_new
        assert torch.allclose(total_before, total_after, atol=1e-6), (
            f"Conservation violated! Max diff: {(total_before - total_after).abs().max():.2e}"
        )

    def test_conservation_via_delta(self):
        """Delta should satisfy: g_new = g - delta, l_new = l + delta."""
        g = torch.randn(self.batch, self.d_gas)
        l = torch.randn(self.batch, self.d_liquid)
        g_new, l_new, delta = self.plate(g, l)

        assert torch.allclose(g_new, g - delta, atol=1e-6)
        assert torch.allclose(l_new, l + delta, atol=1e-6)

    def test_alpha_positive(self):
        """Transfer coefficient α should always be positive."""
        assert (self.plate.alpha > 0).all()

    def test_alpha_init_value(self):
        """Alpha should init close to the specified value."""
        plate = CounterFlowPlate(16, 16, alpha_init=0.5)
        assert torch.allclose(plate.alpha, torch.tensor(0.5), atol=0.01)

    def test_gradients_flow(self):
        """Gradients should flow through the plate without NaN."""
        g = torch.randn(self.batch, self.d_gas, requires_grad=True)
        l = torch.randn(self.batch, self.d_liquid, requires_grad=True)
        g_new, l_new, delta = self.plate(g, l)

        loss = g_new.sum() + l_new.sum()
        loss.backward()

        assert g.grad is not None
        assert l.grad is not None
        assert not torch.isnan(g.grad).any()
        assert not torch.isnan(l.grad).any()

    def test_zero_liquid_nonzero_transfer(self):
        """With zero liquid, driving force should still produce transfer."""
        g = torch.ones(self.batch, self.d_gas)
        l = torch.zeros(self.batch, self.d_liquid)
        _, _, delta = self.plate(g, l)
        # With nonzero gas and zero liquid, there should be some transfer
        assert delta.abs().sum() > 0


# ========== Network Tests ==========


class TestCounterFlowNetwork:
    """Tests for the full CFNN-A network."""

    def setup_method(self):
        self.d_in = 10
        self.d_gas = 16
        self.d_liquid = 16
        self.n_plates = 5
        self.d_out = 3
        self.batch = 8

    def _make_network(self, **kwargs):
        defaults = dict(
            d_in=self.d_in,
            d_gas=self.d_gas,
            d_liquid=self.d_liquid,
            n_plates=self.n_plates,
            d_out=self.d_out,
        )
        defaults.update(kwargs)
        return CounterFlowNetwork(**defaults)

    def test_output_shape(self):
        """Forward pass should produce correct output shape."""
        model = self._make_network()
        x = torch.randn(self.batch, self.d_in)
        out = model(x)
        assert out.shape == (self.batch, self.d_out)

    def test_output_shape_with_context(self):
        """Forward pass with context should produce correct output shape."""
        model = self._make_network()
        x = torch.randn(self.batch, self.d_in)
        ctx = torch.randn(self.batch, self.d_liquid)
        out = model(x, context=ctx)
        assert out.shape == (self.batch, self.d_out)

    def test_shared_plates_same_params(self):
        """With share_plates=True, all plates use the same parameters."""
        model = self._make_network(share_plates=True)
        # All get_plate calls should return the same object
        plates = [model.get_plate(i) for i in range(self.n_plates)]
        for p in plates[1:]:
            assert p is plates[0], "Shared plates should be the same object"

    def test_unique_plates_different_params(self):
        """With share_plates=False, plates have independent parameters."""
        model = self._make_network(share_plates=False)
        plates = [model.get_plate(i) for i in range(self.n_plates)]
        for i in range(1, len(plates)):
            assert plates[i] is not plates[0], "Unique plates should be different objects"

    def test_different_sweeps_different_output(self):
        """Different n_sweeps should produce different outputs."""
        torch.manual_seed(42)
        model1 = self._make_network(n_sweeps=1)
        torch.manual_seed(42)
        model2 = self._make_network(n_sweeps=3)

        # Copy weights from model1 to model2
        model2.load_state_dict(model1.state_dict())

        x = torch.randn(self.batch, self.d_in)
        out1 = model1(x)
        out2 = model2(x)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Different sweep counts should produce different outputs"
        )

    def test_gradients_flow_all_plates(self):
        """Gradients should reach all plates without NaN."""
        model = self._make_network(share_plates=False)
        x = torch.randn(self.batch, self.d_in)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for i, plate in enumerate(model.plates):
            for name, param in plate.named_parameters():
                assert param.grad is not None, f"Plate {i} param {name} has no gradient"
                assert not torch.isnan(param.grad).any(), (
                    f"Plate {i} param {name} has NaN gradient"
                )

    def test_forward_with_intermediates(self):
        """forward_with_intermediates should return all expected keys."""
        model = self._make_network()
        x = torch.randn(self.batch, self.d_in)
        result = model.forward_with_intermediates(x)

        assert "output" in result
        assert "gas_states" in result
        assert "liquid_states" in result
        assert "deltas" in result
        assert "driving_forces" in result
        assert result["output"].shape == (self.batch, self.d_out)
        assert len(result["deltas"]) == self.n_plates
        assert len(result["driving_forces"]) == self.n_plates

    def test_parameter_count_shared_vs_unique(self):
        """Unique plates should have more parameters than shared."""
        model_shared = self._make_network(share_plates=True)
        model_unique = self._make_network(share_plates=False)
        assert model_unique.count_parameters() > model_shared.count_parameters()

    def test_batch_independence(self):
        """Different batch elements should get different outputs for different inputs."""
        model = self._make_network()
        x = torch.randn(self.batch, self.d_in)
        out = model(x)
        # Check that not all rows are identical
        assert not torch.allclose(out[0], out[1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
