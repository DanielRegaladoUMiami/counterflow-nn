"""Unit tests for CFNN-D: DistillationPlate and DistillationNetwork."""

import pytest
import torch
from src.distillation import DistillationPlate, DistillationNetwork


class TestDistillationPlate:
    """Tests for the bidirectional distillation plate."""

    @pytest.fixture
    def plate(self):
        return DistillationPlate(d_gas=16, d_liquid=16, alpha_init=0.1, beta_init=0.05)

    @pytest.fixture
    def batch(self):
        torch.manual_seed(42)
        g = torch.randn(8, 16)
        l = torch.randn(8, 16)
        return g, l

    def test_output_dimensions(self, plate, batch):
        """Output dimensions must match input dimensions."""
        g, l = batch
        g_new, l_new, delta = plate(g, l)
        assert g_new.shape == g.shape
        assert l_new.shape == l.shape
        assert delta.shape == g.shape

    def test_bidirectional_conservation(self, plate, batch):
        """Net conservation: (g_new + l_new) == (g + l) when d_gas == d_liquid."""
        g, l = batch
        g_new, l_new, _ = plate(g, l)
        # Sum before and after should be equal
        total_before = g + l
        total_after = g_new + l_new
        assert torch.allclose(total_before, total_after, atol=1e-5), \
            f"Conservation violated: max diff = {(total_before - total_after).abs().max().item()}"

    def test_alpha_positive(self, plate):
        """Forward transfer coefficient must be positive."""
        assert (plate.alpha > 0).all()

    def test_beta_positive(self, plate):
        """Reverse transfer coefficient must be positive."""
        assert (plate.beta > 0).all()

    def test_beta_init_value(self):
        """Beta should initialize to the specified value."""
        plate = DistillationPlate(16, 16, alpha_init=0.1, beta_init=0.05)
        assert torch.allclose(plate.beta, torch.tensor(0.05), atol=1e-4)

    def test_gradients_flow(self, plate, batch):
        """Gradients should flow through both transfer channels."""
        g, l = batch
        g.requires_grad_(True)
        l.requires_grad_(True)
        g_new, l_new, _ = plate(g, l)
        loss = g_new.sum() + l_new.sum()
        loss.backward()
        assert g.grad is not None and g.grad.abs().sum() > 0
        assert l.grad is not None and l.grad.abs().sum() > 0

    def test_has_reverse_transfer(self, plate):
        """Distillation plate must have reverse transfer components."""
        assert hasattr(plate, 'reverse_equilibrium')
        assert hasattr(plate, 'reverse_transfer')
        assert hasattr(plate, 'log_beta')

    def test_more_params_than_absorption(self):
        """Distillation plate should have more params (bidirectional)."""
        from src.plates import CounterFlowPlate
        abs_plate = CounterFlowPlate(16, 16)
        dist_plate = DistillationPlate(16, 16)
        abs_params = sum(p.numel() for p in abs_plate.parameters())
        dist_params = sum(p.numel() for p in dist_plate.parameters())
        assert dist_params > abs_params


class TestDistillationNetwork:
    """Tests for the full CFNN-D network."""

    @pytest.fixture
    def net(self):
        return DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, n_sweeps=2,
        )

    @pytest.fixture
    def batch_input(self):
        torch.manual_seed(42)
        return torch.randn(8, 4)

    def test_output_shape(self, net, batch_input):
        """Output shape must be (batch, d_out)."""
        out = net(batch_input)
        assert out.shape == (8, 3)

    def test_output_shape_with_context(self, net, batch_input):
        """Output shape correct when context is provided."""
        context = torch.randn(8, 16)
        out = net(batch_input, context=context)
        assert out.shape == (8, 3)

    def test_reflux_ratio_bounded(self, net):
        """Reflux ratio must be in (0, 1)."""
        R = net.reflux_ratio
        assert 0 < R.item() < 1

    def test_reboil_ratio_bounded(self, net):
        """Reboil ratio must be in (0, 1)."""
        Rb = net.reboil_ratio
        assert 0 < Rb.item() < 1

    def test_reflux_ratio_learnable(self, net, batch_input):
        """Reflux ratio should receive gradients."""
        out = net(batch_input)
        out.sum().backward()
        assert net._reflux_logit.grad is not None
        assert net._reflux_logit.grad.abs() > 0

    def test_reboil_ratio_learnable(self, net, batch_input):
        """Reboil ratio should receive gradients."""
        out = net(batch_input)
        out.sum().backward()
        assert net._reboil_logit.grad is not None
        assert net._reboil_logit.grad.abs() > 0

    def test_feed_q_bounded(self, net, batch_input):
        """Feed condition q must be in (0, 1)."""
        feed = net.feed_encoder(batch_input)
        q = net.feed_q(feed)
        assert (q > 0).all() and (q < 1).all()

    def test_shared_plates_same_params_per_section(self):
        """Shared plates within a section should use same parameters."""
        net = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, share_plates_per_section=True,
        )
        # All rect plates return the same module
        assert net.get_rect_plate(0) is net.get_rect_plate(1)
        assert net.get_rect_plate(1) is net.get_rect_plate(2)
        # All strip plates return the same module
        assert net.get_strip_plate(0) is net.get_strip_plate(1)
        # But rect != strip (different sections)
        assert net.get_rect_plate(0) is not net.get_strip_plate(0)

    def test_unique_plates_different_params(self):
        """Unique plates should have different parameters."""
        net = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, share_plates_per_section=False,
        )
        assert net.get_rect_plate(0) is not net.get_rect_plate(1)
        assert net.get_strip_plate(0) is not net.get_strip_plate(1)

    def test_gradients_flow_all_params(self, net, batch_input):
        """Gradients should reach all learnable parameters."""
        out = net(batch_input)
        out.sum().backward()
        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                # At least some parameters should have non-zero gradients
                # (not all will necessarily be non-zero due to architecture)

    def test_forward_with_intermediates(self, net, batch_input):
        """forward_with_intermediates should return all expected keys."""
        result = net.forward_with_intermediates(batch_input)
        expected_keys = {
            'output', 'gas_rect', 'gas_strip', 'liquid_rect', 'liquid_strip',
            'feed_q', 'reflux_ratio', 'reboil_ratio', 'deltas_rect', 'deltas_strip'
        }
        assert set(result.keys()) == expected_keys

    def test_intermediates_output_matches_forward(self, net, batch_input):
        """Intermediates output should match regular forward."""
        torch.manual_seed(0)
        out1 = net(batch_input)
        torch.manual_seed(0)
        result = net.forward_with_intermediates(batch_input)
        out2 = result['output']
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_different_sweeps_different_output(self, batch_input):
        """Different number of sweeps should produce different outputs."""
        torch.manual_seed(42)
        net1 = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, n_sweeps=1,
        )
        torch.manual_seed(42)
        net2 = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, n_sweeps=3,
        )
        out1 = net1(batch_input)
        out2 = net2(batch_input)
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_batch_independence(self, net):
        """Each sample in a batch should be independent."""
        torch.manual_seed(42)
        x = torch.randn(4, 4)
        out_batch = net(x)
        out_single = torch.stack([net(x[i:i+1]) for i in range(4)]).squeeze(1)
        assert torch.allclose(out_batch, out_single, atol=1e-5)

    def test_parameter_count(self):
        """Shared CFNN-D should have fewer params than unique."""
        shared = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, share_plates_per_section=True,
        )
        unique = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=3, n_plates_strip=3,
            d_out=3, share_plates_per_section=False,
        )
        assert shared.count_parameters() < unique.count_parameters()

    def test_asymmetric_sections(self):
        """Network should work with different section sizes."""
        net = DistillationNetwork(
            d_in=4, d_gas=16, d_liquid=16,
            n_plates_rect=5, n_plates_strip=2,
            d_out=3,
        )
        x = torch.randn(4, 4)
        out = net(x)
        assert out.shape == (4, 3)
