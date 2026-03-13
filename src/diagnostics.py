"""
CFNN Diagnostics — ChemE-inspired metrics for analyzing network behavior.

Provides tools to understand what's happening inside a CounterFlow network
using analogies from chemical engineering analysis:

    - Damköhler Number (Da): ratio of transfer rate to information flow rate
      Da >> 1: transfer dominates, plates may be redundant
      Da << 1: flow dominates, may need more plates

    - Plate Efficiency (Murphree): how close each plate gets to equilibrium
      η = 1 means perfect equilibrium plate, η < 1 means real plate

    - Number of Transfer Units (NTU): effective depth of the network

    - Operating Line Analysis: gas vs liquid trajectory through the column

    - Alpha/Beta Statistics: transfer coefficient analysis
"""

import torch
import numpy as np
from typing import Optional


@torch.no_grad()
def damkohler_number(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute the Damköhler number for each plate.

    Da_n = ||transfer_n|| / ||flow_n||

    where transfer_n = ||delta_n|| and flow_n = ||g_n - g_{n-1}|| + ||l_n - l_{n+1}||

    A high Da means the plate is doing a lot of work relative to the
    information passing through it. A low Da means the plate is barely
    active — might not need it.

    Args:
        model: A CounterFlowNetwork or DistillationNetwork
        x: Input tensor (batch, d_in)
        context: Optional context

    Returns:
        dict with 'da_per_plate' (list of floats), 'da_mean', 'da_std'
    """
    result = model.forward_with_intermediates(x, context)

    # For CFNN-A (absorption)
    if 'gas_states' in result and 'deltas' in result:
        gas = result['gas_states']
        deltas = result['deltas']

        da_values = []
        for i, delta in enumerate(deltas):
            transfer_rate = delta.norm(dim=-1).mean().item()
            # Flow rate: change in gas state across this plate
            if i + 1 < len(gas):
                flow_rate = (gas[i + 1] - gas[i]).norm(dim=-1).mean().item()
            else:
                flow_rate = gas[-1].norm(dim=-1).mean().item()
            flow_rate = max(flow_rate, 1e-8)
            da_values.append(transfer_rate / flow_rate)

        return {
            'da_per_plate': da_values,
            'da_mean': np.mean(da_values),
            'da_std': np.std(da_values),
        }

    # For CFNN-D (distillation)
    elif 'deltas_rect' in result and 'deltas_strip' in result:
        da_rect = []
        for delta in result['deltas_rect']:
            da_rect.append(delta.norm(dim=-1).mean().item())

        da_strip = []
        for delta in result['deltas_strip']:
            da_strip.append(delta.norm(dim=-1).mean().item())

        all_da = da_rect + da_strip
        return {
            'da_rectifying': da_rect,
            'da_stripping': da_strip,
            'da_per_plate': all_da,
            'da_mean': np.mean(all_da) if all_da else 0.0,
            'da_std': np.std(all_da) if all_da else 0.0,
        }

    raise ValueError("Model must have forward_with_intermediates returning deltas")


@torch.no_grad()
def murphree_efficiency(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute Murphree plate efficiency for each plate.

    η_n = ||actual change|| / ||maximum possible change||

    where:
        actual change = ||g_n - g_{n-1}|| (how much the gas actually changed)
        max possible change = ||g_{n-1} - E(l_n)|| (driving force = max change if perfect plate)

    η = 1 → plate achieves equilibrium (ideal stage)
    η < 1 → real plate, doesn't fully equilibrate
    η > 1 → super-efficient (overshoot — possible with learned transfer)

    Only supported for CFNN-A currently.
    """
    result = model.forward_with_intermediates(x, context)

    if 'gas_states' not in result:
        raise ValueError("Murphree efficiency requires gas_states (CFNN-A)")

    gas = result['gas_states']
    driving = result['driving_forces']

    efficiencies = []
    for i in range(len(driving)):
        if i + 1 < len(gas):
            actual = (gas[i + 1] - gas[i]).norm(dim=-1).mean().item()
        else:
            actual = 0.0
        max_possible = driving[i].norm(dim=-1).mean().item()
        max_possible = max(max_possible, 1e-8)
        efficiencies.append(actual / max_possible)

    return {
        'efficiency_per_plate': efficiencies,
        'efficiency_mean': np.mean(efficiencies),
        'efficiency_std': np.std(efficiencies),
    }


@torch.no_grad()
def number_of_transfer_units(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute the Number of Transfer Units (NTU).

    NTU = Σ ||delta_n|| / ||g_avg||

    This is the effective "depth" of the network in terms of how much
    total transfer occurs relative to the average stream magnitude.
    Higher NTU = more effective separation/processing.

    Analogous to NTU = ∫(dY / (Y - Y*)) in absorption.
    """
    result = model.forward_with_intermediates(x, context)

    if 'deltas' in result:
        deltas = result['deltas']
        gas = result['gas_states']
    elif 'deltas_rect' in result:
        deltas = result['deltas_rect'] + result['deltas_strip']
        gas = result.get('gas_rect', []) + result.get('gas_strip', [])
    else:
        raise ValueError("Model must return deltas")

    total_transfer = sum(d.norm(dim=-1).mean().item() for d in deltas)
    avg_gas = np.mean([g.norm(dim=-1).mean().item() for g in gas]) if gas else 1.0
    avg_gas = max(avg_gas, 1e-8)

    return total_transfer / avg_gas


@torch.no_grad()
def alpha_statistics(model) -> dict:
    """
    Analyze the transfer coefficients (alpha, beta) across all plates.

    Returns statistics about the learned transfer rates, which indicate
    how much each plate is "working". Plates with very small alpha
    might be redundant; plates with very large alpha might be bottlenecks.
    """
    alphas = []
    betas = []

    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and hasattr(module, 'log_alpha'):
            alphas.append({
                'name': name,
                'mean': module.alpha.mean().item(),
                'std': module.alpha.std().item() if module.alpha.numel() > 1 else 0.0,
                'min': module.alpha.min().item(),
                'max': module.alpha.max().item(),
            })
        if hasattr(module, 'beta') and hasattr(module, 'log_beta'):
            betas.append({
                'name': name,
                'mean': module.beta.mean().item(),
                'std': module.beta.std().item() if module.beta.numel() > 1 else 0.0,
                'min': module.beta.min().item(),
                'max': module.beta.max().item(),
            })

    return {
        'alphas': alphas,
        'betas': betas,
        'n_plates_with_alpha': len(alphas),
        'n_plates_with_beta': len(betas),
    }


@torch.no_grad()
def operating_line_data(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> dict:
    """
    Extract data for McCabe-Thiele style operating line plots.

    Returns gas and liquid norms at each plate, which can be plotted
    as gas_norm vs liquid_norm to create a neural McCabe-Thiele diagram.

    For CFNN-D, also returns separate rectifying and stripping operating lines.
    """
    result = model.forward_with_intermediates(x, context)

    data = {}

    if 'gas_states' in result:
        # CFNN-A
        gas_norms = [g.norm(dim=-1).mean().item() for g in result['gas_states']]
        liquid_norms = [li.norm(dim=-1).mean().item() for li in result['liquid_states']]
        data['gas_norms'] = gas_norms
        data['liquid_norms'] = liquid_norms
        data['type'] = 'absorption'

    if 'gas_rect' in result:
        # CFNN-D
        data['gas_rect_norms'] = [g.norm(dim=-1).mean().item() for g in result['gas_rect']]
        data['liquid_rect_norms'] = [li.norm(dim=-1).mean().item() for li in result['liquid_rect']]
        data['gas_strip_norms'] = [g.norm(dim=-1).mean().item() for g in result['gas_strip']]
        data['liquid_strip_norms'] = [li.norm(dim=-1).mean().item() for li in result['liquid_strip']]
        data['feed_q_mean'] = result['feed_q'].mean().item()
        data['reflux_ratio'] = result['reflux_ratio']
        data['reboil_ratio'] = result['reboil_ratio']
        data['type'] = 'distillation'

    return data


def print_diagnostics(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    model_name: str = "CFNN",
) -> dict:
    """
    Print a comprehensive diagnostic report for the model.

    Args:
        model: CFNN-A or CFNN-D model
        x: Sample input batch
        context: Optional context
        model_name: Name for the report header

    Returns:
        dict with all diagnostic data
    """
    print(f"\n{'='*60}")
    print(f"  CFNN Diagnostics Report: {model_name}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"{'='*60}")

    # Alpha/Beta statistics
    alpha_stats = alpha_statistics(model)
    print(f"\n--- Transfer Coefficients ---")
    for a in alpha_stats['alphas']:
        print(f"  {a['name']}: alpha_mean={a['mean']:.4f}, range=[{a['min']:.4f}, {a['max']:.4f}]")
    for b in alpha_stats['betas']:
        print(f"  {b['name']}: beta_mean={b['mean']:.4f}, range=[{b['min']:.4f}, {b['max']:.4f}]")

    # Damkohler numbers
    da = damkohler_number(model, x, context)
    print(f"\n--- Damkohler Numbers ---")
    print(f"  Da_mean = {da['da_mean']:.4f} +/- {da['da_std']:.4f}")
    for i, d in enumerate(da['da_per_plate']):
        label = "HIGH" if d > 1.0 else "LOW" if d < 0.1 else "OK"
        print(f"  Plate {i}: Da = {d:.4f} [{label}]")

    # NTU
    ntu = number_of_transfer_units(model, x, context)
    print(f"\n--- Network Transfer Units ---")
    print(f"  NTU = {ntu:.4f}")

    # Murphree efficiency (CFNN-A only)
    try:
        eff = murphree_efficiency(model, x, context)
        print(f"\n--- Murphree Plate Efficiency ---")
        print(f"  eta_mean = {eff['efficiency_mean']:.4f} +/- {eff['efficiency_std']:.4f}")
        for i, e in enumerate(eff['efficiency_per_plate']):
            print(f"  Plate {i}: eta = {e:.4f}")
    except (ValueError, KeyError):
        eff = None

    # Operating line data
    op = operating_line_data(model, x, context)
    if op.get('type') == 'distillation':
        print(f"\n--- Distillation Parameters ---")
        print(f"  Feed condition q = {op['feed_q_mean']:.4f}")
        print(f"  Reflux ratio R = {op['reflux_ratio']:.4f}")
        print(f"  Reboil ratio = {op['reboil_ratio']:.4f}")

    print(f"\n{'='*60}\n")

    return {
        'alpha_stats': alpha_stats,
        'damkohler': da,
        'ntu': ntu,
        'murphree': eff,
        'operating_line': op,
    }
