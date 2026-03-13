"""
ChemE-inspired visualizations for CFNN analysis.

Publication-quality plots that map neural network internals to
chemical engineering diagrams:

    - McCabe-Thiele Neural Plot: gas vs liquid operating lines with
      equilibrium curve — the neural analog of the classic distillation diagram
    - Concentration Profiles: feature norms through the tower (Y vs Z)
    - Driving Force Profiles: delta = g - E(l) at each plate
    - Transfer Heatmaps: per-dimension transfer patterns
    - Alpha/Beta Evolution: transfer coefficient landscape
    - Column Schematic: matplotlib tower diagram
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional

from src.diagnostics import (
    operating_line_data, damkohler_number,
    number_of_transfer_units, alpha_statistics,
)


# -- Publication style defaults -----------------------------------------------
STYLE = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

GAS_COLOR = '#E74C3C'       # red -- vapor/gas
LIQUID_COLOR = '#3498DB'    # blue -- liquid
RECT_COLOR = '#E67E22'      # orange -- rectifying
STRIP_COLOR = '#2980B9'     # dark blue -- stripping
EQUIL_COLOR = '#2ECC71'     # green -- equilibrium curve
FEED_COLOR = '#9B59B6'      # purple -- feed plate
TRANSFER_COLOR = '#27AE60'  # green -- transfer amounts


def _apply_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(STYLE)


# =============================================================================
#  McCabe-Thiele Neural Plot
# =============================================================================

def mccabe_thiele_plot(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "Neural McCabe-Thiele Diagram",
    show_steps: bool = True,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a neural McCabe-Thiele diagram.

    Maps ||liquid|| (x-axis) vs ||gas|| (y-axis) through the tower,
    analogous to plotting X vs Y in a real distillation column.

    The stepping pattern between operating line and equilibrium curve
    shows how many "ideal stages" the network effectively uses.

    Args:
        model: CFNN-A or CFNN-D model (must have forward_with_intermediates)
        x: Input batch (batch, d_in)
        context: Optional context
        title: Plot title
        show_steps: If True, draw step construction lines
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    _apply_style()
    op = operating_line_data(model, x, context)

    fig, ax = plt.subplots(figsize=figsize)

    if op.get('type') == 'absorption':
        gas_n = op['gas_norms']
        liq_n = op['liquid_norms']
        min_len = min(len(gas_n), len(liq_n))

        # Operating line
        ax.plot(liq_n[:min_len], gas_n[:min_len], 'o-',
                color=GAS_COLOR, linewidth=2.5, markersize=10,
                label='Operating line', zorder=5)

        # Step construction
        if show_steps and min_len > 1:
            for i in range(min_len - 1):
                ax.plot([liq_n[i], liq_n[i+1]], [gas_n[i], gas_n[i]],
                        '--', color='gray', alpha=0.5, linewidth=1)
                ax.plot([liq_n[i+1], liq_n[i+1]], [gas_n[i], gas_n[i+1]],
                        '--', color='gray', alpha=0.5, linewidth=1)

        # Annotate plates
        for i in range(min_len):
            ax.annotate(f'P{i}', (liq_n[i], gas_n[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight='bold')

        # 45 degree line
        max_val = max(max(gas_n[:min_len]), max(liq_n[:min_len])) * 1.1
        ax.plot([0, max_val], [0, max_val], '--', color='black',
                alpha=0.3, linewidth=1, label='y = x (diagonal)')

    elif op.get('type') == 'distillation':
        gr = op['gas_rect_norms']
        lr = op['liquid_rect_norms']
        gs_list = op['gas_strip_norms']
        ls_list = op['liquid_strip_norms']
        min_r = min(len(gr), len(lr))
        min_s = min(len(gs_list), len(ls_list))

        # Rectifying operating line
        ax.plot(lr[:min_r], gr[:min_r], 'o-',
                color=RECT_COLOR, linewidth=2.5, markersize=10,
                label=f'Rectifying (R={op["reflux_ratio"]:.3f})', zorder=5)

        # Stripping operating line
        ax.plot(ls_list[:min_s], gs_list[:min_s], 's-',
                color=STRIP_COLOR, linewidth=2.5, markersize=10,
                label=f'Stripping (Rb={op["reboil_ratio"]:.3f})', zorder=5)

        # Step construction for rectifying
        if show_steps and min_r > 1:
            for i in range(min_r - 1):
                ax.plot([lr[i], lr[i+1]], [gr[i], gr[i]],
                        '--', color=RECT_COLOR, alpha=0.3, linewidth=1)
                ax.plot([lr[i+1], lr[i+1]], [gr[i], gr[i+1]],
                        '--', color=RECT_COLOR, alpha=0.3, linewidth=1)

        # Step construction for stripping
        if show_steps and min_s > 1:
            for i in range(min_s - 1):
                ax.plot([ls_list[i], ls_list[i+1]], [gs_list[i], gs_list[i]],
                        '--', color=STRIP_COLOR, alpha=0.3, linewidth=1)
                ax.plot([ls_list[i+1], ls_list[i+1]], [gs_list[i], gs_list[i+1]],
                        '--', color=STRIP_COLOR, alpha=0.3, linewidth=1)

        # Annotate
        for i in range(min_r):
            ax.annotate(f'R{i}', (lr[i], gr[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color=RECT_COLOR, fontweight='bold')
        for i in range(min_s):
            ax.annotate(f'S{i}', (ls_list[i], gs_list[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color=STRIP_COLOR, fontweight='bold')

        # Feed q annotation
        ax.annotate(f'q = {op["feed_q_mean"]:.3f}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=FEED_COLOR,
                              alpha=0.2))

        # 45 degree line
        all_vals = gr[:min_r] + gs_list[:min_s] + lr[:min_r] + ls_list[:min_s]
        max_val = max(all_vals) * 1.1 if all_vals else 1.0
        ax.plot([0, max_val], [0, max_val], '--', color='black',
                alpha=0.3, linewidth=1, label='y = x')

    ax.set_xlabel('||liquid|| (liquid composition analog)')
    ax.set_ylabel('||gas|| (gas composition analog)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


# =============================================================================
#  Concentration Profiles
# =============================================================================

def concentration_profile(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "Concentration Profile",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot gas and liquid feature norms through the tower.

    Analogous to plotting Y (gas composition) and X (liquid composition)
    as a function of tower height Z.

    For CFNN-D, shows rectifying and stripping sections with a feed plate marker.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)

    fig, ax = plt.subplots(figsize=figsize)

    if 'gas_states' in result:
        # CFNN-A
        gas_norms = [g.norm(dim=-1).mean().item() for g in result['gas_states']]
        liq_norms = [li.norm(dim=-1).mean().item() for li in result['liquid_states']]

        plates_g = list(range(len(gas_norms)))
        plates_l = list(range(len(liq_norms)))

        ax.plot(plates_g, gas_norms, 'o-', color=GAS_COLOR,
                linewidth=2.5, markersize=8, label='Gas ||g_n|| (ascending)')
        ax.plot(plates_l, liq_norms, 's-', color=LIQUID_COLOR,
                linewidth=2.5, markersize=8, label='Liquid ||l_n|| (descending)')

        # Shade the region between curves
        min_len = min(len(gas_norms), len(liq_norms))
        ax.fill_between(range(min_len), gas_norms[:min_len], liq_norms[:min_len],
                        alpha=0.1, color='gray', label='Driving force region')

        ax.set_xlabel('Plate Number (bottom to top)')

    elif 'gas_rect' in result:
        # CFNN-D
        gr = [g.norm(dim=-1).mean().item() for g in result['gas_rect']]
        gs_list = [g.norm(dim=-1).mean().item() for g in result['gas_strip']]
        lr = [li.norm(dim=-1).mean().item() for li in result['liquid_rect']]
        ls_list = [li.norm(dim=-1).mean().item() for li in result['liquid_strip']]

        n_s = len(gs_list)
        n_r = len(gr)

        # Combined plate indices: stripping (0..n_s-1), feed, rectifying (n_s+1..n_s+n_r)
        strip_plates = list(range(n_s))
        rect_plates = list(range(n_s + 1, n_s + 1 + n_r))
        feed_plate = n_s

        # Gas through full column
        ax.plot(strip_plates, gs_list, 'o-', color=GAS_COLOR, linewidth=2.5,
                markersize=8, label='Gas (stripping)')
        ax.plot(rect_plates, gr, 'o-', color=RECT_COLOR, linewidth=2.5,
                markersize=8, label='Gas (rectifying)')

        # Liquid through full column
        ax.plot(strip_plates, ls_list, 's-', color=LIQUID_COLOR, linewidth=2.5,
                markersize=8, label='Liquid (stripping)')
        ax.plot(rect_plates, lr, 's-', color='#5DADE2', linewidth=2.5,
                markersize=8, label='Liquid (rectifying)')

        # Feed plate marker
        ax.axvline(x=feed_plate, color=FEED_COLOR, linewidth=2,
                   linestyle='--', label=f'Feed (q={result["feed_q"].mean().item():.3f})')

        ax.set_xlabel('Plate Number (bottom to feed to top)')

    ax.set_ylabel('Feature Norm')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


# =============================================================================
#  Driving Force & Transfer Profiles
# =============================================================================

def driving_force_profile(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "Driving Force & Transfer Profile",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot driving force and transfer amount at each plate.

    Left panel: driving force ||delta|| = ||g - E(l)|| per plate
    Right panel: transfer amount ||Delta|| per plate

    For CFNN-D, colors distinguish rectifying vs stripping sections.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if 'driving_forces' in result:
        # CFNN-A
        df_norms = [d.norm(dim=-1).mean().item() for d in result['driving_forces']]
        delta_norms = [d.norm(dim=-1).mean().item() for d in result['deltas']]

        ax1.bar(range(len(df_norms)), df_norms, color=RECT_COLOR, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax1.set_xlabel('Plate Number')
        ax1.set_ylabel('||driving force||')
        ax1.set_title('Driving Force (Y - Y*)')

        ax2.bar(range(len(delta_norms)), delta_norms, color=TRANSFER_COLOR, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Plate Number')
        ax2.set_ylabel('||transfer||')
        ax2.set_title('Transfer Amount (Delta)')

    elif 'deltas_rect' in result:
        # CFNN-D
        dr = [d.norm(dim=-1).mean().item() for d in result['deltas_rect']]
        ds_list = [d.norm(dim=-1).mean().item() for d in result['deltas_strip']]

        all_deltas = dr + ds_list
        colors = [RECT_COLOR] * len(dr) + [STRIP_COLOR] * len(ds_list)

        ax1.bar(range(len(all_deltas)), all_deltas, color=colors, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax1.axvline(x=len(dr) - 0.5, color=FEED_COLOR, linewidth=2,
                    linestyle='--', label='Feed plate')
        ax1.set_xlabel('Plate Number')
        ax1.set_ylabel('||net transfer||')
        ax1.set_title('Net Transfer (coral=rect, blue=strip)')
        ax1.legend()

        # Show rectifying vs stripping transfer magnitudes
        if dr and ds_list:
            ax2.bar(['Rectifying\n(avg)', 'Stripping\n(avg)'],
                    [np.mean(dr), np.mean(ds_list)],
                    color=[RECT_COLOR, STRIP_COLOR], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
            ax2.set_ylabel('Mean ||transfer||')
            ax2.set_title('Section Comparison')

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(title, fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


# =============================================================================
#  Transfer Heatmap (per-dimension)
# =============================================================================

def transfer_heatmap(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "Transfer Heatmap (per dimension)",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap showing which dimensions transfer the most at each plate.

    Rows = dimensions of the gas/liquid stream
    Columns = plates
    Color = mean absolute transfer amount

    This reveals which "features" are being exchanged at each stage
    of the column.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)

    if 'deltas' in result:
        deltas = result['deltas']
    elif 'deltas_rect' in result:
        deltas = result['deltas_rect'] + result['deltas_strip']
    else:
        raise ValueError("Model must return deltas")

    # Build heatmap matrix: (d_gas, n_plates)
    heatmap = np.array([d.abs().mean(dim=0).numpy() for d in deltas]).T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Plate Number')
    ax.set_ylabel('Feature Dimension')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Mean |transfer|')

    # Mark feed plate for CFNN-D
    if 'deltas_rect' in result:
        feed_idx = len(result['deltas_rect'])
        ax.axvline(x=feed_idx - 0.5, color=FEED_COLOR, linewidth=2,
                   linestyle='--', label='Feed plate')
        ax.legend()

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


# =============================================================================
#  Diagnostic Dashboard
# =============================================================================

def diagnostic_dashboard(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    model_name: str = "CFNN",
    figsize: tuple = (18, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Complete diagnostic dashboard combining all key visualizations.

    6-panel layout:
        [McCabe-Thiele] [Concentration Profile]
        [Driving Force ] [Transfer Heatmap    ]
        [Damkohler     ] [Alpha Statistics    ]
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)
    da = damkohler_number(model, x, context)
    ntu = number_of_transfer_units(model, x, context)
    a_stats = alpha_statistics(model)

    fig = plt.figure(figsize=figsize)
    grid = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # -- Panel 1: McCabe-Thiele --
    ax1 = fig.add_subplot(grid[0, 0])
    op = operating_line_data(model, x, context)
    if op.get('type') == 'absorption':
        gn = op['gas_norms']
        ln = op['liquid_norms']
        ml = min(len(gn), len(ln))
        ax1.plot(ln[:ml], gn[:ml], 'o-', color=GAS_COLOR, linewidth=2, markersize=7)
        mv = max(max(gn[:ml]), max(ln[:ml])) * 1.1
        ax1.plot([0, mv], [0, mv], '--', color='black', alpha=0.3)
        for i in range(ml):
            ax1.annotate(f'P{i}', (ln[i], gn[i]), fontsize=8,
                         textcoords="offset points", xytext=(5, 5))
    elif op.get('type') == 'distillation':
        gr, lr = op['gas_rect_norms'], op['liquid_rect_norms']
        gs_l, ls_l = op['gas_strip_norms'], op['liquid_strip_norms']
        mr, ms = min(len(gr), len(lr)), min(len(gs_l), len(ls_l))
        ax1.plot(lr[:mr], gr[:mr], 'o-', color=RECT_COLOR, linewidth=2, markersize=7, label='Rect')
        ax1.plot(ls_l[:ms], gs_l[:ms], 's-', color=STRIP_COLOR, linewidth=2, markersize=7, label='Strip')
        av = gr[:mr] + gs_l[:ms] + lr[:mr] + ls_l[:ms]
        mv = max(av) * 1.1 if av else 1
        ax1.plot([0, mv], [0, mv], '--', color='black', alpha=0.3)
        ax1.legend(fontsize=8)
    ax1.set_xlabel('||liquid||')
    ax1.set_ylabel('||gas||')
    ax1.set_title('McCabe-Thiele')
    ax1.grid(True, alpha=0.2)

    # -- Panel 2: Concentration Profile --
    ax2 = fig.add_subplot(grid[0, 1])
    if 'gas_states' in result:
        gn = [g.norm(dim=-1).mean().item() for g in result['gas_states']]
        ln = [li.norm(dim=-1).mean().item() for li in result['liquid_states']]
        ax2.plot(range(len(gn)), gn, 'o-', color=GAS_COLOR, linewidth=2, label='Gas')
        ax2.plot(range(len(ln)), ln, 's-', color=LIQUID_COLOR, linewidth=2, label='Liquid')
    elif 'gas_rect' in result:
        gr = [g.norm(dim=-1).mean().item() for g in result['gas_rect']]
        gs_l = [g.norm(dim=-1).mean().item() for g in result['gas_strip']]
        lr = [li.norm(dim=-1).mean().item() for li in result['liquid_rect']]
        ls_l = [li.norm(dim=-1).mean().item() for li in result['liquid_strip']]
        ns = len(gs_l)
        ax2.plot(range(ns), gs_l, 'o-', color=GAS_COLOR, linewidth=2, label='Gas (strip)')
        ax2.plot(range(ns+1, ns+1+len(gr)), gr, 'o-', color=RECT_COLOR, linewidth=2, label='Gas (rect)')
        ax2.plot(range(ns), ls_l, 's-', color=LIQUID_COLOR, linewidth=2, label='Liq (strip)')
        ax2.plot(range(ns+1, ns+1+len(lr)), lr, 's-', color='#5DADE2', linewidth=2, label='Liq (rect)')
        ax2.axvline(x=ns, color=FEED_COLOR, linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Plate')
    ax2.set_ylabel('Feature Norm')
    ax2.set_title('Concentration Profile')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # -- Panel 3: Transfer Amount --
    ax3 = fig.add_subplot(grid[1, 0])
    if 'deltas' in result:
        dn = [d.norm(dim=-1).mean().item() for d in result['deltas']]
        ax3.bar(range(len(dn)), dn, color=TRANSFER_COLOR, alpha=0.8)
    elif 'deltas_rect' in result:
        dr = [d.norm(dim=-1).mean().item() for d in result['deltas_rect']]
        ds_l = [d.norm(dim=-1).mean().item() for d in result['deltas_strip']]
        ad = dr + ds_l
        cols = [RECT_COLOR]*len(dr) + [STRIP_COLOR]*len(ds_l)
        ax3.bar(range(len(ad)), ad, color=cols, alpha=0.8)
        ax3.axvline(x=len(dr)-0.5, color=FEED_COLOR, linewidth=1.5, linestyle='--')
    ax3.set_xlabel('Plate')
    ax3.set_ylabel('||transfer||')
    ax3.set_title('Transfer Amount')
    ax3.grid(True, alpha=0.2, axis='y')

    # -- Panel 4: Transfer Heatmap --
    ax4 = fig.add_subplot(grid[1, 1])
    if 'deltas' in result:
        deltas_list = result['deltas']
    else:
        deltas_list = result.get('deltas_rect', []) + result.get('deltas_strip', [])
    if deltas_list:
        hm = np.array([d.abs().mean(dim=0).numpy() for d in deltas_list]).T
        im = ax4.imshow(hm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax4, label='|transfer|')
    ax4.set_xlabel('Plate')
    ax4.set_ylabel('Dimension')
    ax4.set_title('Transfer Heatmap')

    # -- Panel 5: Damkohler Numbers --
    ax5 = fig.add_subplot(grid[2, 0])
    da_vals = da['da_per_plate']
    if 'da_rectifying' in da:
        cols = [RECT_COLOR]*len(da.get('da_rectifying', [])) + \
               [STRIP_COLOR]*len(da.get('da_stripping', []))
    else:
        cols = [TRANSFER_COLOR] * len(da_vals)
    ax5.bar(range(len(da_vals)), da_vals, color=cols, alpha=0.8)
    ax5.axhline(y=1.0, color='red', linewidth=1.5, linestyle='--', label='Da=1')
    ax5.set_xlabel('Plate')
    ax5.set_ylabel('Damkohler Number')
    ax5.set_title(f'Da per Plate (mean={da["da_mean"]:.3f}, NTU={ntu:.3f})')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2, axis='y')

    # -- Panel 6: Alpha/Beta Statistics --
    ax6 = fig.add_subplot(grid[2, 1])
    labels, means, stds = [], [], []
    for a in a_stats['alphas']:
        short_name = a['name'].split('.')[-1] if '.' in a['name'] else a['name']
        labels.append(f"a {short_name}")
        means.append(a['mean'])
        stds.append(a['std'])
    for b in a_stats['betas']:
        short_name = b['name'].split('.')[-1] if '.' in b['name'] else b['name']
        labels.append(f"b {short_name}")
        means.append(b['mean'])
        stds.append(b['std'])
    if labels:
        colors_ab = [GAS_COLOR]*len(a_stats['alphas']) + [LIQUID_COLOR]*len(a_stats['betas'])
        ax6.barh(range(len(labels)), means, xerr=stds,
                 color=colors_ab, alpha=0.8, capsize=3)
        ax6.set_yticks(range(len(labels)))
        ax6.set_yticklabels(labels, fontsize=9)
        ax6.set_xlabel('Coefficient Value')
        ax6.set_title('Transfer Coefficients')
        ax6.grid(True, alpha=0.2, axis='x')

    fig.suptitle(f'{model_name} -- Diagnostic Dashboard (params={model.count_parameters()})',
                 fontsize=16, y=1.01)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
#  Column Schematic
# =============================================================================

def column_schematic(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "CFNN Column Schematic",
    figsize: tuple = (6, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a schematic of the CFNN column showing plate arrangement,
    stream flows, and transfer amounts.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, 5)

    is_distillation = 'gas_rect' in result

    if not is_distillation:
        n_plates = len(result['deltas'])
        ax.set_ylim(-1, n_plates + 2)

        for i in range(n_plates):
            y = i + 0.5
            delta_norm = result['deltas'][i].norm(dim=-1).mean().item()
            width = min(4, max(0.5, delta_norm * 10))

            rect = mpatches.FancyBboxPatch((0.5, y - 0.15), width, 0.3,
                                            boxstyle="round,pad=0.05",
                                            facecolor=TRANSFER_COLOR, alpha=0.6,
                                            edgecolor='black')
            ax.add_patch(rect)
            ax.text(2.5, y, f'Plate {i}\nD={delta_norm:.3f}',
                    ha='center', va='center', fontsize=9)

        # Arrows
        ax.annotate('', xy=(0.2, n_plates + 0.5), xytext=(0.2, -0.5),
                    arrowprops=dict(arrowstyle='->', color=GAS_COLOR, lw=2.5))
        ax.text(-0.5, n_plates / 2, 'Gas (up)', color=GAS_COLOR,
                fontsize=12, fontweight='bold', rotation=90, va='center')

        ax.annotate('', xy=(4.5, -0.5), xytext=(4.5, n_plates + 0.5),
                    arrowprops=dict(arrowstyle='->', color=LIQUID_COLOR, lw=2.5))
        ax.text(4.8, n_plates / 2, 'Liquid (down)', color=LIQUID_COLOR,
                fontsize=12, fontweight='bold', rotation=90, va='center')

    else:
        n_r = len(result['deltas_rect'])
        n_s = len(result['deltas_strip'])
        n_total = n_r + n_s + 1
        ax.set_ylim(-1, n_total + 2)

        # Stripping section (bottom)
        for i in range(n_s):
            y = i + 0.5
            dn = result['deltas_strip'][n_s - 1 - i].norm(dim=-1).mean().item()
            width = min(4, max(0.5, dn * 10))
            rect = mpatches.FancyBboxPatch((0.5, y - 0.15), width, 0.3,
                                            boxstyle="round,pad=0.05",
                                            facecolor=STRIP_COLOR, alpha=0.5,
                                            edgecolor='black')
            ax.add_patch(rect)
            ax.text(2.5, y, f'Strip {n_s-1-i}\nD={dn:.3f}',
                    ha='center', va='center', fontsize=8)

        # Feed plate
        feed_y = n_s + 0.5
        rect = mpatches.FancyBboxPatch((0.2, feed_y - 0.2), 4.0, 0.4,
                                        boxstyle="round,pad=0.05",
                                        facecolor=FEED_COLOR, alpha=0.5,
                                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        q_val = result['feed_q'].mean().item()
        ax.text(2.5, feed_y, f'FEED (q={q_val:.3f})',
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Rectifying section (top)
        for i in range(n_r):
            y = n_s + 1.5 + i
            dn = result['deltas_rect'][i].norm(dim=-1).mean().item()
            width = min(4, max(0.5, dn * 10))
            rect = mpatches.FancyBboxPatch((0.5, y - 0.15), width, 0.3,
                                            boxstyle="round,pad=0.05",
                                            facecolor=RECT_COLOR, alpha=0.5,
                                            edgecolor='black')
            ax.add_patch(rect)
            ax.text(2.5, y, f'Rect {i}\nD={dn:.3f}',
                    ha='center', va='center', fontsize=8)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig
