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
#  Column Schematic — P&ID Style
# =============================================================================

def _draw_vessel(ax, x_left, x_right, y_bot, y_top, **kw):
    """Draw a rounded-rectangle column vessel."""
    from matplotlib.path import Path
    import matplotlib.patches as mpatches
    r = 0.3
    verts = [
        (x_left + r, y_bot), (x_right - r, y_bot),                 # bottom
        (x_right, y_bot), (x_right, y_bot + r),                    # BR corner
        (x_right, y_top - r), (x_right, y_top), (x_right - r, y_top),  # TR
        (x_left + r, y_top), (x_left, y_top), (x_left, y_top - r),     # TL
        (x_left, y_bot + r), (x_left, y_bot), (x_left + r, y_bot),     # BL
    ]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    defaults = dict(facecolor='#F8F9FA', edgecolor='#2C3E50', linewidth=2.5)
    defaults.update(kw)
    patch = mpatches.PathPatch(path, **defaults)
    ax.add_patch(patch)


def column_schematic_pid(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "CFNN Column — P&ID Schematic",
    figsize: tuple = (10, 14),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    P&ID-style column schematic with internal tray detail, condenser/reboiler
    vessels, stream arrows with labels, and per-plate diagnostics.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)
    da_data = damkohler_number(model, x, context)
    da_plates = da_data['per_plate']

    is_distillation = 'gas_rect' in result
    fig, ax = plt.subplots(figsize=figsize)

    # Column geometry
    col_left, col_right = 2.0, 8.0
    col_cx = (col_left + col_right) / 2
    plate_h = 1.0
    tray_inset = 0.3

    if not is_distillation:
        # --- CFNN-A (Absorption) ---
        n_plates = len(result['deltas'])
        col_bot = 0.5
        col_top = col_bot + (n_plates + 1) * plate_h

        _draw_vessel(ax, col_left, col_right, col_bot, col_top)

        # Internal trays
        for i in range(n_plates):
            y = col_bot + (i + 0.5) * plate_h + 0.25
            dn = result['deltas'][i].norm(dim=-1).mean().item()
            da_val = da_plates[i] if i < len(da_plates) else 0

            # Tray line (width proportional to transfer)
            alpha_line = min(1.0, 0.3 + dn * 3)
            lw = max(1.5, min(5, dn * 20))
            ax.plot([col_left + tray_inset, col_right - tray_inset], [y, y],
                    color=TRANSFER_COLOR, linewidth=lw, alpha=alpha_line,
                    solid_capstyle='round')

            # Tray label (right side)
            ax.text(col_right + 0.3, y,
                    f'P{i}  Da={da_val:.3f}  ||\u0394||={dn:.3f}',
                    fontsize=9, va='center', fontfamily='monospace',
                    color='#2C3E50')

            # Small downcomers
            if i < n_plates - 1:
                dc_x = col_left + tray_inset + 0.15 if i % 2 == 0 else col_right - tray_inset - 0.15
                ax.annotate('', xy=(dc_x, y - 0.15), xytext=(dc_x, y + 0.15),
                            arrowprops=dict(arrowstyle='->', color=LIQUID_COLOR,
                                            lw=1.2, alpha=0.5))

        # Gas arrow (up, left side)
        gas_x = col_left - 0.8
        ax.annotate('', xy=(gas_x, col_top + 0.8), xytext=(gas_x, col_bot - 0.3),
                    arrowprops=dict(arrowstyle='->', color=GAS_COLOR, lw=3))
        ax.text(gas_x - 0.6, (col_bot + col_top) / 2, 'GAS',
                color=GAS_COLOR, fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')

        # Liquid arrow (down, right side)
        liq_x = col_right + 3.5
        ax.annotate('', xy=(liq_x, col_bot - 0.3), xytext=(liq_x, col_top + 0.8),
                    arrowprops=dict(arrowstyle='->', color=LIQUID_COLOR, lw=3))
        ax.text(liq_x + 0.6, (col_bot + col_top) / 2, 'LIQUID',
                color=LIQUID_COLOR, fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')

        # Gas inlet label
        gas_norms = [g.norm(dim=-1).mean().item() for g in result['gas_states']]
        liq_norms = [l_.norm(dim=-1).mean().item() for l_ in result['liquid_states']]
        ax.text(gas_x, col_bot - 0.6, f'||g\u2080||={gas_norms[0]:.2f}',
                fontsize=9, ha='center', color=GAS_COLOR, fontfamily='monospace')
        ax.text(gas_x, col_top + 1.1, f'||g_N||={gas_norms[-1]:.2f}',
                fontsize=9, ha='center', color=GAS_COLOR, fontfamily='monospace')
        ax.text(liq_x, col_top + 1.1, f'||l\u2080||={liq_norms[0]:.2f}',
                fontsize=9, ha='center', color=LIQUID_COLOR, fontfamily='monospace')
        ax.text(liq_x, col_bot - 0.6, f'||l_N||={liq_norms[-1]:.2f}',
                fontsize=9, ha='center', color=LIQUID_COLOR, fontfamily='monospace')

        ax.set_xlim(-0.5, col_right + 5)
        ax.set_ylim(col_bot - 1.5, col_top + 2)

    else:
        # --- CFNN-D (Distillation) ---
        n_r = len(result['deltas_rect'])
        n_s = len(result['deltas_strip'])
        n_total = n_r + n_s + 1  # +1 for feed
        col_bot = 1.5
        col_top = col_bot + (n_total + 1) * plate_h

        _draw_vessel(ax, col_left, col_right, col_bot, col_top)

        reflux = result['reflux_ratio'].mean().item()
        reboil = result['reboil_ratio'].mean().item()
        q_val = result['feed_q'].mean().item()

        # --- Condenser (top) ---
        cond_y = col_top + 0.8
        cond_rect = mpatches.FancyBboxPatch(
            (col_cx - 1.5, cond_y - 0.3), 3.0, 0.6,
            boxstyle="round,pad=0.1", facecolor='#D5F5E3',
            edgecolor='#27AE60', linewidth=2)
        ax.add_patch(cond_rect)
        ax.text(col_cx, cond_y, f'CONDENSER  R={reflux:.3f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='#27AE60')
        # Pipe from column to condenser
        ax.plot([col_cx, col_cx], [col_top, cond_y - 0.3],
                color='#2C3E50', linewidth=2)
        # Reflux return arrow
        ax.annotate('', xy=(col_left + 0.5, col_top),
                    xytext=(col_cx - 1.5, cond_y - 0.1),
                    arrowprops=dict(arrowstyle='->', color=LIQUID_COLOR,
                                    lw=2, connectionstyle='arc3,rad=0.3'))
        ax.text(col_left - 0.3, cond_y - 0.2, 'reflux',
                fontsize=8, color=LIQUID_COLOR, fontstyle='italic')
        # Distillate out
        ax.annotate('', xy=(col_cx + 2.8, cond_y),
                    xytext=(col_cx + 1.5, cond_y),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
        ax.text(col_cx + 3.0, cond_y, 'Distillate',
                fontsize=9, va='center', color='#27AE60', fontweight='bold')

        # --- Reboiler (bottom) ---
        reb_y = col_bot - 1.0
        reb_rect = mpatches.FancyBboxPatch(
            (col_cx - 1.5, reb_y - 0.3), 3.0, 0.6,
            boxstyle="round,pad=0.1", facecolor='#FADBD8',
            edgecolor=GAS_COLOR, linewidth=2)
        ax.add_patch(reb_rect)
        ax.text(col_cx, reb_y, f'REBOILER  Rb={reboil:.3f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=GAS_COLOR)
        # Pipe from column to reboiler
        ax.plot([col_cx, col_cx], [col_bot, reb_y + 0.3],
                color='#2C3E50', linewidth=2)
        # Vapor return arrow
        ax.annotate('', xy=(col_right - 0.5, col_bot),
                    xytext=(col_cx + 1.5, reb_y + 0.1),
                    arrowprops=dict(arrowstyle='->', color=GAS_COLOR,
                                    lw=2, connectionstyle='arc3,rad=-0.3'))
        ax.text(col_right + 0.3, reb_y + 0.2, 'boilup',
                fontsize=8, color=GAS_COLOR, fontstyle='italic')
        # Bottoms out
        ax.annotate('', xy=(col_cx + 2.8, reb_y),
                    xytext=(col_cx + 1.5, reb_y),
                    arrowprops=dict(arrowstyle='->', color=GAS_COLOR, lw=2))
        ax.text(col_cx + 3.0, reb_y, 'Bottoms',
                fontsize=9, va='center', color=GAS_COLOR, fontweight='bold')

        # --- Stripping plates (bottom of column) ---
        da_idx = 0
        for i in range(n_s):
            y = col_bot + (i + 0.5) * plate_h + 0.25
            dn = result['deltas_strip'][n_s - 1 - i].norm(dim=-1).mean().item()
            da_val = da_plates[da_idx] if da_idx < len(da_plates) else 0
            da_idx += 1

            lw = max(1.5, min(5, dn * 20))
            ax.plot([col_left + tray_inset, col_right - tray_inset], [y, y],
                    color=STRIP_COLOR, linewidth=lw, alpha=0.7,
                    solid_capstyle='round')

            ax.text(col_right + 0.3, y,
                    f'S{n_s-1-i}  Da={da_val:.3f}  ||\u0394||={dn:.3f}',
                    fontsize=9, va='center', fontfamily='monospace',
                    color=STRIP_COLOR)

        # --- Feed plate ---
        feed_y = col_bot + (n_s + 0.5) * plate_h + 0.25
        ax.plot([col_left + tray_inset, col_right - tray_inset],
                [feed_y, feed_y],
                color=FEED_COLOR, linewidth=4, solid_capstyle='round')
        # Feed arrow from left
        ax.annotate('', xy=(col_left, feed_y),
                    xytext=(col_left - 1.5, feed_y),
                    arrowprops=dict(arrowstyle='->', color=FEED_COLOR, lw=3))
        ax.text(col_left - 1.8, feed_y + 0.3,
                f'FEED\nq={q_val:.3f}', fontsize=10, fontweight='bold',
                ha='center', color=FEED_COLOR)

        # --- Rectifying plates (top of column) ---
        for i in range(n_r):
            y = col_bot + (n_s + 1 + i + 0.5) * plate_h + 0.25
            dn = result['deltas_rect'][i].norm(dim=-1).mean().item()
            da_val = da_plates[da_idx] if da_idx < len(da_plates) else 0
            da_idx += 1

            lw = max(1.5, min(5, dn * 20))
            ax.plot([col_left + tray_inset, col_right - tray_inset], [y, y],
                    color=RECT_COLOR, linewidth=lw, alpha=0.7,
                    solid_capstyle='round')

            ax.text(col_right + 0.3, y,
                    f'R{i}  Da={da_val:.3f}  ||\u0394||={dn:.3f}',
                    fontsize=9, va='center', fontfamily='monospace',
                    color=RECT_COLOR)

        # Section labels inside column
        strip_mid = col_bot + (n_s / 2) * plate_h + 0.25
        rect_mid = col_bot + (n_s + 1 + n_r / 2) * plate_h + 0.25
        ax.text(col_cx, strip_mid, 'STRIPPING', fontsize=11,
                ha='center', va='center', color=STRIP_COLOR,
                fontweight='bold', alpha=0.3, fontfamily='monospace')
        ax.text(col_cx, rect_mid, 'RECTIFYING', fontsize=11,
                ha='center', va='center', color=RECT_COLOR,
                fontweight='bold', alpha=0.3, fontfamily='monospace')

        ax.set_xlim(-1, col_right + 5.5)
        ax.set_ylim(reb_y - 1.5, cond_y + 1.5)

    # Model info box
    params = model.count_parameters()
    info = f'Parameters: {params:,}'
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='#BDC3C7'))

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


# =============================================================================
#  Column Schematic — Sankey / Flow Style
# =============================================================================

def column_schematic_sankey(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "CFNN Column — Flow Diagram",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Sankey-style flow diagram where stream widths are proportional to
    ||gas|| and ||liquid|| norms, and cross-flows show transfer amounts.
    """
    _apply_style()
    result = model.forward_with_intermediates(x, context)
    is_distillation = 'gas_rect' in result

    fig, ax = plt.subplots(figsize=figsize)

    if not is_distillation:
        gas_norms = [g.norm(dim=-1).mean().item() for g in result['gas_states']]
        liq_norms = [l_.norm(dim=-1).mean().item() for l_ in result['liquid_states']]
        delta_norms = [d.norm(dim=-1).mean().item() for d in result['deltas']]
        n_plates = len(delta_norms)

        # Layout
        plate_spacing = 2.0
        gas_x = 2.0    # gas stream center
        liq_x = 10.0   # liquid stream center
        norm_scale = 0.15  # width scale

        for i in range(n_plates + 1):
            y = i * plate_spacing

            # Gas stream segment
            if i < len(gas_norms):
                gw = max(0.15, gas_norms[i] * norm_scale)
                if i < n_plates:
                    gw_next = max(0.15, gas_norms[i + 1] * norm_scale) if i + 1 < len(gas_norms) else gw
                    y_next = (i + 1) * plate_spacing
                    # Tapered gas stream
                    verts_g = [
                        (gas_x - gw / 2, y), (gas_x + gw / 2, y),
                        (gas_x + gw_next / 2, y_next), (gas_x - gw_next / 2, y_next),
                        (gas_x - gw / 2, y),
                    ]
                    from matplotlib.patches import Polygon
                    poly = Polygon(verts_g, facecolor=GAS_COLOR, alpha=0.4,
                                   edgecolor=GAS_COLOR, linewidth=1)
                    ax.add_patch(poly)

            # Liquid stream segment (flows down, so draw top to bottom)
            if i < len(liq_norms):
                lw = max(0.15, liq_norms[i] * norm_scale)
                if i < n_plates:
                    lw_next = max(0.15, liq_norms[i + 1] * norm_scale) if i + 1 < len(liq_norms) else lw
                    y_next = (i + 1) * plate_spacing
                    verts_l = [
                        (liq_x - lw / 2, y), (liq_x + lw / 2, y),
                        (liq_x + lw_next / 2, y_next), (liq_x - lw_next / 2, y_next),
                        (liq_x - lw / 2, y),
                    ]
                    from matplotlib.patches import Polygon
                    poly = Polygon(verts_l, facecolor=LIQUID_COLOR, alpha=0.4,
                                   edgecolor=LIQUID_COLOR, linewidth=1)
                    ax.add_patch(poly)

            # Transfer cross-flow at each plate
            if i < n_plates:
                y_plate = y + plate_spacing * 0.5
                dn = delta_norms[i]
                tw = max(0.05, dn * 2)  # arrow width

                # Horizontal transfer arrow (gas -> liquid)
                ax.annotate(
                    '', xy=(liq_x - 0.5, y_plate),
                    xytext=(gas_x + 0.5, y_plate),
                    arrowprops=dict(arrowstyle='->', color=TRANSFER_COLOR,
                                    lw=max(1, tw * 3), alpha=0.7))

                # Plate label
                ax.text((gas_x + liq_x) / 2, y_plate + 0.3,
                        f'Plate {i}', fontsize=10, ha='center',
                        fontweight='bold', color='#2C3E50')
                ax.text((gas_x + liq_x) / 2, y_plate - 0.3,
                        f'||\u0394||={dn:.3f}', fontsize=9, ha='center',
                        fontfamily='monospace', color=TRANSFER_COLOR)

                # Plate background
                plate_rect = mpatches.FancyBboxPatch(
                    (gas_x + 0.8, y_plate - 0.4), liq_x - gas_x - 1.6, 0.8,
                    boxstyle="round,pad=0.1", facecolor='#F8F9FA',
                    edgecolor='#BDC3C7', linewidth=1, alpha=0.5, zorder=0)
                ax.add_patch(plate_rect)

        # Stream labels
        ax.text(gas_x, -0.8, f'GAS IN\n||g\u2080||={gas_norms[0]:.2f}',
                ha='center', fontsize=11, fontweight='bold', color=GAS_COLOR)
        top_y = n_plates * plate_spacing
        ax.text(gas_x, top_y + 0.8,
                f'GAS OUT\n||g_N||={gas_norms[-1]:.2f}',
                ha='center', fontsize=11, fontweight='bold', color=GAS_COLOR)

        ax.text(liq_x, top_y + 0.8,
                f'LIQUID IN\n||l\u2080||={liq_norms[0]:.2f}',
                ha='center', fontsize=11, fontweight='bold', color=LIQUID_COLOR)
        ax.text(liq_x, -0.8,
                f'LIQUID OUT\n||l_N||={liq_norms[-1]:.2f}',
                ha='center', fontsize=11, fontweight='bold', color=LIQUID_COLOR)

        # Direction arrows
        ax.annotate('', xy=(gas_x - 1, top_y), xytext=(gas_x - 1, 0),
                    arrowprops=dict(arrowstyle='->', color=GAS_COLOR,
                                    lw=2, alpha=0.3))
        ax.annotate('', xy=(liq_x + 1, 0), xytext=(liq_x + 1, top_y),
                    arrowprops=dict(arrowstyle='->', color=LIQUID_COLOR,
                                    lw=2, alpha=0.3))

        ax.set_xlim(-0.5, liq_x + 2.5)
        ax.set_ylim(-2, top_y + 2)

    else:
        # --- CFNN-D Sankey ---
        gas_r = [g.norm(dim=-1).mean().item() for g in result['gas_rect']]
        liq_r = [l_.norm(dim=-1).mean().item() for l_ in result['liquid_rect']]
        gas_s = [g.norm(dim=-1).mean().item() for g in result['gas_strip']]
        liq_s = [l_.norm(dim=-1).mean().item() for l_ in result['liquid_strip']]
        delta_r = [d.norm(dim=-1).mean().item() for d in result['deltas_rect']]
        delta_s = [d.norm(dim=-1).mean().item() for d in result['deltas_strip']]

        reflux = result['reflux_ratio'].mean().item()
        reboil = result['reboil_ratio'].mean().item()
        q_val = result['feed_q'].mean().item()

        plate_spacing = 2.0
        gas_x, liq_x = 2.0, 10.0
        norm_scale = 0.15
        cx = (gas_x + liq_x) / 2

        # Build combined plate list: strip (bottom) + feed + rect (top)
        all_plates = []
        for i in range(len(delta_s)):
            all_plates.append(('strip', i, delta_s[len(delta_s) - 1 - i]))
        all_plates.append(('feed', 0, 0))
        for i in range(len(delta_r)):
            all_plates.append(('rect', i, delta_r[i]))

        for idx, (ptype, pi, dn) in enumerate(all_plates):
            y = idx * plate_spacing
            y_next = (idx + 1) * plate_spacing

            if ptype == 'feed':
                # Feed plate
                feed_rect = mpatches.FancyBboxPatch(
                    (gas_x - 0.5, y + plate_spacing * 0.2),
                    liq_x - gas_x + 1, plate_spacing * 0.6,
                    boxstyle="round,pad=0.15", facecolor=FEED_COLOR,
                    edgecolor=FEED_COLOR, linewidth=2, alpha=0.3)
                ax.add_patch(feed_rect)
                ax.text(cx, y + plate_spacing * 0.5,
                        f'FEED PLATE  q={q_val:.3f}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color=FEED_COLOR)
                # Feed arrow
                ax.annotate('', xy=(gas_x - 0.5, y + plate_spacing * 0.5),
                            xytext=(gas_x - 2.5, y + plate_spacing * 0.5),
                            arrowprops=dict(arrowstyle='->', color=FEED_COLOR,
                                            lw=3))
                ax.text(gas_x - 3.0, y + plate_spacing * 0.5, 'FEED',
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color=FEED_COLOR)
            else:
                # Normal plate
                color = STRIP_COLOR if ptype == 'strip' else RECT_COLOR
                label = f'S{pi}' if ptype == 'strip' else f'R{pi}'
                tw = max(1, dn * 3)

                plate_rect = mpatches.FancyBboxPatch(
                    (gas_x + 0.8, y + plate_spacing * 0.25),
                    liq_x - gas_x - 1.6, plate_spacing * 0.5,
                    boxstyle="round,pad=0.1", facecolor=color,
                    edgecolor=color, linewidth=1, alpha=0.15, zorder=0)
                ax.add_patch(plate_rect)

                y_mid = y + plate_spacing * 0.5
                ax.annotate(
                    '', xy=(liq_x - 0.5, y_mid),
                    xytext=(gas_x + 0.5, y_mid),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=tw, alpha=0.6))
                ax.text(cx, y_mid + 0.35, label, fontsize=10,
                        ha='center', fontweight='bold', color=color)
                ax.text(cx, y_mid - 0.35,
                        f'||\u0394||={dn:.3f}', fontsize=9, ha='center',
                        fontfamily='monospace', color=color)

        # Top / bottom labels
        top_y = len(all_plates) * plate_spacing
        ax.text(cx, top_y + 0.8,
                f'Condenser (R={reflux:.3f})',
                ha='center', fontsize=12, fontweight='bold', color='#27AE60',
                bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.8,
                          edgecolor='#27AE60'))
        ax.text(cx, -1.2,
                f'Reboiler (Rb={reboil:.3f})',
                ha='center', fontsize=12, fontweight='bold', color=GAS_COLOR,
                bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8,
                          edgecolor=GAS_COLOR))

        # Stream labels
        ax.text(gas_x, -0.5, 'GAS', ha='center', fontsize=11,
                fontweight='bold', color=GAS_COLOR)
        ax.text(liq_x, top_y + 0.2, 'LIQUID', ha='center', fontsize=11,
                fontweight='bold', color=LIQUID_COLOR)

        ax.set_xlim(-4, liq_x + 3)
        ax.set_ylim(-2.5, top_y + 2.5)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return fig


def column_schematic(
    model,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    title: str = "CFNN Column Schematic",
    figsize: tuple = (10, 14),
    save_path: Optional[str] = None,
    style: str = "pid",
) -> plt.Figure:
    """
    Draw a column schematic. Wrapper that dispatches to the chosen style.

    Args:
        style: 'pid' for P&ID engineering style, 'sankey' for flow diagram.
    """
    if style == 'sankey':
        return column_schematic_sankey(
            model, x, context, title=title, figsize=figsize, save_path=save_path)
    return column_schematic_pid(
        model, x, context, title=title, figsize=figsize, save_path=save_path)
