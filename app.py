"""
Gradio Space for the CounterFlow Neural Network.

An interactive tour of the physically-exact AbsorptionTower — a multistage
countercurrent gas absorber rendered as a differentiable PyTorch layer.

Design notes
------------
The UI follows an editorial aesthetic: warm cream canvas, serif display
type, terracotta accent, generous margins, and a single column of
information per tab.  The McCabe–Thiele diagram is the hero; every
other element defers to it.

Run locally:
    python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.absorption_tower import AbsorptionTower


# ============================================================================
# Design tokens — Anthropic-inspired palette
# ============================================================================

C_CANVAS      = "#F5F1E8"   # warm cream
C_SURFACE     = "#FAF7F0"   # lighter cream for panels
C_INK         = "#1A1915"   # near-black warm
C_SUBTLE      = "#6B6558"   # muted brown-grey for secondary text
C_ACCENT      = "#C96342"   # terracotta, signature
C_ACCENT_DK   = "#A04A2E"
C_TEAL        = "#3F7D6E"   # muted, for secondary series
C_STONE       = "#B8997A"   # warm stone, tertiary
C_RULE        = "#E3DACC"   # soft rule / border

FONT_SERIF = "ui-serif, 'Iowan Old Style', 'Apple Garamond', 'Baskerville', Georgia, serif"
FONT_SANS  = "ui-sans-serif, -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO  = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace"


# ---------------------------------------------------------------------------
# Matplotlib styling — matched to the UI
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor":      C_CANVAS,
    "axes.facecolor":        C_CANVAS,
    "savefig.facecolor":     C_CANVAS,
    "axes.edgecolor":        C_INK,
    "axes.labelcolor":       C_INK,
    "xtick.color":           C_INK,
    "ytick.color":           C_INK,
    "text.color":            C_INK,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.9,
    "font.family":           "serif",
    "font.serif":            ["DejaVu Serif", "Baskerville", "Georgia"],
    "mathtext.fontset":      "dejavuserif",
    "font.size":             11,
    "axes.titlesize":        12,
    "axes.titleweight":      "regular",
    "axes.titlepad":         14,
    "axes.labelpad":         8,
    "grid.color":            C_RULE,
    "grid.linewidth":        0.7,
    "grid.alpha":            1.0,
    "legend.frameon":        False,
    "legend.fontsize":       9.5,
})


# ============================================================================
# Tab 1 — Interactive absorber (McCabe-Thiele)
# ============================================================================

def _build_tower(y_feed, x_top, m, L_over_G, E, N, b):
    t = AbsorptionTower(
        d=1, n_stages=int(N),
        L_over_G_init=L_over_G, m_init=m,
        E_init=max(min(E, 0.9999), 1e-4),
        b_init=b,
    )
    with torch.no_grad():
        prof = t.profiles(
            torch.tensor([[float(y_feed)]]),
            torch.tensor([[float(x_top)]]),
        )
    return t, prof


def mccabe_thiele_figure(y_feed, x_top, m, L_over_G, E, N, b):
    N = int(N)
    t, prof = _build_tower(y_feed, x_top, m, L_over_G, E, N, b)

    y_st = prof["y_stages"][0, :, 0].numpy()
    x_st = prof["x_stages"][0, :, 0].numpy()
    y_top_val = float(prof["y_top"].item())
    x_bot_val = float(prof["x_bot"].item())

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    fig.patch.set_facecolor(C_CANVAS)

    # Equilibrium line
    x_lin = np.linspace(0, max(x_bot_val * 1.25, 1e-3), 200)
    y_eq = m * x_lin + b
    ax.plot(x_lin, y_eq, color=C_ACCENT, lw=2.0,
            label=f"Equilibrium    y* = {m:.2f}·x + {b:.2g}")

    # Operating line
    x_op = np.linspace(x_top, max(x_bot_val * 1.05, x_top + 1e-3), 200)
    y_op = y_top_val + L_over_G * (x_op - x_top)
    ax.plot(x_op, y_op, color=C_TEAL, lw=2.0,
            label=f"Operating      y = y₁ + (L/G)(x − x₀),  L/G = {L_over_G:.2f}")

    # Staircase — understated neutral
    for n in range(N):
        x_n_real = x_st[n]
        y_n = y_st[n]
        y_np1 = y_st[n + 1]
        x_np1 = x_st[n + 1]
        ax.plot([x_n_real, x_n_real], [y_n, y_np1],
                color=C_INK, lw=0.9, alpha=0.55)
        ax.plot([x_n_real, x_np1], [y_np1, y_np1],
                color=C_INK, lw=0.9, alpha=0.55)
        ax.plot(x_n_real, y_n, "o", color=C_INK, ms=3.5, alpha=0.85)
        ax.annotate(f"{n+1}", xy=(x_n_real, y_n),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=9, color=C_SUBTLE, va="center", alpha=0.9)

    # Endpoint markers
    ax.plot(x_top, y_top_val, "o", color=C_TEAL, ms=8, zorder=5)
    ax.plot(x_bot_val, y_feed, "s", color=C_ACCENT_DK, ms=8, zorder=5)

    ax.annotate(
        f"top\n($x_0$, $y_1$) = ({x_top:.3f}, {y_top_val:.4f})",
        xy=(x_top, y_top_val), xytext=(28, 28),
        textcoords="offset points", fontsize=9, color=C_TEAL,
        va="bottom", ha="left",
        arrowprops=dict(arrowstyle="-", color=C_TEAL, alpha=0.4, lw=0.7),
    )
    ax.annotate(
        f"bottom\n($x_N$, $y_{{N+1}}$) = ({x_bot_val:.3f}, {y_feed:.4f})",
        xy=(x_bot_val, y_feed), xytext=(-18, -42),
        textcoords="offset points", fontsize=9, color=C_ACCENT_DK,
        va="top", ha="right",
        arrowprops=dict(arrowstyle="-", color=C_ACCENT_DK, alpha=0.4, lw=0.7),
    )

    A_val = L_over_G / m
    frac = (y_feed - y_top_val) / max(y_feed - (m * x_top + b), 1e-12)
    ax.set_title(
        f"McCabe–Thiele    ·    N = {N}    ·    A = {A_val:.2f}    ·    "
        f"E = {E:.2f}    ·    absorbed {frac*100:.1f}%",
        fontsize=11.5, color=C_INK, pad=18,
    )
    ax.set_xlabel("x    liquid mole fraction of solute")
    ax.set_ylabel("y    gas mole fraction of solute")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def stage_profile_table(y_feed, x_top, m, L_over_G, E, N, b):
    N = int(N)
    _, prof = _build_tower(y_feed, x_top, m, L_over_G, E, N, b)
    y_st = prof["y_stages"][0, :, 0].numpy()
    x_st = prof["x_stages"][0, :, 0].numpy()
    rows = []
    for n in range(N + 1):
        if n == 0:
            where = "top — solvent inlet"
        elif n == N:
            where = "bottom — gas inlet"
        else:
            where = f"tray {n}"
        rows.append({
            "n": n,
            "location": where,
            "x_n": round(float(x_st[n]), 5),
            "y_{n+1}": round(float(y_st[n]), 5),
        })
    return pd.DataFrame(rows)


def kpi_card(y_feed, x_top, m, L_over_G, E, N, b):
    t, _ = _build_tower(y_feed, x_top, m, L_over_G, E, N, b)
    with torch.no_grad():
        y_top, x_bot = t(torch.tensor([[float(y_feed)]]),
                          torch.tensor([[float(x_top)]]))
    y_top_val = float(y_top.item())
    x_bot_val = float(x_bot.item())
    A = L_over_G / m
    frac = (y_feed - y_top_val) / max(y_feed - (m * x_top + b), 1e-12)

    if A > 1.05:
        regime = "Absorption feasible. A > 1."
        regime_class = "ok"
    elif abs(A - 1.0) <= 0.05:
        regime = "Pinched. A ≈ 1 — add stages or solvent."
        regime_class = "warn"
    else:
        regime = "Stripping regime. A < 1 — solute wants to leave the liquid."
        regime_class = "err"

    balance_lhs = y_feed - y_top_val
    balance_rhs = L_over_G * (x_bot_val - x_top)

    return f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Absorption factor</div>
    <div class="kpi-value">{A:.3f}</div>
    <div class="kpi-sub">A = L / (m · G)</div>
  </div>
  <div class="kpi-card accent">
    <div class="kpi-label">Absorbed</div>
    <div class="kpi-value">{frac*100:.1f}<span class="kpi-unit">%</span></div>
    <div class="kpi-sub">of the thermodynamically removable amount</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Lean gas    y₁</div>
    <div class="kpi-value small">{y_top_val:.5f}</div>
    <div class="kpi-sub">leaves the top of the tower</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Rich liquid    x_N</div>
    <div class="kpi-value small">{x_bot_val:.5f}</div>
    <div class="kpi-sub">leaves the bottom of the tower</div>
  </div>
</div>

<div class="regime {regime_class}">{regime}</div>

<div class="balance-check">
  <span class="balance-label">Mass balance</span>
  <span class="balance-eq">G·(y_feed − y₁)  <span class="eq">=</span>  L·(x_N − x₀)</span>
  <span class="balance-vals">{balance_lhs:.5f}    ·    {balance_rhs:.5f}</span>
</div>
"""


def update_absorber(y_feed, x_top, m, L_over_G, E, N, b):
    fig = mccabe_thiele_figure(y_feed, x_top, m, L_over_G, E, N, b)
    table = stage_profile_table(y_feed, x_top, m, L_over_G, E, N, b)
    card = kpi_card(y_feed, x_top, m, L_over_G, E, N, b)
    return fig, table, card


# ============================================================================
# Tab 2 — Training demo
# ============================================================================

def training_demo(true_m, true_LG, true_E, true_N, noise_sigma, n_train,
                  n_epochs):
    torch.manual_seed(0)
    gt = AbsorptionTower(
        d=1, n_stages=int(true_N),
        L_over_G_init=true_LG, m_init=true_m,
        E_init=min(max(true_E, 1e-4), 0.9999),
    )
    y_feed = torch.rand(int(n_train), 1) * 0.5 + 0.05
    x_top = torch.rand(int(n_train), 1) * 0.05
    with torch.no_grad():
        y_top_true, _ = gt(y_feed, x_top)
    y_top_obs = y_top_true + torch.randn_like(y_top_true) * noise_sigma

    model = AbsorptionTower(
        d=1, n_stages=int(true_N),
        L_over_G_init=0.8, m_init=1.2, E_init=0.5,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    hist = {"m": [], "L/G": [], "E": [], "loss": []}
    for _ in range(int(n_epochs)):
        opt.zero_grad()
        pred, _ = model(y_feed, x_top)
        loss = (pred - y_top_obs).pow(2).mean()
        loss.backward()
        opt.step()
        hist["m"].append(float(model.m.item()))
        hist["L/G"].append(float(model.L_over_G.item()))
        hist["E"].append(float(model.E.item()))
        hist["loss"].append(float(loss.item()))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor(C_CANVAS)

    ax = axes[0]
    ax.plot(hist["loss"], color=C_INK, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss    (log)")
    ax.set_title("Training loss")
    ax.grid(True)

    ax = axes[1]
    ax.plot(hist["m"],   color=C_ACCENT,   lw=1.8, label="m")
    ax.axhline(true_m,   color=C_ACCENT,   ls=(0, (3, 3)), alpha=0.45)
    ax.plot(hist["L/G"], color=C_TEAL,     lw=1.8, label="L/G")
    ax.axhline(true_LG,  color=C_TEAL,     ls=(0, (3, 3)), alpha=0.45)
    ax.plot(hist["E"],   color=C_STONE,    lw=1.8, label="E")
    ax.axhline(true_E,   color=C_STONE,    ls=(0, (3, 3)), alpha=0.45)
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameter value")
    ax.set_title("Learned    vs    ground truth (dashed)")
    ax.legend(loc="center right")
    ax.grid(True)
    fig.tight_layout()

    def pct(a, b):
        return abs(a - b) / max(abs(b), 1e-12) * 100

    summary = f"""
<div class="recovery">
  <div class="recovery-title">Recovery</div>
  <table class="recovery-table">
    <thead><tr><th>parameter</th><th>truth</th><th>learned</th><th>error</th></tr></thead>
    <tbody>
      <tr><td>m</td>    <td>{true_m:.3f}</td>  <td>{hist['m'][-1]:.3f}</td>    <td>{pct(hist['m'][-1], true_m):.1f}%</td></tr>
      <tr><td>L/G</td>  <td>{true_LG:.3f}</td> <td>{hist['L/G'][-1]:.3f}</td>  <td>{pct(hist['L/G'][-1], true_LG):.1f}%</td></tr>
      <tr><td>E</td>    <td>{true_E:.3f}</td>  <td>{hist['E'][-1]:.3f}</td>    <td>{pct(hist['E'][-1], true_E):.1f}%</td></tr>
    </tbody>
  </table>
  <div class="recovery-note">Final training MSE  ·  <span class="mono">{hist['loss'][-1]:.2e}</span></div>
</div>
"""
    return fig, summary


# ============================================================================
# Tab 3 — About
# ============================================================================

ABOUT_MD = """
### A physically-exact tower, in closed form

At every tray *n*, three equations hold simultaneously:

<div class="eq-block">
<strong>Equilibrium</strong>    y*<sub>n</sub> = m · x<sub>n</sub> + b<br>
<strong>Murphree efficiency</strong>    y<sub>n</sub> = y<sub>n+1</sub> + E · (y*<sub>n</sub> − y<sub>n+1</sub>)<br>
<strong>Operating line</strong>    y<sub>n+1</sub> = y<sub>1</sub> + (L/G) · (x<sub>n</sub> − x<sub>0</sub>)
</div>

Substituting gives a linear recurrence in y<sub>n</sub> that collapses to a
closed-form solution for y<sub>1</sub> — the Kremser equation, generalised
to real trays.  Every term (m, L/G, E, b) is a learnable PyTorch
parameter, so the whole tower is a single differentiable operation.

### Validated against the textbook

- Treybal (1980) Ex. 8.2 — acetone in air–water
- Seader et al. (2011) Ex. 6.1 — n-butane absorber
- A = 1 pinch case  —  handled by a removable-singularity branch
- Real-tray Murphree E = 0.7 — agrees with brute-force tray-by-tray iteration

All four match reference values to better than 10<sup>−3</sup> relative error.

### What this demo shows

The interactive tab lets you move a tower through its design space and
watch the McCabe–Thiele diagram redraw itself under exact physics.  The
training tab shows the same module learning its own parameters back from
noisy observations — the layer works as both a solver and a learnable
component.

Source: [`src/absorption_tower.py`](https://github.com/DanielRegaladoUMiami/counterflow-nn/blob/main/src/absorption_tower.py) in the repo.
"""


# ============================================================================
# CSS — editorial, warm, quiet
# ============================================================================

CUSTOM_CSS = f"""
:root {{
    --canvas:  {C_CANVAS};
    --surface: {C_SURFACE};
    --ink:     {C_INK};
    --subtle:  {C_SUBTLE};
    --accent:  {C_ACCENT};
    --accent-dark: {C_ACCENT_DK};
    --rule:    {C_RULE};
    --teal:    {C_TEAL};
}}

html, body, .gradio-container, gradio-app {{
    background: var(--canvas) !important;
    color: var(--ink) !important;
    font-family: {FONT_SANS};
}}

.gradio-container {{
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 48px 32px 72px !important;
}}

/* --- Hero --- */
.hero {{
    border-bottom: 1px solid var(--rule);
    padding-bottom: 28px;
    margin-bottom: 36px;
}}
.hero h1 {{
    font-family: {FONT_SERIF};
    font-weight: 400;
    font-size: 44px;
    line-height: 1.08;
    letter-spacing: -0.02em;
    margin: 0 0 14px 0;
    color: var(--ink);
}}
.hero h1 em {{
    font-style: italic;
    color: var(--accent);
}}
.hero p {{
    font-family: {FONT_SERIF};
    font-size: 18px;
    line-height: 1.55;
    color: var(--subtle);
    max-width: 720px;
    margin: 0;
}}
.eyebrow {{
    display: inline-block;
    font-family: {FONT_SANS};
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    font-weight: 600;
}}

/* --- Tabs --- */
.tab-nav button, .tabs button {{
    font-family: {FONT_SANS} !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    color: var(--subtle) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 10px 4px !important;
    margin-right: 22px !important;
    font-weight: 500 !important;
}}
.tab-nav button.selected, .tabs button.selected {{
    color: var(--ink) !important;
    border-bottom-color: var(--accent) !important;
}}

/* --- Panels --- */
.panel, .block {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* --- Section headers --- */
.section-title {{
    font-family: {FONT_SERIF};
    font-size: 22px;
    font-weight: 400;
    letter-spacing: -0.01em;
    margin: 28px 0 6px 0;
    color: var(--ink);
}}
.section-desc {{
    font-family: {FONT_SERIF};
    font-size: 15px;
    line-height: 1.55;
    color: var(--subtle);
    margin: 0 0 20px 0;
    max-width: 680px;
}}

/* --- Sliders --- */
.gr-slider label, label span {{
    font-family: {FONT_MONO} !important;
    font-size: 12px !important;
    color: var(--subtle) !important;
    letter-spacing: 0.02em !important;
    font-weight: 500 !important;
}}
input[type="range"] {{
    accent-color: var(--accent) !important;
}}

/* --- KPI grid --- */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 1px;
    background: var(--rule);
    border: 1px solid var(--rule);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0 18px 0;
}}
.kpi-card {{
    background: var(--surface);
    padding: 18px 20px;
}}
.kpi-card.accent {{
    background: #FDF4EF;
}}
.kpi-label {{
    font-family: {FONT_MONO};
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--subtle);
    margin-bottom: 10px;
}}
.kpi-value {{
    font-family: {FONT_SERIF};
    font-size: 32px;
    font-weight: 400;
    letter-spacing: -0.015em;
    color: var(--ink);
    line-height: 1;
}}
.kpi-value.small {{
    font-size: 22px;
}}
.kpi-value .kpi-unit {{
    font-size: 18px;
    color: var(--subtle);
    margin-left: 2px;
}}
.kpi-sub {{
    font-family: {FONT_SANS};
    font-size: 11.5px;
    color: var(--subtle);
    margin-top: 8px;
    line-height: 1.4;
}}
.kpi-card.accent .kpi-value {{
    color: var(--accent-dark);
}}

.regime {{
    font-family: {FONT_SERIF};
    font-size: 14px;
    padding: 12px 16px;
    border-left: 3px solid var(--accent);
    background: var(--surface);
    color: var(--ink);
    margin-bottom: 14px;
}}
.regime.warn {{ border-color: #D4A84B; }}
.regime.err  {{ border-color: #9E4A3A; color: #9E4A3A; }}

.balance-check {{
    font-family: {FONT_MONO};
    font-size: 12px;
    color: var(--subtle);
    padding: 10px 0;
    border-top: 1px solid var(--rule);
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
}}
.balance-label {{
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 10.5px;
    color: var(--subtle);
}}
.balance-eq {{ color: var(--ink); }}
.balance-eq .eq {{ color: var(--accent); font-weight: bold; }}
.balance-vals {{ color: var(--ink); letter-spacing: 0.02em; }}

/* --- Recovery table --- */
.recovery {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 22px 24px;
    margin-top: 8px;
}}
.recovery-title {{
    font-family: {FONT_MONO};
    font-size: 10.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--subtle);
    margin-bottom: 14px;
}}
.recovery-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: {FONT_SERIF};
    font-size: 16px;
}}
.recovery-table th {{
    text-align: left;
    padding: 8px 12px 8px 0;
    font-family: {FONT_MONO};
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--subtle);
    font-weight: 500;
    border-bottom: 1px solid var(--rule);
}}
.recovery-table td {{
    padding: 10px 12px 10px 0;
    border-bottom: 1px solid var(--rule);
    color: var(--ink);
}}
.recovery-table tr:last-child td {{ border-bottom: none; }}
.recovery-table td:first-child {{ font-family: {FONT_MONO}; color: var(--accent); }}
.recovery-note {{
    margin-top: 14px;
    font-family: {FONT_SANS};
    font-size: 12px;
    color: var(--subtle);
}}
.recovery-note .mono {{ font-family: {FONT_MONO}; color: var(--ink); }}

/* --- About copy --- */
.about h3, .markdown h3 {{
    font-family: {FONT_SERIF} !important;
    font-size: 20px !important;
    font-weight: 400 !important;
    margin: 28px 0 10px 0 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em !important;
}}
.about p, .markdown p {{
    font-family: {FONT_SERIF};
    font-size: 15.5px;
    line-height: 1.65;
    color: var(--ink);
    max-width: 720px;
}}
.about a, .markdown a {{
    color: var(--accent-dark);
    text-decoration: underline;
    text-decoration-color: var(--rule);
    text-underline-offset: 3px;
}}
.eq-block {{
    font-family: {FONT_MONO};
    font-size: 13px;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 16px 20px;
    margin: 16px 0;
    line-height: 1.9;
    color: var(--ink);
}}
.eq-block strong {{ color: var(--accent-dark); font-weight: 600; }}

/* --- Buttons --- */
button.primary, .gr-button-primary, button[variant="primary"] {{
    background: var(--ink) !important;
    color: var(--canvas) !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: {FONT_SANS} !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 22px !important;
}}
button.primary:hover, .gr-button-primary:hover {{
    background: var(--accent) !important;
}}

/* --- Dataframe --- */
table {{
    font-family: {FONT_MONO} !important;
    font-size: 12.5px !important;
}}
table thead th {{
    background: var(--surface) !important;
    color: var(--subtle) !important;
    font-size: 10.5px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    border-bottom: 1px solid var(--rule) !important;
}}
table tbody td {{
    color: var(--ink) !important;
    border-bottom: 1px solid var(--rule) !important;
    padding: 8px 12px !important;
}}

/* --- Hide gradio branding --- */
footer {{ display: none !important; }}

/* --- Footer (ours) --- */
.page-foot {{
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid var(--rule);
    font-family: {FONT_SANS};
    font-size: 12px;
    color: var(--subtle);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}}
.page-foot a {{ color: var(--accent-dark); text-decoration: none; }}
.page-foot a:hover {{ text-decoration: underline; }}
"""


# ============================================================================
# Build the app
# ============================================================================

HERO = """
<div class="hero">
  <div class="eyebrow">CounterFlow Neural Network</div>
  <h1>A gas absorber, written<br/>as a <em>differentiable layer</em>.</h1>
  <p>Multistage countercurrent absorption, solved in closed form with Kremser and
  Murphree efficiency. Every physical parameter is learnable. Move the sliders —
  the McCabe–Thiele diagram redraws under exact physics.</p>
</div>
"""

FOOT = """
<div class="page-foot">
  <div>CounterFlow NN    ·    physically-exact AbsorptionTower    ·    built with PyTorch</div>
  <div><a href="https://github.com/DanielRegaladoUMiami/counterflow-nn">source on github</a></div>
</div>
"""


def build_app():
    with gr.Blocks(title="CounterFlow NN — AbsorptionTower") as demo:
        gr.HTML(HERO)

        with gr.Tabs():
            with gr.Tab("Interactive tower"):
                gr.HTML(
                    "<div class='section-title'>The tower, live.</div>"
                    "<p class='section-desc'>Seven physical knobs. Every change "
                    "propagates through the exact Kremser solution and redraws the "
                    "diagram, the stage table, and the design KPIs.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        y_feed = gr.Slider(0.001, 0.3, value=0.05, step=0.001,
                                            label="y_feed     inlet gas mole fraction")
                        x_top = gr.Slider(0.0, 0.1, value=0.0, step=0.001,
                                           label="x_top      inlet solvent mole fraction")
                        m = gr.Slider(0.1, 3.0, value=0.7, step=0.05,
                                       label="m          Henry's constant")
                        LG = gr.Slider(0.1, 4.0, value=1.5, step=0.05,
                                        label="L/G        solvent / gas molar ratio")
                        E = gr.Slider(0.05, 1.0, value=0.85, step=0.05,
                                       label="E          Murphree plate efficiency")
                        N = gr.Slider(1, 15, value=6, step=1,
                                       label="N          number of equilibrium stages")
                        b = gr.Slider(-0.02, 0.05, value=0.0, step=0.005,
                                       label="b          equilibrium intercept")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label=None, show_label=False)
                        kpis = gr.HTML()

                gr.HTML(
                    "<div class='section-title'>Stage-by-stage profile</div>"
                    "<p class='section-desc'>Composition on every tray, top to "
                    "bottom. The top of the table is the solvent inlet; the bottom "
                    "is the gas feed.</p>"
                )
                table = gr.Dataframe(label=None, show_label=False,
                                      interactive=False)

                inputs = [y_feed, x_top, m, LG, E, N, b]
                outputs = [plot, table, kpis]
                for w in inputs:
                    w.change(update_absorber, inputs=inputs, outputs=outputs)
                demo.load(update_absorber, inputs=inputs, outputs=outputs)

            with gr.Tab("Learn from data"):
                gr.HTML(
                    "<div class='section-title'>Can the tower rediscover its own physics?</div>"
                    "<p class='section-desc'>We generate noisy y_top observations from a "
                    "ground-truth tower, then fit a fresh AbsorptionTower starting from "
                    "poor initial guesses. Gradient descent should walk the learned "
                    "parameters back to the truth.</p>"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        true_m = gr.Slider(0.2, 2.0, value=0.7, step=0.05,
                                             label="true m")
                        true_LG = gr.Slider(0.5, 3.0, value=1.8, step=0.05,
                                              label="true L/G")
                        true_E = gr.Slider(0.2, 0.99, value=0.85, step=0.05,
                                             label="true E")
                        true_N = gr.Slider(2, 10, value=5, step=1,
                                             label="N stages")
                        noise = gr.Slider(0.0, 0.02, value=0.003, step=0.001,
                                            label="noise σ on y_top")
                        n_train = gr.Slider(16, 512, value=128, step=16,
                                              label="training samples")
                        n_epochs = gr.Slider(50, 1000, value=300, step=50,
                                               label="epochs")
                        go = gr.Button("Fit the tower", variant="primary")
                    with gr.Column(scale=2):
                        curve_plot = gr.Plot(label=None, show_label=False)
                        summary = gr.HTML()

                go.click(training_demo,
                         inputs=[true_m, true_LG, true_E, true_N,
                                 noise, n_train, n_epochs],
                         outputs=[curve_plot, summary])

            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD, elem_classes=["about"])

        gr.HTML(FOOT)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
