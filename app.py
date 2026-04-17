"""
Gradio Space for the CounterFlow Neural Network.

Headline demo: the physically-exact AbsorptionTower, rendered as a live
McCabe–Thiele diagram with sliders for every physical parameter.

Tabs:
    1. Interactive Absorber  —  sliders → McCabe–Thiele diagram + KPIs
    2. Training Demo         —  fit the tower to noisy data, inspect
                                learned (m, L/G, E) against ground truth.
    3. About                 —  brief physics recap.

Run locally:
    python app.py
"""

from __future__ import annotations

import io
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
# Tab 1 — Interactive absorber (McCabe-Thiele)
# ============================================================================

def mccabe_thiele_figure(y_feed, x_top, m, L_over_G, E, N, b):
    """Render operating line + equilibrium line + staircase between stages."""
    N = int(N)
    t = AbsorptionTower(
        d=1, n_stages=N,
        L_over_G_init=L_over_G, m_init=m,
        E_init=max(min(E, 0.9999), 1e-4),
        b_init=b,
    )
    with torch.no_grad():
        prof = t.profiles(
            torch.tensor([[float(y_feed)]]),
            torch.tensor([[float(x_top)]]),
        )

    y_st = prof["y_stages"][0, :, 0].numpy()   # length N+1, index 0 = y_1, N = y_{N+1}
    x_st = prof["x_stages"][0, :, 0].numpy()   # length N+1
    y_top_val = float(prof["y_top"].item())
    x_bot_val = float(prof["x_bot"].item())

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Equilibrium line y* = m·x + b over the range of interest
    x_lin = np.linspace(0, max(x_bot_val * 1.2, 1e-3), 200)
    y_eq = m * x_lin + b
    ax.plot(x_lin, y_eq, color="#c0392b", lw=2, label=f"Equilibrium y* = {m:.2f}x + {b:.2g}")

    # Operating line through (x_top, y_top) with slope L/G
    x_op = np.linspace(x_top, max(x_bot_val * 1.05, x_top + 1e-3), 200)
    y_op = y_top_val + L_over_G * (x_op - x_top)
    ax.plot(x_op, y_op, color="#2980b9", lw=2,
            label=f"Operating y = y₁ + (L/G)(x − x₀),  L/G = {L_over_G:.2f}")

    # Endpoints
    ax.plot(x_top, y_top_val, "o", color="#2980b9", ms=9, zorder=5)
    ax.plot(x_bot_val, y_feed, "s", color="#16a085", ms=9, zorder=5)
    ax.annotate(f"top\n(x₀, y₁) = ({x_top:.3f}, {y_top_val:.3f})",
                xy=(x_top, y_top_val), xytext=(10, 15),
                textcoords="offset points", fontsize=9,
                color="#2980b9")
    ax.annotate(f"bottom\n(x_N, y_feed) = ({x_bot_val:.3f}, {y_feed:.3f})",
                xy=(x_bot_val, y_feed), xytext=(10, -25),
                textcoords="offset points", fontsize=9,
                color="#16a085")

    # Staircase between operating and equilibrium lines
    # We go from (x_0, y_1) up to (x_N, y_{N+1}) using real computed stages.
    for n in range(N):
        x_n = x_st[n] if n > 0 else x_top
        # horizontal segment: (x_n, y_{n+1}) ← on the operating line
        # First draw the vertical from tray n level to operating line:
        # Using computed x_n and y_{n+1}:
        x_n_real = x_st[n]
        y_np1 = y_st[n + 1]
        # horizontal (equilibrium side): from (x_n_real, y_{n})  to (x_n_real, y_{n+1})
        y_n = y_st[n]
        ax.plot([x_n_real, x_n_real], [y_n, y_np1],
                color="#7f8c8d", lw=1.1, alpha=0.9)
        # vertical (operating side): from (x_n_real, y_{n+1}) to (x_{n+1}, y_{n+1})
        x_np1 = x_st[n + 1]
        ax.plot([x_n_real, x_np1], [y_np1, y_np1],
                color="#7f8c8d", lw=1.1, alpha=0.9)
        ax.plot(x_n_real, y_n, "ko", ms=4)
        ax.text(x_n_real + 0.002, y_n + 0.002, f"{n+1}", fontsize=8,
                color="#2c3e50")

    ax.set_xlabel("x  —  liquid mole fraction of solute", fontsize=11)
    ax.set_ylabel("y  —  gas mole fraction of solute", fontsize=11)
    A_val = L_over_G / m
    frac = (y_feed - y_top_val) / max(y_feed - (m * x_top + b), 1e-12)
    ax.set_title(
        f"McCabe–Thiele  |  N = {N},  A = L/(mG) = {A_val:.2f},  "
        f"E = {E:.2f},  absorbed = {frac*100:.1f}%",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def stage_profile_table(y_feed, x_top, m, L_over_G, E, N, b):
    """Return a dataframe of the tray-by-tray compositions."""
    N = int(N)
    t = AbsorptionTower(
        d=1, n_stages=N,
        L_over_G_init=L_over_G, m_init=m,
        E_init=max(min(E, 0.9999), 1e-4),
        b_init=b,
    )
    with torch.no_grad():
        prof = t.profiles(
            torch.tensor([[float(y_feed)]]),
            torch.tensor([[float(x_top)]]),
        )
    y_st = prof["y_stages"][0, :, 0].numpy()
    x_st = prof["x_stages"][0, :, 0].numpy()
    rows = []
    for n in range(N + 1):
        if n == 0:
            label = "top (solvent in)"
        elif n == N:
            label = "bottom (feed gas)"
        else:
            label = f"tray {n}"
        rows.append({
            "n": n,
            "location": label,
            "x_n": float(x_st[n]),
            "y_{n+1}": float(y_st[n]),
        })
    return pd.DataFrame(rows)


def kpis_md(y_feed, x_top, m, L_over_G, E, N, b):
    y_top, x_bot = AbsorptionTower(
        d=1, n_stages=int(N),
        L_over_G_init=L_over_G, m_init=m,
        E_init=max(min(E, 0.9999), 1e-4),
        b_init=b,
    )(torch.tensor([[float(y_feed)]]), torch.tensor([[float(x_top)]]))

    y_top_val = float(y_top.item())
    x_bot_val = float(x_bot.item())
    A = L_over_G / m
    frac = (y_feed - y_top_val) / max(y_feed - (m * x_top + b), 1e-12)

    status = "✅ A > 1, absorption feasible" if A > 1.0 else (
        "⚠️ A ≈ 1, pinched — need more stages or more solvent"
        if abs(A - 1.0) < 0.05 else "❌ A < 1, gas stripping regime (wrong direction)"
    )
    return (
        f"### KPIs\n"
        f"- **Absorption factor A = L/(mG)** : `{A:.3f}`\n"
        f"- **Fraction absorbed** : `{frac*100:.2f} %`\n"
        f"- **y₁ (lean gas out)** : `{y_top_val:.5f}`\n"
        f"- **x_N (rich liquid out)** : `{x_bot_val:.5f}`\n"
        f"- **Mass-balance check** `G(y_feed − y₁) vs L(x_N − x₀)` : "
        f"`{y_feed - y_top_val:.5f}` vs `{L_over_G * (x_bot_val - x_top):.5f}`\n\n"
        f"{status}"
    )


def update_absorber(y_feed, x_top, m, L_over_G, E, N, b):
    fig = mccabe_thiele_figure(y_feed, x_top, m, L_over_G, E, N, b)
    table = stage_profile_table(y_feed, x_top, m, L_over_G, E, N, b)
    kpis = kpis_md(y_feed, x_top, m, L_over_G, E, N, b)
    return fig, table, kpis


# ============================================================================
# Tab 2 — Training demo
# ============================================================================

def training_demo(true_m, true_LG, true_E, true_N, noise_sigma, n_train,
                  n_epochs):
    """
    Generate noisy data from a ground-truth tower, fit an AbsorptionTower
    starting from random (bad) guesses, and show how close the learned
    parameters get to truth.
    """
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

    # Start from bad guesses
    model = AbsorptionTower(d=1, n_stages=int(true_N),
                             L_over_G_init=0.8, m_init=1.2, E_init=0.5)
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

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(hist["loss"], color="#34495e")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hist["m"], label="m", color="#c0392b")
    ax.axhline(true_m, color="#c0392b", ls="--", alpha=0.4)
    ax.plot(hist["L/G"], label="L/G", color="#2980b9")
    ax.axhline(true_LG, color="#2980b9", ls="--", alpha=0.4)
    ax.plot(hist["E"], label="E", color="#16a085")
    ax.axhline(true_E, color="#16a085", ls="--", alpha=0.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameter value")
    ax.set_title("Learned vs ground truth (dashed)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    summary = (
        f"### Recovery\n"
        f"| parameter | ground truth | learned | error |\n"
        f"|-----------|--------------|---------|-------|\n"
        f"| m         | {true_m:.3f} | {hist['m'][-1]:.3f} | "
        f"{abs(hist['m'][-1] - true_m)/true_m*100:.1f} % |\n"
        f"| L/G       | {true_LG:.3f} | {hist['L/G'][-1]:.3f} | "
        f"{abs(hist['L/G'][-1] - true_LG)/true_LG*100:.1f} % |\n"
        f"| E         | {true_E:.3f} | {hist['E'][-1]:.3f} | "
        f"{abs(hist['E'][-1] - true_E)/true_E*100:.1f} % |\n\n"
        f"Final training MSE: `{hist['loss'][-1]:.2e}`"
    )
    return fig, summary


# ============================================================================
# Tab 3 — About
# ============================================================================

ABOUT_MD = """
## CounterFlow Neural Network — AbsorptionTower

A PyTorch module that implements a multistage countercurrent gas absorber
**in closed form**, with every physical parameter learnable by gradient
descent.

At each tray *n*:

- **Equilibrium:** y*ₙ = m·xₙ + b
- **Murphree efficiency:** yₙ = y_{n+1} + E·(y*ₙ − y_{n+1})
- **Operating line:** y_{n+1} = y₁ + (L/G)·(xₙ − x₀)

Substituting yields the linear recurrence

$$y_n = \\beta\\,y_{n+1} + \\gamma\\,y_1 + \\delta,\\ \\ \\beta=(1{-}E)+E\\,m G/L,$$

which collapses to the closed-form Kremser solution for y₁. The whole
tower becomes a single differentiable step — no tray-by-tray iteration,
no fixed-point solver.

The module is validated against:

- **Treybal (1980) Ex. 8.2** — acetone/air/water absorber
- **Seader (2011) Ex. 6.1** — n-butane oil absorber
- **A = 1 pinch case** — stable via removable-singularity branch
- **Murphree real-tray case** — matches tray-by-tray iteration

Source code: the `src/absorption_tower.py` module in this repo.
"""


# ============================================================================
# Build the app
# ============================================================================

def build_app():
    with gr.Blocks(title="CounterFlow NN — AbsorptionTower") as demo:
        gr.Markdown("# CounterFlow Neural Network — AbsorptionTower")
        gr.Markdown(
            "*Multistage countercurrent gas absorber as a differentiable "
            "PyTorch layer.  Move the sliders to watch the McCabe–Thiele "
            "diagram rearrange itself — the physics is enforced exactly.*"
        )

        with gr.Tab("Interactive absorber"):
            with gr.Row():
                with gr.Column(scale=1):
                    y_feed = gr.Slider(0.001, 0.3, value=0.05, step=0.001,
                                        label="y_feed (inlet gas mole fraction)")
                    x_top = gr.Slider(0.0, 0.1, value=0.0, step=0.001,
                                       label="x_top (inlet solvent mole fraction)")
                    m = gr.Slider(0.1, 3.0, value=0.7, step=0.05,
                                   label="m  (Henry's constant)")
                    LG = gr.Slider(0.1, 4.0, value=1.5, step=0.05,
                                    label="L/G  (solvent/gas molar ratio)")
                    E = gr.Slider(0.05, 1.0, value=0.85, step=0.05,
                                   label="E  (Murphree plate efficiency)")
                    N = gr.Slider(1, 15, value=6, step=1,
                                   label="N  (number of stages)")
                    b = gr.Slider(-0.02, 0.05, value=0.0, step=0.005,
                                   label="b  (equilibrium intercept)")
                with gr.Column(scale=2):
                    plot = gr.Plot(label="McCabe–Thiele")
                    kpis = gr.Markdown()
            table = gr.Dataframe(label="Stage-by-stage compositions",
                                  interactive=False)

            inputs = [y_feed, x_top, m, LG, E, N, b]
            outputs = [plot, table, kpis]
            for w in inputs:
                w.change(update_absorber, inputs=inputs, outputs=outputs)
            demo.load(update_absorber, inputs=inputs, outputs=outputs)

        with gr.Tab("Training demo"):
            gr.Markdown(
                "**Can the tower recover its own physical parameters?**  \n"
                "We simulate noisy `y_top` observations from a ground-truth "
                "tower, then fit a fresh `AbsorptionTower` starting from bad "
                "guesses (m=1.2, L/G=0.8, E=0.5).  Gradient descent should "
                "walk the learned parameters back to the truth."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    true_m = gr.Slider(0.2, 2.0, value=0.7, step=0.05, label="true m")
                    true_LG = gr.Slider(0.5, 3.0, value=1.8, step=0.05, label="true L/G")
                    true_E = gr.Slider(0.2, 0.99, value=0.85, step=0.05, label="true E")
                    true_N = gr.Slider(2, 10, value=5, step=1, label="N stages")
                    noise = gr.Slider(0.0, 0.02, value=0.003, step=0.001,
                                        label="noise σ on y_top")
                    n_train = gr.Slider(16, 512, value=128, step=16,
                                          label="# training samples")
                    n_epochs = gr.Slider(50, 1000, value=300, step=50,
                                           label="# epochs")
                    go = gr.Button("Train", variant="primary")
                with gr.Column(scale=2):
                    curve_plot = gr.Plot(label="Training curves")
                    summary = gr.Markdown()
            go.click(training_demo,
                     inputs=[true_m, true_LG, true_E, true_N, noise, n_train, n_epochs],
                     outputs=[curve_plot, summary])

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
