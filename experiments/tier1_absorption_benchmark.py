"""
Tier 1 — AbsorptionNetwork benchmark.

Two tasks, same training loop, matched parameter counts:

  Task A — "moons": 2D non-linear classification (generic sanity check).
  Task B — "kremser-inverse": given noisy (y_feed, y_top, L/G, N) pairs,
           predict the Henry's constant m that generated them.  A task
           where knowing the Kremser physics should help.

We compare:
    MLP              — vanilla baseline
    AbsorptionNet    — the physically-exact tower as a layer

We report final test accuracy / MSE and the convergence curve.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from src.absorption_tower import AbsorptionNetwork, AbsorptionTower


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Task A — two moons
# ---------------------------------------------------------------------------

def task_moons(seed=0):
    torch.manual_seed(seed)
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    n_tr = 1600
    return (X[:n_tr], y[:n_tr]), (X[n_tr:], y[n_tr:])


# ---------------------------------------------------------------------------
# Task B — inverse Kremser regression
# ---------------------------------------------------------------------------

def generate_kremser_inverse(n=4000, seed=0):
    """Inputs: [y_feed, y_top_noisy, L/G, N/10]. Target: m."""
    g = torch.Generator().manual_seed(seed)
    y_feed = torch.rand(n, generator=g) * 0.5 + 0.1    # 0.1..0.6
    LG = torch.rand(n, generator=g) * 3.0 + 0.5         # 0.5..3.5
    m = torch.rand(n, generator=g) * 1.8 + 0.1          # 0.1..1.9
    Nstages = torch.randint(2, 12, (n,), generator=g).float()
    A = LG / m
    # Kremser: fraction = (A^{N+1} - A)/(A^{N+1} - 1)
    pow_N1 = A.pow(Nstages + 1)
    frac = torch.where(
        (A - 1.0).abs() < 1e-4,
        Nstages / (Nstages + 1.0),
        (pow_N1 - A) / (pow_N1 - 1.0),
    )
    y_top = y_feed * (1.0 - frac)
    y_top_noisy = y_top + torch.randn(n, generator=g) * 0.005

    X = torch.stack([y_feed, y_top_noisy, LG, Nstages / 10.0], dim=1)
    target = m.unsqueeze(1)
    return X, target


def task_kremser(seed=0):
    X, y = generate_kremser_inverse(n=4000, seed=seed)
    n_tr = 3200
    return (X[:n_tr], y[:n_tr]), (X[n_tr:], y[n_tr:])


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------

def train_model(model, train, test, *, n_epochs, loss_fn, metric_fn,
                lr=1e-3, batch_size=128, log_every=5, name="model"):
    Xtr, ytr = train
    Xte, yte = test
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    curve = []
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            pred = model(Xtr[b])
            loss = loss_fn(pred, ytr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(Xte)
            m_te = metric_fn(pred_te, yte)
        curve.append(m_te)
        if (epoch + 1) % log_every == 0:
            print(f"  [{name}] epoch {epoch+1:3d}/{n_epochs}  test_metric={m_te:.4f}")

    return curve


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(pred, y):
    return (pred.argmax(-1) == y).float().mean().item()


def regression_metric(pred, y):
    return float(nn.functional.mse_loss(pred, y).item())


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_moons():
    print("\n" + "=" * 70)
    print("TASK A — Two moons (binary classification)")
    print("=" * 70)
    train, test = task_moons(seed=0)

    torch.manual_seed(1)
    mlp = MLP(d_in=2, d_hidden=32, d_out=2, depth=3)
    torch.manual_seed(1)
    absn = AbsorptionNetwork(d_in=2, d_tower=16, d_out=2, n_stages=4,
                              L_over_G_init=1.5, m_init=0.7, E_init=0.7)
    print(f"  MLP params:          {param_count(mlp)}")
    print(f"  AbsorptionNet params:{param_count(absn)}")

    loss_fn = nn.CrossEntropyLoss()
    print("\n-- MLP --")
    c_mlp = train_model(mlp, train, test, n_epochs=40, loss_fn=loss_fn,
                         metric_fn=classification_metrics, name="MLP")
    print("\n-- AbsorptionNet --")
    c_abs = train_model(absn, train, test, n_epochs=40, loss_fn=loss_fn,
                         metric_fn=classification_metrics, name="AbsorptionNet")

    print("\nFinal test accuracy:")
    print(f"  MLP          : {c_mlp[-1]*100:.2f} %")
    print(f"  AbsorptionNet: {c_abs[-1]*100:.2f} %")
    return c_mlp, c_abs


def run_kremser():
    print("\n" + "=" * 70)
    print("TASK B — Inverse Kremser (regression, physics-structured)")
    print("=" * 70)
    train, test = task_kremser(seed=0)

    torch.manual_seed(1)
    mlp = MLP(d_in=4, d_hidden=64, d_out=1, depth=3)
    torch.manual_seed(1)
    absn = AbsorptionNetwork(d_in=4, d_tower=32, d_out=1, n_stages=6,
                              L_over_G_init=1.5, m_init=0.7, E_init=0.9)
    print(f"  MLP params:          {param_count(mlp)}")
    print(f"  AbsorptionNet params:{param_count(absn)}")

    loss_fn = nn.MSELoss()
    print("\n-- MLP --")
    c_mlp = train_model(mlp, train, test, n_epochs=50, loss_fn=loss_fn,
                         metric_fn=regression_metric, name="MLP", lr=3e-3)
    print("\n-- AbsorptionNet --")
    c_abs = train_model(absn, train, test, n_epochs=50, loss_fn=loss_fn,
                         metric_fn=regression_metric, name="AbsorptionNet", lr=3e-3)

    print("\nFinal test MSE (lower is better):")
    print(f"  MLP          : {c_mlp[-1]:.5f}")
    print(f"  AbsorptionNet: {c_abs[-1]:.5f}")
    return c_mlp, c_abs


def main() -> int:
    run_moons()
    run_kremser()
    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
