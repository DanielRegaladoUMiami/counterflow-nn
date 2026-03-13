"""
Tier 1 — Proof of concept: Train CFNN-A on synthetic 2D datasets.

Datasets: moons, circles, XOR
Goal: Verify CFNN learns and compare with MLP baseline.

Usage:
    python experiments/tier1_synthetic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.network import CounterFlowNetwork
from src.utils import load_synthetic_dataset, prepare_data, train_model


class MLPBaseline(torch.nn.Module):
    """Simple MLP baseline with comparable parameter count."""
    def __init__(self, d_in, d_hidden, d_out, n_layers=3):
        super().__init__()
        layers = [torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(d_hidden, d_hidden), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(d_hidden, d_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def plot_decision_boundary(model, X, y, title="", ax=None, device="cpu"):
    """Plot 2D decision boundary."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.no_grad():
        Z = model(grid).argmax(-1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k", s=20)
    ax.set_title(title)
    return ax


def run_experiment(dataset_name, n_seeds=5):
    """Run CFNN vs MLP on a synthetic dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    cfnn_accs, mlp_accs = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(n_seeds):
        X, y = load_synthetic_dataset(dataset_name, n_samples=1000, seed=seed)
        train_loader, test_loader, d_in, n_classes = prepare_data(X, y, seed=seed)

        # CFNN-A
        cfnn = CounterFlowNetwork(d_in=d_in, d_gas=32, d_liquid=32, n_plates=5, d_out=n_classes, n_sweeps=2)
        h1 = train_model(cfnn, train_loader, test_loader, n_epochs=100, lr=1e-3, device=device, verbose=False)
        cfnn_acc = max(h1["test_metrics"])

        # MLP baseline (match params roughly)
        mlp = MLPBaseline(d_in=d_in, d_hidden=45, d_out=n_classes, n_layers=3)
        h2 = train_model(mlp, train_loader, test_loader, n_epochs=100, lr=1e-3, device=device, verbose=False)
        mlp_acc = max(h2["test_metrics"])

        cfnn_accs.append(cfnn_acc)
        mlp_accs.append(mlp_acc)

        if seed == 0:
            print(f"  CFNN params: {cfnn.count_parameters()}, MLP params: {mlp.count_parameters()}")

    print(f"  CFNN Accuracy: {np.mean(cfnn_accs):.4f} ± {np.std(cfnn_accs):.4f}")
    print(f"  MLP  Accuracy: {np.mean(mlp_accs):.4f} ± {np.std(mlp_accs):.4f}")

    return {
        "dataset": dataset_name,
        "cfnn_mean": np.mean(cfnn_accs), "cfnn_std": np.std(cfnn_accs),
        "mlp_mean": np.mean(mlp_accs), "mlp_std": np.std(mlp_accs),
    }


if __name__ == "__main__":
    results = []
    for ds in ["moons", "circles", "xor"]:
        results.append(run_experiment(ds))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'CFNN':>18} {'MLP':>18}")
    print("-" * 50)
    for r in results:
        print(f"{r['dataset']:<12} {r['cfnn_mean']:.4f}±{r['cfnn_std']:.4f}   {r['mlp_mean']:.4f}±{r['mlp_std']:.4f}")
