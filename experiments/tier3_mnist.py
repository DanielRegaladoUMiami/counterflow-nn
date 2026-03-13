"""
Tier 3: CFNN-A and CFNN-D on MNIST and FashionMNIST.

This is the scale-up test: can CFNN handle higher-dimensional inputs
(784 = 28x28 flattened images) and 10 classes?

Usage:
    python experiments/tier3_mnist.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.network import CounterFlowNetwork
from src.distillation import DistillationNetwork
from src.diagnostics import print_diagnostics
from src.utils import train_model


class MLPBaseline(nn.Module):
    def __init__(self, d_in=784, d_hidden=128, d_out=10, n_layers=4):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FlattenWrapper(nn.Module):
    """Wrapper to flatten image inputs for CFNN models."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def count_parameters(self):
        return self.model.count_parameters()


def get_mnist_loaders(dataset_name="MNIST", batch_size=128):
    """Get MNIST or FashionMNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset_cls = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    train_ds = dataset_cls(root='./data', train=True, download=True, transform=transform)
    test_ds = dataset_cls(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def build_models_tier3(d_in=784, d_out=10, d_gas=64, d_liquid=64):
    """Build models for MNIST-scale experiments."""
    cfnn_a = FlattenWrapper(CounterFlowNetwork(
        d_in=d_in, d_gas=d_gas, d_liquid=d_liquid,
        n_plates=6, d_out=d_out, n_sweeps=2,
    ))
    cfnn_d = FlattenWrapper(DistillationNetwork(
        d_in=d_in, d_gas=d_gas, d_liquid=d_liquid,
        n_plates_rect=3, n_plates_strip=3,
        d_out=d_out, n_sweeps=2,
        reflux_ratio=0.3, reboil_ratio=0.2,
    ))
    # MLP with comparable params
    target = cfnn_d.count_parameters()
    d_h = max(32, int(np.sqrt(target / 4)))
    mlp = MLPBaseline(d_in, d_h, d_out, n_layers=4)

    return {
        "CFNN-A": cfnn_a,
        "CFNN-D": cfnn_d,
        "MLP": mlp,
    }


def run_tier3(dataset_name="MNIST", n_seeds=3, n_epochs=20):
    """Run Tier 3 comparison on MNIST or FashionMNIST."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader = get_mnist_loaders(dataset_name, batch_size=128)
        models = build_models_tier3()

        for name, model in models.items():
            print(f"  {dataset_name} | {name} | seed {seed} | params={model.count_parameters()}")

            h = train_model(
                model, train_loader, test_loader,
                n_epochs=n_epochs, lr=1e-3, device=device,
                task="classification", verbose=True, print_every=5,
            )
            best_acc = max(h["test_metrics"])
            results.append({
                "dataset": dataset_name,
                "model": name,
                "seed": seed,
                "accuracy": best_acc,
                "params": model.count_parameters(),
            })
            print(f"  -> Best accuracy: {best_acc:.4f}\n")

    return results


if __name__ == "__main__":
    all_results = []

    for ds in ["MNIST", "FashionMNIST"]:
        print(f"\n{'='*60}")
        print(f"Tier 3: {ds}")
        print(f"{'='*60}")
        all_results.extend(run_tier3(ds, n_seeds=3, n_epochs=20))

    df = pd.DataFrame(all_results)
    summary = df.groupby(["dataset", "model"])["accuracy"].agg(["mean", "std"]).round(4)

    print("\n" + "=" * 60)
    print("TIER 3 RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    param_summary = df.groupby(["dataset", "model"])["params"].first().unstack()
    print("\n--- Parameter Counts ---")
    print(param_summary.to_string())

    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/tier3_mnist.csv", index=False)
    print("\nSaved to experiments/results/tier3_mnist.csv")
