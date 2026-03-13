"""
Head-to-head comparison: CFNN-A vs MLP vs ResNet-MLP on all datasets.

Usage:
    python experiments/compare_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.network import CounterFlowNetwork
from src.utils import load_synthetic_dataset, load_uci_dataset, prepare_data, train_model


class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=4):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResMLPBaseline(nn.Module):
    """MLP with skip connections every 2 layers."""
    def __init__(self, d_in, d_hidden, d_out, n_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList()
        for _ in range((n_layers - 2) // 2):
            self.blocks.append(nn.Sequential(
                nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
            ))
        self.output = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = torch.relu(self.input_proj(x))
        for block in self.blocks:
            h = torch.relu(h + block(h))
        return self.output(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_models(d_in, d_out, d_gas=32, d_liquid=32, n_plates=5):
    """Build all models with roughly matched parameters."""
    cfnn = CounterFlowNetwork(d_in=d_in, d_gas=d_gas, d_liquid=d_liquid, n_plates=n_plates, d_out=d_out)
    target_params = cfnn.count_parameters()
    # Approximate hidden size for MLP to match param count
    d_h = max(16, int(np.sqrt(target_params / 4)))
    mlp = MLPBaseline(d_in, d_h, d_out, n_layers=4)
    resmlp = ResMLPBaseline(d_in, d_h, d_out, n_layers=4)
    return {"CFNN-A": cfnn, "MLP": mlp, "ResMLP": resmlp}


def run_comparison(dataset_name, dataset_type="synthetic", n_seeds=5, n_epochs=100):
    """Compare all models on one dataset."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(n_seeds):
        if dataset_type == "synthetic":
            X, y = load_synthetic_dataset(dataset_name, seed=seed)
            task = "classification"
        else:
            X, y, task = load_uci_dataset(dataset_name)

        train_ld, test_ld, d_in, n_cls = prepare_data(X, y, seed=seed)
        d_out = n_cls if task == "classification" else 1
        models = build_models(d_in, d_out)

        for name, model in models.items():
            h = train_model(model, train_ld, test_ld, n_epochs=n_epochs, lr=1e-3, device=device, task=task, verbose=False)
            metric = max(h["test_metrics"]) if task == "classification" else min(h["test_metrics"])
            results.append({"dataset": dataset_name, "model": name, "seed": seed,
                           "metric": metric, "task": task, "params": model.count_parameters()})

    return results


if __name__ == "__main__":
    all_results = []

    # Synthetic
    for ds in ["moons", "circles", "xor"]:
        print(f"Running {ds}...")
        all_results.extend(run_comparison(ds, "synthetic"))

    # UCI
    for ds in ["iris", "wine", "breast_cancer"]:
        print(f"Running {ds}...")
        all_results.extend(run_comparison(ds, "uci"))

    df = pd.DataFrame(all_results)
    summary = df.groupby(["dataset", "model"])["metric"].agg(["mean", "std"]).round(4)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/baseline_comparison.csv", index=False)
    print("\nSaved to experiments/results/baseline_comparison.csv")
