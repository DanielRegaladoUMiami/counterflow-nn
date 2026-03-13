"""
Tier 2: CFNN-D (Distillation) vs CFNN-A vs baselines on UCI datasets.

Tests whether the bidirectional transfer + feed plate + reflux/reboil
architecture provides benefits over the simpler absorption model.

Usage:
    python experiments/tier2_distillation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.network import CounterFlowNetwork
from src.distillation import DistillationNetwork
from src.diagnostics import print_diagnostics
from src.utils import (
    load_synthetic_dataset, load_uci_dataset, prepare_data, train_model
)


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


def build_models_tier2(d_in, d_out, d_gas=32, d_liquid=32):
    """Build CFNN-A, CFNN-D, and MLP with comparable parameters."""
    cfnn_a = CounterFlowNetwork(
        d_in=d_in, d_gas=d_gas, d_liquid=d_liquid,
        n_plates=5, d_out=d_out, n_sweeps=2,
    )
    cfnn_d = DistillationNetwork(
        d_in=d_in, d_gas=d_gas, d_liquid=d_liquid,
        n_plates_rect=3, n_plates_strip=2,
        d_out=d_out, n_sweeps=2,
        reflux_ratio=0.3, reboil_ratio=0.2,
    )
    # MLP matched to CFNN-D param count
    target = cfnn_d.count_parameters()
    d_h = max(16, int(np.sqrt(target / 4)))
    mlp = MLPBaseline(d_in, d_h, d_out, n_layers=4)

    return {
        "CFNN-A": cfnn_a,
        "CFNN-D": cfnn_d,
        "MLP": mlp,
    }


def run_tier2(dataset_name, dataset_type="uci", n_seeds=5, n_epochs=150):
    """Run Tier 2 comparison on one dataset."""
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
        models = build_models_tier2(d_in, d_out)

        for name, model in models.items():
            h = train_model(
                model, train_ld, test_ld,
                n_epochs=n_epochs, lr=1e-3, device=device,
                task=task, verbose=False,
            )
            metric = max(h["test_metrics"]) if task == "classification" else min(h["test_metrics"])
            results.append({
                "dataset": dataset_name,
                "model": name,
                "seed": seed,
                "metric": metric,
                "task": task,
                "params": model.count_parameters(),
            })

            # Print diagnostics for first seed only
            if seed == 0 and name.startswith("CFNN"):
                model.to("cpu")
                sample_x = torch.FloatTensor(
                    X[:32] if not isinstance(X, torch.Tensor) else X[:32]
                )
                from sklearn.preprocessing import StandardScaler
                sample_x = torch.FloatTensor(StandardScaler().fit_transform(sample_x.numpy()))
                print_diagnostics(model, sample_x, model_name=f"{name} on {dataset_name}")

    return results


if __name__ == "__main__":
    all_results = []

    # Synthetic datasets (where CFNN-A already works well — test CFNN-D too)
    for ds in ["moons", "circles", "xor"]:
        print(f"\n{'='*60}")
        print(f"Running Tier 2: {ds}")
        print(f"{'='*60}")
        all_results.extend(run_tier2(ds, "synthetic"))

    # UCI datasets
    for ds in ["iris", "wine", "breast_cancer"]:
        print(f"\n{'='*60}")
        print(f"Running Tier 2: {ds}")
        print(f"{'='*60}")
        all_results.extend(run_tier2(ds, "uci"))

    # California Housing (regression)
    print(f"\n{'='*60}")
    print(f"Running Tier 2: california_housing")
    print(f"{'='*60}")
    all_results.extend(run_tier2("california_housing", "uci", n_epochs=200))

    df = pd.DataFrame(all_results)
    summary = df.groupby(["dataset", "model"])["metric"].agg(["mean", "std"]).round(4)

    print("\n" + "=" * 60)
    print("TIER 2 RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # Parameter count comparison
    param_summary = df.groupby(["dataset", "model"])["params"].first().unstack()
    print("\n--- Parameter Counts ---")
    print(param_summary.to_string())

    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/tier2_distillation.csv", index=False)
    print("\nSaved to experiments/results/tier2_distillation.csv")
