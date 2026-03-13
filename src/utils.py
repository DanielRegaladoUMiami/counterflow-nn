"""Training utilities: data loading, training loops, metric computation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, load_iris, load_wine, load_breast_cancer


def load_synthetic_dataset(name="moons", n_samples=1000, noise=0.2, seed=42):
    """Load synthetic 2D dataset: 'moons', 'circles', or 'xor'."""
    rng = np.random.RandomState(seed)
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == "circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    elif name == "xor":
        X = rng.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += noise * rng.randn(n_samples, 2)
        return X, y
    raise ValueError(f"Unknown dataset: {name}")


def load_uci_dataset(name="iris"):
    """Load UCI dataset. Returns (X, y, task_type)."""
    loaders = {
        "iris": (load_iris, "classification"),
        "wine": (load_wine, "classification"),
        "breast_cancer": (load_breast_cancer, "classification"),
    }
    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        return d.data, d.target, "regression"
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    d = loaders[name][0]()
    return d.data, d.target, loaders[name][1]


def prepare_data(X, y, test_size=0.2, batch_size=64, seed=42, scale=True):
    """Split, scale, and create DataLoaders. Returns (train_loader, test_loader, d_in, n_classes)."""
    stratify = y if len(np.unique(y)) < 20 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    if scale:
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    X_tr_t, X_te_t = torch.FloatTensor(X_tr), torch.FloatTensor(X_te)
    n_classes = len(np.unique(y))
    if n_classes > 20:
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(-1)
        y_te_t = torch.FloatTensor(y_te).unsqueeze(-1)
        n_classes = 1
    else:
        y_tr_t, y_te_t = torch.LongTensor(y_tr), torch.LongTensor(y_te)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)
    return train_loader, test_loader, X_tr.shape[1], n_classes


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch. Returns average loss."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="classification"):
    """Evaluate model. Returns dict with 'loss' and 'accuracy' or 'rmse'."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, targs_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item()
        if task == "classification":
            correct += (out.argmax(-1) == yb).sum().item()
            total += yb.shape[0]
        else:
            preds_all.append(out.cpu())
            targs_all.append(yb.cpu())
    res = {"loss": total_loss / max(len(loader), 1)}
    if task == "classification":
        res["accuracy"] = correct / max(total, 1)
    else:
        res["rmse"] = torch.sqrt(((torch.cat(preds_all) - torch.cat(targs_all)) ** 2).mean()).item()
    return res


def train_model(model, train_loader, test_loader, n_epochs=100, lr=1e-3, device=None, task="classification", verbose=True, print_every=10):
    """Full training loop. Returns history dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_losses": [], "test_losses": [], "test_metrics": []}
    for epoch in range(1, n_epochs + 1):
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, criterion, device, task)
        history["train_losses"].append(tl)
        history["test_losses"].append(te["loss"])
        mk = "accuracy" if task == "classification" else "rmse"
        history["test_metrics"].append(te[mk])
        if verbose and epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | Train: {tl:.4f} | Test: {te['loss']:.4f} | {mk}: {te[mk]:.4f}")
    return history
