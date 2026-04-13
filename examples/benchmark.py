"""Benchmark: BLADE vs. PyTorch on MNIST MLP (identical architecture/hyperparams).

Usage:
    python examples/benchmark.py [--mnist-root data/mnist] [--epochs 5] [--batch-size 64] [--lr 1e-3]

Outputs a JSON summary to stdout and a human-readable table to stderr.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BLADE

import blade
import blade.nn   as bnn
import blade.optim as boptim
import blade.data  as bdata


class BladeMLP(bnn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = bnn.Flatten(start_dim=1)
        self.fc1 = bnn.Linear(784, 256)
        self.fc2 = bnn.Linear(256, 128)
        self.fc3 = bnn.Linear(128, 10)
        self.relu = bnn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def blade_count_correct(logits, labels):
    preds = blade.ops.argmax(logits, 1)
    matches = blade.ops.eq(preds, labels)
    return blade.ops.sum(matches).item()


def run_blade(mnist_root, epochs, batch_size, lr):
    train_ds = bdata.MNIST(mnist_root, bdata.MNISTSplit.Train)
    test_ds = bdata.MNIST(mnist_root, bdata.MNISTSplit.Test)
    train_loader = bdata.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = bdata.DataLoader(test_ds,  batch_size=256,        shuffle=False)

    model = BladeMLP()
    optimizer = boptim.Adam(model.parameters(), lr=lr)
    criterion = bnn.CrossEntropyLoss()

    epoch_results = []
    
    # Warm-up (not timed)
    model.train()
    for inputs, labels in train_loader:
        logits = model(inputs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
    
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_start = time.perf_counter()

        for inputs, labels in train_loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        correct, total = 0, 0
        for inputs, labels in test_loader:
            logits = model(inputs)
            correct += blade_count_correct(logits, labels)
            total += labels.shape[0]

        epoch_time = time.perf_counter() - epoch_start
        epoch_results.append({
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "acc": correct / total,
            "time_s": epoch_time,
        })
        print(f"  [blade] epoch {epoch}/{epochs}  "
              f"loss={total_loss/n_batches:.4f}  "
              f"acc={correct/total:.4f}  "
              f"({epoch_time:.1f}s)", flush=True)

    total_time = time.perf_counter() - total_start
    return epoch_results, total_time


# PyTorch

def run_torch(mnist_root, epochs, batch_size, lr):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),  # already [0,1] like BLADE
    ])
    train_ds = datasets.MNIST(mnist_root, train=True,  download=False,
                                transform=transform)
    test_ds  = datasets.MNIST(mnist_root, train=False, download=False,
                               transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_results = []
    
    # Warm-up (not timed)
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        break

    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_start = time.perf_counter()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_time = time.perf_counter() - epoch_start
        epoch_results.append({
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "acc": correct / total,
            "time_s": epoch_time,
        })
        print(f"  [torch] epoch {epoch}/{epochs}  "
              f"loss={total_loss/n_batches:.4f}  "
              f"acc={correct/total:.4f}  "
              f"({epoch_time:.1f}s)", flush=True)

    total_time = time.perf_counter() - total_start
    return epoch_results, total_time


# Check torchvision MNIST data

def ensure_torchvision_data(mnist_root):
    """torchvision MNIST expects: <root>/MNIST/raw/<idx-files>
    Our layout is:               <mnist_root>/<idx-files>
    We symlink <tv_root>/MNIST -> <mnist_root> so both share the same files.
    Returns the torchvision root (the parent of MNIST/)."""
    parent = os.path.dirname(os.path.abspath(mnist_root))
    link = os.path.join(parent, "MNIST")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(mnist_root), link)
    return parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist-root", default="data/mnist")
    ap.add_argument("--epochs",     type=int,   default=5)
    ap.add_argument("--batch-size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("BLADE", flush=True)
    print("=" * 60, flush=True)
    blade_epochs, blade_total = run_blade(args.mnist_root, args.epochs, args.batch_size, args.lr)

    torch_epochs, torch_total = [], 0.0
    if not args.skip_torch:
        try:
            tv_root = ensure_torchvision_data(args.mnist_root)
            print("=" * 60, flush=True)
            print("PyTorch", flush=True)
            print("=" * 60, flush=True)
            torch_epochs, torch_total = run_torch(tv_root, args.epochs, args.batch_size, args.lr)
        except Exception as e:
            print(f"[warn] PyTorch benchmark failed: {e}", file=sys.stderr)

    # summary table
    print("\n" + "=" * 60, flush=True)
    print(f"{'Epoch':>5}  {'BLADE loss':>10}  {'BLADE acc':>9}  "
          f"{'Torch loss':>10}  {'Torch acc':>9}", flush=True)
    print("-" * 60, flush=True)
    for i in range(args.epochs):
        br = blade_epochs[i]
        if torch_epochs:
            tr = torch_epochs[i]
            print(f"{br['epoch']:>5}  {br['loss']:>10.4f}  {br['acc']:>9.4f}  "
                  f"{tr['loss']:>10.4f}  {tr['acc']:>9.4f}", flush=True)
        else:
            print(f"{br['epoch']:>5}  {br['loss']:>10.4f}  {br['acc']:>9.4f}",
                  flush=True)
    print("=" * 60, flush=True)
    if torch_epochs:
        speedup = blade_total / torch_total if torch_total > 0 else float("inf")
        print(f"Total time: BLADE={blade_total:.1f}s  "
              f"PyTorch={torch_total:.1f}s  "
              f"(PyTorch is {speedup:.1f}x faster)", flush=True)
    else:
        print(f"Total time: BLADE={blade_total:.1f}s", flush=True)

    # JSON output
    result = {
        "blade": {"epochs": blade_epochs, "total_time_s": blade_total},
        "torch": {"epochs": torch_epochs, "total_time_s": torch_total},
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Results saved to benchmark_results.json", flush=True)


if __name__ == "__main__":
    main()
