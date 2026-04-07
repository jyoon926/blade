"""Simple MLP trained on MNIST using blade.

Usage:
    # 1. Download data (once)
    python examples/download_mnist.py

    # 2. Train
    python examples/train_mnist_mlp.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blade
import blade.nn as nn
import blade.optim as optim
import blade.data as data


# ---- Model ------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # Register submodules so parameters() recurses into them
        self.register_module("fc1", self.fc1)
        self.register_module("fc2", self.fc2)
        self.register_module("fc3", self.fc3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ---- Helpers ----------------------------------------------------------------

def count_correct(logits, labels):
    """Count correctly predicted samples in a batch."""
    preds = blade.ops.argmax(logits, 1)
    n = len(labels.storage())
    return sum(
        int(preds.storage()[i]) == int(labels.storage()[i])
        for i in range(n)
    )


# ---- Training ---------------------------------------------------------------

def train(mnist_root="data/mnist", epochs=5, batch_size=64, lr=1e-3):
    train_ds = data.MNIST(mnist_root, data.MNISTSplit.Train)
    test_ds  = data.MNIST(mnist_root, data.MNISTSplit.Test)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=256,        shuffle=False)

    model     = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for inputs, labels in train_loader:
            logits = model(inputs)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        # Evaluation
        model.eval()
        correct, total = 0, 0
        for inputs, labels in test_loader:
            logits  = model(inputs)
            correct += count_correct(logits, labels)
            total   += len(labels.storage())

        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  test_acc={correct/total:.4f}")


if __name__ == "__main__":
    train()
