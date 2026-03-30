"""
MNIST CNN training example.

Usage:
    cd blade && python examples/mnist_cnn.py --data /path/to/mnist
"""
import sys
import time
import argparse

sys.path.insert(0, ".")
import blade
import blade.nn as nn
import blade.optim as optim
import blade.data as data


# ---- Model ------------------------------------------------------------------

class CNN(nn.Module):
    """LeNet-style CNN for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.drop  = nn.Dropout(0.25)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # -> (N, 32, 28, 28)
        x = self.pool(x)               # -> (N, 32, 14, 14)
        x = self.relu(self.conv2(x))   # -> (N, 64, 14, 14)
        x = self.pool(x)               # -> (N, 64,  7,  7)
        x = self.drop(x)
        x = self.flat(x)               # -> (N, 3136)
        x = self.relu(self.fc1(x))     # -> (N, 128)
        x = self.drop(x)
        return self.fc2(x)             # -> (N, 10)  logits


# ---- Training loop ----------------------------------------------------------

def accuracy(logits, labels):
    # logits: (N, 10)  labels: (N,)
    # TODO: implement argmax once available
    return 0.0


def train(args):
    train_ds = data.MNIST(args.data, data.MNISTSplit.Train)
    test_ds  = data.MNIST(args.data, data.MNISTSplit.Test)

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        elapsed = time.time() - t0

        # Eval
        model.eval()
        correct = total = 0
        for x, y in test_loader:
            logits = model(x)
            # TODO: accumulate accuracy once argmax is available
            total += 1

        print(f"Epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}"
              f"  time={elapsed:.1f}s")


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/mnist",  help="MNIST data directory")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=64,  dest="batch_size")
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()
    train(args)
