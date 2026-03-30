# BLADE

Backpropagation Library for Automatic Differentiation.

A mini deep learning framework in C++ with a Python binding via pybind11.
Built for the course Machine Learning Systems Implementation (CSCI 637).

## Features

- n-dimensional float32 tensors with OpenMP-parallelized ops
- reverse-mode automatic differentiation (define-by-run)
- neural network modules: Linear, Conv2d, BatchNorm2d, Dropout, pooling
- optimizers: SGD (momentum, Nesterov) and Adam
- MNIST data loader

## Build

```bash
./compile.sh
```

Requires CMake, a C++17 compiler, OpenMP, and pybind11 (`pip install pybind11`).

## Usage

```python
import blade
import blade.nn as nn
import blade.optim as optim

x = blade.Tensor.randn([32, 784], requires_grad=True)

model = nn.Linear(784, 10)
opt = optim.Adam(model.parameters(), lr=1e-3)

out = model(x)
loss = nn.cross_entropy(out, labels)
loss.backward()
opt.step()
```

See `examples/mnist_cnn.py` for a full CNN training example.
