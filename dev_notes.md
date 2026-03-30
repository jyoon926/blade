# BLADE dev notes

## implemented

- Tensor class: flat float32 storage, row-major strides, static factories (zeros, ones, randn, uniform, arange)
- element-wise ops with autograd: add, sub, mul, div, neg, pow, exp, log, abs, clamp
- global sum and mean with autograd
- matmul (2D and batched) with sparse skip optimization from hw3_cpu
- activations: relu, leaky_relu, sigmoid, tanh (softmax/log_softmax are stubs, need dim-wise sum first)
- autograd Node with closure-based BackwardFn, run_backward with DFS topo sort
- nn::Module base: register_parameter/register_module, parameters() recursion, train/eval propagation
- layers: Linear, Conv2d (forward stub), BatchNorm2d (forward stub), Dropout, Flatten
- Kaiming uniform init for Linear and Conv2d
- loss: mse_loss implemented, cross_entropy/nll_loss are stubs (need gather + log_softmax)
- pooling: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d (all stubs)
- SGD with momentum, weight decay, Nesterov
- Adam with bias-corrected moment estimates
- Dataset base class and DataLoader with shuffle (collate is a stub)
- MNIST IDX binary reader, normalizes pixels to [0, 1]
- pybind11 bindings for all modules, PyModule trampoline for Python subclassing

## bugs fixed

- layers.cpp: `bias` parameter shadowing `this->bias` member in Linear/Conv2d constructors
- bindings.cpp: `py::arg("grad") = Tensor()` crashes at import before Tensor is registered, switched to `py::none()` + lambda
- bindings.cpp: MNISTSplit enum used as default arg before being registered, moved enum def above MNIST class
- node.cpp: default Tensor() has empty data_ but numel() returns 1 (empty product), the `numel() == 0` check in run_backward never triggered, fixed by checking `storage().empty()`

## todo

- conv2d forward (im2col -> matmul) and backward
- max_pool2d forward and backward (need to save argmax mask)
- batch_norm per-channel mean/var and backward
- dim-wise sum/mean (needed for softmax, cross_entropy, loss reductions)
- softmax and log_softmax (numerically stable)
- cross_entropy and nll_loss (need a gather op)
- DataLoader::collate (stack samples into a batch tensor)
- shared_ptr Tensor refactor so gradient identity is correct for non-leaf tensors
- Module save/load
- MNIST download support
