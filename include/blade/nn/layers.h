#pragma once
#include "blade/nn/module.h"

namespace blade {
namespace nn {

// Fully-connected layer: y = x W^T + b
// weight: (out_features, in_features)   bias: (out_features,)
class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Tensor forward(const Tensor& input) override;

    Tensor weight;
    Tensor bias;

private:
    bool use_bias_;
    void reset_parameters();
};

// Flattens all dimensions from start_dim to end_dim into one.
class Flatten : public Module {
public:
    explicit Flatten(int start_dim = 1, int end_dim = -1);
    Tensor forward(const Tensor& input) override;

private:
    int start_dim_, end_dim_;
};

// Applies a sequence of modules in order.
// Usage: Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
class Sequential : public Module {
public:
    Sequential() = default;

    // Append a module to the sequence (registered as "0", "1", ...).
    void add(std::shared_ptr<Module> mod);

    Tensor forward(const Tensor& input) override;

private:
    size_t count_ = 0;
};

} // namespace nn
} // namespace blade
