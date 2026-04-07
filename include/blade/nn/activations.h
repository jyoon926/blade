#pragma once
#include "blade/nn/module.h"
#include "blade/ops.h"

namespace blade {
namespace nn {

struct ReLU : Module {
    Tensor forward(const Tensor& x) override { return ops::relu(x); }
};

struct Sigmoid : Module {
    Tensor forward(const Tensor& x) override { return ops::sigmoid(x); }
};

struct Tanh : Module {
    Tensor forward(const Tensor& x) override { return ops::tanh(x); }
};

} // namespace nn
} // namespace blade
