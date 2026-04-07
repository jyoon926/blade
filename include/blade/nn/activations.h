#pragma once
#include "blade/nn/module.h"
#include "blade/ops.h"

namespace blade {
namespace nn {

struct ReLU : Module {
    Tensor forward(const Tensor& x) override { return ops::relu(x); }
};

class LeakyReLU : public Module {
public:
    explicit LeakyReLU(float neg_slope = 0.01f) : neg_slope_(neg_slope) {}
    Tensor forward(const Tensor& x) override { return ops::leaky_relu(x, neg_slope_); }
private:
    float neg_slope_;
};

struct Sigmoid : Module {
    Tensor forward(const Tensor& x) override { return ops::sigmoid(x); }
};

struct Tanh : Module {
    Tensor forward(const Tensor& x) override { return ops::tanh(x); }
};

class Softmax : public Module {
public:
    explicit Softmax(int dim = -1) : dim_(dim) {}
    Tensor forward(const Tensor& x) override { return ops::softmax(x, dim_); }
private:
    int dim_;
};

class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int dim = -1) : dim_(dim) {}
    Tensor forward(const Tensor& x) override { return ops::log_softmax(x, dim_); }
private:
    int dim_;
};

} // namespace nn
} // namespace blade
