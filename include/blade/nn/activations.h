#pragma once
#include "blade/nn/module.h"

namespace blade {
namespace nn {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

class LeakyReLU : public Module {
public:
    explicit LeakyReLU(float neg_slope = 0.01f);
    Tensor forward(const Tensor& input) override;
private:
    float neg_slope_;
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

// Applied along `dim`.
class Softmax : public Module {
public:
    explicit Softmax(int dim = -1);
    Tensor forward(const Tensor& input) override;
private:
    int dim_;
};

class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int dim = -1);
    Tensor forward(const Tensor& input) override;
private:
    int dim_;
};

} // namespace nn
} // namespace blade
