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

// 2-D convolution: (N, C_in, H, W) -> (N, C_out, H', W')
class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels,
           size_t kernel_size,
           size_t stride = 1, size_t padding = 0, bool bias = true);
    Tensor forward(const Tensor& input) override;

    Tensor weight;  // (out_channels, in_channels, kernel_size, kernel_size)
    Tensor bias;    // (out_channels,)

private:
    size_t in_ch_, out_ch_, ksize_, stride_, padding_;
    bool use_bias_;
    void reset_parameters();
};

// Batch normalisation over the channel dimension.
class BatchNorm2d : public Module {
public:
    explicit BatchNorm2d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);
    Tensor forward(const Tensor& input) override;

    Tensor weight;        // scale (gamma)
    Tensor bias;          // shift (beta)
    Tensor running_mean;
    Tensor running_var;

private:
    size_t num_features_;
    float eps_, momentum_;
};

// Randomly zeros elements with probability p during training.
class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5f);
    Tensor forward(const Tensor& input) override;

private:
    float p_;
};

// Flattens all dimensions from start_dim to end_dim into one.
class Flatten : public Module {
public:
    explicit Flatten(int start_dim = 1, int end_dim = -1);
    Tensor forward(const Tensor& input) override;

private:
    int start_dim_, end_dim_;
};

} // namespace nn
} // namespace blade
