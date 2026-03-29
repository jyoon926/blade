#include "blade/nn/layers.h"
#include "blade/ops.h"
#include <cmath>
#include <stdexcept>

namespace blade {
namespace nn {

// ---- Linear -----------------------------------------------------------------

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : use_bias_(bias) {
    weight = Tensor({out_features, in_features});
    if (use_bias_) this->bias = Tensor({out_features});
    register_parameter("weight", weight);
    if (use_bias_) register_parameter("bias", this->bias);
    reset_parameters();
}

void Linear::reset_parameters() {
    // Kaiming uniform initialisation (same as PyTorch default)
    float fan_in = (float)weight.shape()[1];
    float bound = std::sqrt(1.f / fan_in);
    weight = Tensor::uniform(weight.shape(), -bound, bound);
    weight.set_requires_grad(true);
    if (use_bias_) {
        bias = Tensor::uniform(bias.shape(), -bound, bound);
        bias.set_requires_grad(true);
    }
}

Tensor Linear::forward(const Tensor& input) {
    return ops::linear(input, weight, bias);
}

// ---- Conv2d -----------------------------------------------------------------

Conv2d::Conv2d(size_t in_channels, size_t out_channels,
               size_t kernel_size, size_t stride, size_t padding, bool bias)
    : in_ch_(in_channels), out_ch_(out_channels),
      ksize_(kernel_size), stride_(stride), padding_(padding), use_bias_(bias) {
    weight = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    if (use_bias_) this->bias = Tensor({out_channels});
    register_parameter("weight", weight);
    if (use_bias_) register_parameter("bias", this->bias);
    reset_parameters();
}

void Conv2d::reset_parameters() {
    // Kaiming uniform (fan_in = in_ch * kH * kW)
    float fan_in = (float)(in_ch_ * ksize_ * ksize_);
    float bound = std::sqrt(1.f / fan_in);
    weight = Tensor::uniform(weight.shape(), -bound, bound);
    weight.set_requires_grad(true);
    if (use_bias_) {
        bias = Tensor::uniform(bias.shape(), -bound, bound);
        bias.set_requires_grad(true);
    }
}

Tensor Conv2d::forward(const Tensor& input) {
    return ops::conv2d(input, weight, bias, (int)stride_, (int)padding_);
}

// ---- BatchNorm2d ------------------------------------------------------------

BatchNorm2d::BatchNorm2d(size_t num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    weight      = Tensor::ones({num_features});
    bias        = Tensor::zeros({num_features});
    running_mean = Tensor::zeros({num_features});
    running_var  = Tensor::ones({num_features});
    register_parameter("weight", weight);
    register_parameter("bias", bias);
}

Tensor BatchNorm2d::forward(const Tensor& input) {
    return ops::batch_norm(input, weight, bias, running_mean, running_var,
                           training_, momentum_, eps_);
}

// ---- Dropout ----------------------------------------------------------------

Dropout::Dropout(float p) : p_(p) {}

Tensor Dropout::forward(const Tensor& input) {
    return ops::dropout(input, p_, training_);
}

// ---- Flatten ----------------------------------------------------------------

Flatten::Flatten(int start_dim, int end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

Tensor Flatten::forward(const Tensor& input) {
    return ops::flatten(input, start_dim_, end_dim_);
}

} // namespace nn
} // namespace blade
