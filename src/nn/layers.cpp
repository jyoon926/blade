#include "blade/nn/layers.h"
#include "blade/ops.h"
#include <cmath>

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

// ---- Flatten ----------------------------------------------------------------

Flatten::Flatten(int start_dim, int end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

Tensor Flatten::forward(const Tensor& input) {
    return ops::flatten(input, start_dim_, end_dim_);
}

// ---- Sequential -------------------------------------------------------------

void Sequential::add(std::shared_ptr<Module> mod) {
    register_module(std::to_string(count_++), mod);
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& [name, mod] : named_children())
        x = mod->forward(x);
    return x;
}

} // namespace nn
} // namespace blade
