#pragma once
#include "blade/nn/module.h"

namespace blade {
namespace nn {

// (N, C, H, W) -> (N, C, H', W')
class MaxPool2d : public Module {
public:
    explicit MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    Tensor forward(const Tensor& input) override;
private:
    size_t ksize_, stride_, padding_;
};

class AvgPool2d : public Module {
public:
    explicit AvgPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    Tensor forward(const Tensor& input) override;
private:
    size_t ksize_, stride_, padding_;
};

// Adaptive pooling reduces spatial dims to a fixed output size.
class AdaptiveAvgPool2d : public Module {
public:
    explicit AdaptiveAvgPool2d(size_t output_size);     // square output
    AdaptiveAvgPool2d(size_t out_h, size_t out_w);
    Tensor forward(const Tensor& input) override;
private:
    size_t out_h_, out_w_;
};

} // namespace nn
} // namespace blade
