#include "blade/nn/pooling.h"
#include "blade/ops.h"

namespace blade {
namespace nn {

// Default stride = kernel_size when 0 is passed.
MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride, size_t padding)
    : ksize_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),
      padding_(padding) {}

Tensor MaxPool2d::forward(const Tensor& input) {
    return ops::max_pool2d(input, (int)ksize_, (int)stride_, (int)padding_);
}

AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride, size_t padding)
    : ksize_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),
      padding_(padding) {}

Tensor AvgPool2d::forward(const Tensor& input) {
    return ops::avg_pool2d(input, (int)ksize_, (int)stride_, (int)padding_);
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(size_t output_size)
    : out_h_(output_size), out_w_(output_size) {}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(size_t out_h, size_t out_w)
    : out_h_(out_h), out_w_(out_w) {}

Tensor AdaptiveAvgPool2d::forward(const Tensor& input) {
    // TODO: compute adaptive stride/kernel per spatial dimension
    (void)input;
    return Tensor({1});
}

} // namespace nn
} // namespace blade
