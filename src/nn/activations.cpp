#include "blade/nn/activations.h"
#include "blade/ops.h"

namespace blade {
namespace nn {

Tensor ReLU::forward(const Tensor& input)          { return ops::relu(input); }

LeakyReLU::LeakyReLU(float neg_slope) : neg_slope_(neg_slope) {}
Tensor LeakyReLU::forward(const Tensor& input)     { return ops::leaky_relu(input, neg_slope_); }

Tensor Sigmoid::forward(const Tensor& input)       { return ops::sigmoid(input); }
Tensor Tanh::forward(const Tensor& input)          { return ops::tanh(input); }

Softmax::Softmax(int dim) : dim_(dim) {}
Tensor Softmax::forward(const Tensor& input)       { return ops::softmax(input, dim_); }

LogSoftmax::LogSoftmax(int dim) : dim_(dim) {}
Tensor LogSoftmax::forward(const Tensor& input)    { return ops::log_softmax(input, dim_); }

} // namespace nn
} // namespace blade
