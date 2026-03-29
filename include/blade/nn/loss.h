#pragma once
#include "blade/tensor.h"

namespace blade {
namespace nn {

// Cross-entropy loss combining LogSoftmax + NLLLoss.
// input: (N, C) logits    target: (N,) class indices
Tensor cross_entropy(const Tensor& input, const Tensor& target);

// Negative log-likelihood loss.
// input: (N, C) log-probabilities    target: (N,) class indices
Tensor nll_loss(const Tensor& input, const Tensor& target);

// Mean-squared-error loss.
Tensor mse_loss(const Tensor& input, const Tensor& target);

// Binary cross-entropy (input must be probabilities in (0,1)).
Tensor binary_cross_entropy(const Tensor& input, const Tensor& target);

// Module wrappers for use in Sequential-style networks.

class CrossEntropyLoss {
public:
    Tensor operator()(const Tensor& input, const Tensor& target) const;
};

class MSELoss {
public:
    Tensor operator()(const Tensor& input, const Tensor& target) const;
};

class NLLLoss {
public:
    Tensor operator()(const Tensor& input, const Tensor& target) const;
};

} // namespace nn
} // namespace blade
