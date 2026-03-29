#include "blade/nn/loss.h"
#include "blade/ops.h"
#include <stdexcept>

namespace blade {
namespace nn {

Tensor cross_entropy(const Tensor& input, const Tensor& target) {
    // input: (N, C) logits   target: (N,) integer class indices
    // = NLLLoss(log_softmax(input, dim=1), target)
    Tensor log_probs = ops::log_softmax(input, 1);
    return nll_loss(log_probs, target);
}

Tensor nll_loss(const Tensor& input, const Tensor& target) {
    // TODO: gather log-probs at target indices; return -mean
    (void)input; (void)target;
    return Tensor({1});
}

Tensor mse_loss(const Tensor& input, const Tensor& target) {
    if (input.shape() != target.shape())
        throw std::runtime_error("mse_loss: shape mismatch");
    Tensor diff = ops::sub(input, target);
    Tensor sq   = ops::mul(diff, diff);
    return ops::mean(sq);
}

Tensor binary_cross_entropy(const Tensor& input, const Tensor& target) {
    // -[t * log(p) + (1-t) * log(1-p)]
    // TODO: numerically stable version
    (void)input; (void)target;
    return Tensor({1});
}

// ---- Module wrappers --------------------------------------------------------

Tensor CrossEntropyLoss::operator()(const Tensor& input, const Tensor& target) const {
    return cross_entropy(input, target);
}

Tensor MSELoss::operator()(const Tensor& input, const Tensor& target) const {
    return mse_loss(input, target);
}

Tensor NLLLoss::operator()(const Tensor& input, const Tensor& target) const {
    return nll_loss(input, target);
}

} // namespace nn
} // namespace blade
