#include "blade/nn/loss.h"
#include "blade/ops.h"
#include "blade/node.h"
#include <stdexcept>
#include <memory>

namespace blade {
static std::shared_ptr<Tensor> share(const Tensor& t) {
    return std::make_shared<Tensor>(t);
}
} // namespace blade

namespace blade {
namespace nn {

Tensor cross_entropy(const Tensor& input, const Tensor& target) {
    // input: (N, C) logits   target: (N,) integer class indices
    // = NLLLoss(log_softmax(input, dim=1), target)
    Tensor log_probs = ops::log_softmax(input, 1);
    return nll_loss(log_probs, target);
}

Tensor nll_loss(const Tensor& input, const Tensor& target) {
    // input: (N, C) log-probs   target: (N,) class indices
    // loss = -mean(input[i, target[i]])
    if (input.ndim() != 2)
        throw std::runtime_error("nll_loss: input must be 2D (N, C)");
    const size_t N = input.shape()[0];
    const size_t C = input.shape()[1];
    const float* pi = input.data_ptr();
    const float* pt = target.data_ptr();

    auto sel = std::make_shared<std::vector<size_t>>(N);
    float sum = 0.f;
    for (size_t i = 0; i < N; ++i) {
        size_t cls = (size_t)pt[i];
        (*sel)[i] = i * C + cls;
        sum += pi[i * C + cls];
    }
    Tensor out = Tensor::full({1}, -sum / (float)N);

    if (input.requires_grad()) {
        out.set_requires_grad(true);
        auto si = blade::share(input);
        auto fn = [si, sel, N](const Tensor& g) -> std::vector<Tensor> {
            Tensor gi = Tensor::zeros(si->shape());
            float* pgi = gi.data_ptr();
            float gval = -g.data_ptr()[0] / (float)N;
            for (size_t i = 0; i < N; ++i)
                pgi[(*sel)[i]] = gval;
            return {gi};
        };
        out.set_grad_fn(std::make_shared<Node>("NLLLossBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{si}));
    }
    return out;
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
