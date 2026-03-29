#include "blade/optim/sgd.h"
#include <omp.h>
#include <stdexcept>

#define NUM_THREADS 8

namespace blade {
namespace optim {

SGD::SGD(std::vector<Tensor*> params, float lr,
         float momentum, float weight_decay, bool nesterov)
    : Optimizer(std::move(params)),
      lr_(lr), momentum_(momentum),
      weight_decay_(weight_decay), nesterov_(nesterov) {
    velocity_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i)
        velocity_[i].assign(params_[i]->numel(), 0.f);
}

void SGD::step() {
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        Tensor* p = params_[pi];
        if (!p->has_grad()) continue;

        float* w  = p->data_ptr();
        const float* g = p->grad().data_ptr();
        float* v  = velocity_[pi].data();
        const size_t n = p->numel();

        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            float grad = g[i] + weight_decay_ * w[i];
            v[i] = momentum_ * v[i] + grad;
            float update = nesterov_ ? (grad + momentum_ * v[i]) : v[i];
            w[i] -= lr_ * update;
        }
    }
    ++step_count_;
}

} // namespace optim
} // namespace blade
