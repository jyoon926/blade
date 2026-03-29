#include "blade/optim/adam.h"
#include <omp.h>
#include <cmath>

#define NUM_THREADS 8

namespace blade {
namespace optim {

Adam::Adam(std::vector<Tensor*> params, float lr,
           float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        m_[i].assign(params_[i]->numel(), 0.f);
        v_[i].assign(params_[i]->numel(), 0.f);
    }
}

void Adam::step() {
    ++step_count_;
    float t = (float)step_count_;
    // Bias-correction factors
    float bc1 = 1.f - std::pow(beta1_, t);
    float bc2 = 1.f - std::pow(beta2_, t);
    float lr_t = lr_ * std::sqrt(bc2) / bc1;

    for (size_t pi = 0; pi < params_.size(); ++pi) {
        Tensor* p = params_[pi];
        if (!p->has_grad()) continue;

        float* w        = p->data_ptr();
        const float* g  = p->grad().data_ptr();
        float* m        = m_[pi].data();
        float* v        = v_[pi].data();
        const size_t n  = p->numel();

        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            float grad = g[i] + weight_decay_ * w[i];
            m[i] = beta1_ * m[i] + (1.f - beta1_) * grad;
            v[i] = beta2_ * v[i] + (1.f - beta2_) * grad * grad;
            w[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps_);
        }
    }
}

} // namespace optim
} // namespace blade
