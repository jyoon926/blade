#pragma once
#include "blade/optim/optimizer.h"

namespace blade {
namespace optim {

// Adam optimizer (Kingma & Ba, 2015).
// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
// p_t = p_{t-1} - lr * m_t_hat / (sqrt(v_t_hat) + eps)
class Adam : public Optimizer {
public:
    explicit Adam(std::vector<Tensor*> params,
                  float lr = 1e-3f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float eps = 1e-8f,
                  float weight_decay = 0.f);

    void step() override;

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<std::vector<float>> m_;   // first moment
    std::vector<std::vector<float>> v_;   // second moment
};

} // namespace optim
} // namespace blade
