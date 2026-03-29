#pragma once
#include "blade/optim/optimizer.h"

namespace blade {
namespace optim {

// SGD with optional momentum and weight decay.
// v_{t+1} = mu * v_t + grad
// p_{t+1} = p_t - lr * v_{t+1}
class SGD : public Optimizer {
public:
    explicit SGD(std::vector<Tensor*> params,
                 float lr,
                 float momentum = 0.f,
                 float weight_decay = 0.f,
                 bool nesterov = false);

    void step() override;

private:
    float lr_, momentum_, weight_decay_;
    bool nesterov_;
    std::vector<std::vector<float>> velocity_;  // momentum buffers, one per param
};

} // namespace optim
} // namespace blade
