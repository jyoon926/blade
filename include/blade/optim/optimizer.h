#pragma once
#include "blade/tensor.h"
#include <vector>

namespace blade {
namespace optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> params);
    virtual ~Optimizer() = default;

    // Apply one gradient-descent step.
    virtual void step() = 0;

    // Zero all parameter gradients.
    void zero_grad();

protected:
    std::vector<Tensor*> params_;
    size_t step_count_ = 0;
};

} // namespace optim
} // namespace blade
