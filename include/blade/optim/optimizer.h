#pragma once
#include "blade/tensor.h"
#include <vector>

namespace blade {
namespace optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> params)
        : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() { for (auto* p : params_) p->zero_grad(); }

protected:
    std::vector<Tensor*> params_;
    size_t step_count_ = 0;
};

} // namespace optim
} // namespace blade
