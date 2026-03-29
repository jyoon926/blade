#include "blade/optim/optimizer.h"

namespace blade {
namespace optim {

Optimizer::Optimizer(std::vector<Tensor*> params)
    : params_(std::move(params)) {}

void Optimizer::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

} // namespace optim
} // namespace blade
