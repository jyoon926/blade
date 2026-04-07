#pragma once
// Internal helpers shared across ops/*.cpp — not part of the public API.
#include "blade/tensor.h"
#include "blade/node.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <vector>

#define NUM_THREADS 8

namespace blade {

inline std::shared_ptr<Tensor> share(const Tensor& t) {
    return std::make_shared<Tensor>(t);
}

inline bool any_requires_grad(std::initializer_list<const Tensor*> ts) {
    for (auto* t : ts)
        if (t && t->requires_grad()) return true;
    return false;
}

// Unary op: backward uses saved *input* values.
// fwd(x) -> output,  bwd(x) -> local derivative (multiplied by incoming grad).
template<typename FwdFn, typename BwdFn>
inline Tensor elementwise_unary(const char* bwd_name, const Tensor& a, FwdFn fwd, BwdFn bwd) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = fwd(pa[i]);
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>(bwd_name,
            [sa, bwd](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga(sa->shape());
                const float* pi = sa->data_ptr();
                const float* pg = g.data_ptr();
                const size_t n  = g.numel();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t i = 0; i < n; ++i) ga.data_ptr()[i] = pg[i] * bwd(pi[i]);
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

// Unary op: backward uses saved *output* values.
// fwd(x) -> output,  bwd(y) -> local derivative where y = fwd(x).
template<typename FwdFn, typename BwdFn>
inline Tensor elementwise_unary_save_out(const char* bwd_name, const Tensor& a, FwdFn fwd, BwdFn bwd) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = fwd(pa[i]);
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa   = share(a);
        auto sout = share(out);
        out.set_grad_fn(std::make_shared<Node>(bwd_name,
            [sout, bwd](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga(sout->shape());
                const float* py = sout->data_ptr();
                const float* pg = g.data_ptr();
                const size_t n  = g.numel();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t i = 0; i < n; ++i) ga.data_ptr()[i] = pg[i] * bwd(py[i]);
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

} // namespace blade
