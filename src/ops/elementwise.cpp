#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

// ---- binary arithmetic ------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("add: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        out.set_grad_fn(std::make_shared<Node>("AddBackward",
            [sa, sb](const Tensor& g) -> std::vector<Tensor> { return {g, g}; },
            std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor add(const Tensor& a, float s) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] + s;
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("AddScalarBackward",
            [sa](const Tensor& g) -> std::vector<Tensor> { return {g}; },
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("sub: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] - pb[i];
    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        out.set_grad_fn(std::make_shared<Node>("SubBackward",
            [sa, sb](const Tensor& g) -> std::vector<Tensor> {
                Tensor gb(g.shape());
                const size_t n = g.numel();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t i = 0; i < n; ++i) gb.data_ptr()[i] = -g.data_ptr()[i];
                return {g, gb};
            }, std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor sub(const Tensor& a, float s) { return add(a, -s); }

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("mul: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] * pb[i];
    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        out.set_grad_fn(std::make_shared<Node>("MulBackward",
            [sa, sb](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga(sa->shape()), gb(sb->shape());
                const size_t n = g.numel();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    ga.data_ptr()[i] = g.data_ptr()[i] * sb->data_ptr()[i];
                    gb.data_ptr()[i] = g.data_ptr()[i] * sa->data_ptr()[i];
                }
                return {ga, gb};
            }, std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor mul(const Tensor& a, float s) {
    return elementwise_unary("MulScalarBackward", a,
        [s](float x) { return x * s; },
        [s](float)   { return s; });
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("div: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] / pb[i];
    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        out.set_grad_fn(std::make_shared<Node>("DivBackward",
            [sa, sb](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga(sa->shape()), gb(sb->shape());
                const size_t n = g.numel();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    float bv = sb->data_ptr()[i];
                    ga.data_ptr()[i] =  g.data_ptr()[i] / bv;
                    gb.data_ptr()[i] = -g.data_ptr()[i] * sa->data_ptr()[i] / (bv * bv);
                }
                return {ga, gb};
            }, std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor div(const Tensor& a, float s) { return mul(a, 1.f / s); }
Tensor neg(const Tensor& a)          { return mul(a, -1.f); }

// ---- unary (use helpers from internal.h) ------------------------------------

Tensor pow(const Tensor& a, float e) {
    return elementwise_unary("PowBackward", a,
        [e](float x) { return std::pow(x, e); },
        [e](float x) { return e * std::pow(x, e - 1.f); });
}

Tensor sqrt(const Tensor& a) { return pow(a, 0.5f); }

Tensor exp(const Tensor& a) {
    return elementwise_unary_save_out("ExpBackward", a,
        [](float x) { return std::exp(x); },
        [](float y) { return y; });              // d exp(x)/dx = exp(x) = y
}

Tensor log(const Tensor& a) {
    return elementwise_unary("LogBackward", a,
        [](float x) { return std::log(x); },
        [](float x) { return 1.f / x; });
}

Tensor abs(const Tensor& a) {
    return elementwise_unary("AbsBackward", a,
        [](float x) { return std::abs(x); },
        [](float x) { return x >= 0.f ? 1.f : -1.f; });
}

Tensor clamp(const Tensor& a, float lo, float hi) {
    return elementwise_unary("ClampBackward", a,
        [lo, hi](float x) { return std::min(std::max(x, lo), hi); },
        [lo, hi](float x) { return (x > lo && x < hi) ? 1.f : 0.f; });
}

} // namespace ops
} // namespace blade
