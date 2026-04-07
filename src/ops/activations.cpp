#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

Tensor relu(const Tensor& a) {
    return elementwise_unary("ReLUBackward", a,
        [](float x) { return x > 0.f ? x : 0.f; },
        [](float x) { return x > 0.f ? 1.f : 0.f; });
}

Tensor leaky_relu(const Tensor& a, float neg_slope) {
    return elementwise_unary("LeakyReLUBackward", a,
        [neg_slope](float x) { return x > 0.f ? x : neg_slope * x; },
        [neg_slope](float x) { return x > 0.f ? 1.f : neg_slope; });
}

Tensor sigmoid(const Tensor& a) {
    return elementwise_unary_save_out("SigmoidBackward", a,
        [](float x) { return 1.f / (1.f + std::exp(-x)); },
        [](float y) { return y * (1.f - y); });
}

Tensor tanh(const Tensor& a) {
    return elementwise_unary_save_out("TanhBackward", a,
        [](float x) { return std::tanh(x); },
        [](float y) { return 1.f - y * y; });
}

// Helper: compute outer/inner dimensions around a reduction dim.
static void softmax_dims(const Tensor& a, int& dim,
                         size_t& outer, size_t& inner, size_t& reduce_size) {
    const int ndim = (int)a.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::runtime_error("softmax: dim out of range");
    const auto& sh = a.shape();
    reduce_size = sh[dim];
    outer = 1; inner = 1;
    for (int i = 0;       i < dim;  ++i) outer *= sh[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= sh[i];
}

Tensor softmax(const Tensor& a, int dim) {
    size_t outer, inner, reduce_size;
    softmax_dims(a, dim, outer, inner, reduce_size);

    Tensor out = Tensor::zeros(a.shape());
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    for (size_t oi = 0; oi < outer; ++oi)
        for (size_t ii = 0; ii < inner; ++ii) {
            const size_t base = oi * reduce_size * inner + ii;
            float mx = pa[base];
            for (size_t k = 1; k < reduce_size; ++k)
                mx = std::max(mx, pa[base + k * inner]);
            float se = 0.f;
            for (size_t k = 0; k < reduce_size; ++k) {
                float e = std::exp(pa[base + k * inner] - mx);
                po[base + k * inner] = e; se += e;
            }
            for (size_t k = 0; k < reduce_size; ++k) po[base + k * inner] /= se;
        }

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a), sout = share(out);
        out.set_grad_fn(std::make_shared<Node>("SoftmaxBackward",
            [sout, outer, inner, reduce_size](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga = Tensor::zeros(sout->shape());
                const float* ps = sout->data_ptr();
                const float* pg = g.data_ptr();
                float* pga = ga.data_ptr();
                for (size_t oi = 0; oi < outer; ++oi)
                    for (size_t ii = 0; ii < inner; ++ii) {
                        const size_t base = oi * reduce_size * inner + ii;
                        float dot = 0.f;
                        for (size_t k = 0; k < reduce_size; ++k)
                            dot += pg[base + k * inner] * ps[base + k * inner];
                        for (size_t k = 0; k < reduce_size; ++k) {
                            size_t idx = base + k * inner;
                            pga[idx] = ps[idx] * (pg[idx] - dot);
                        }
                    }
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor log_softmax(const Tensor& a, int dim) {
    size_t outer, inner, reduce_size;
    softmax_dims(a, dim, outer, inner, reduce_size);

    Tensor out = Tensor::zeros(a.shape());
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    for (size_t oi = 0; oi < outer; ++oi)
        for (size_t ii = 0; ii < inner; ++ii) {
            const size_t base = oi * reduce_size * inner + ii;
            float mx = pa[base];
            for (size_t k = 1; k < reduce_size; ++k)
                mx = std::max(mx, pa[base + k * inner]);
            float se = 0.f;
            for (size_t k = 0; k < reduce_size; ++k)
                se += std::exp(pa[base + k * inner] - mx);
            float log_se = std::log(se);
            for (size_t k = 0; k < reduce_size; ++k) {
                size_t idx = base + k * inner;
                po[idx] = (pa[idx] - mx) - log_se;
            }
        }

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a), sout = share(out);
        out.set_grad_fn(std::make_shared<Node>("LogSoftmaxBackward",
            [sout, outer, inner, reduce_size](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga = Tensor::zeros(sout->shape());
                const float* plsm = sout->data_ptr();
                const float* pg = g.data_ptr();
                float* pga = ga.data_ptr();
                for (size_t oi = 0; oi < outer; ++oi)
                    for (size_t ii = 0; ii < inner; ++ii) {
                        const size_t base = oi * reduce_size * inner + ii;
                        float sg = 0.f;
                        for (size_t k = 0; k < reduce_size; ++k)
                            sg += pg[base + k * inner];
                        for (size_t k = 0; k < reduce_size; ++k) {
                            size_t idx = base + k * inner;
                            pga[idx] = pg[idx] - std::exp(plsm[idx]) * sg;
                        }
                    }
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

} // namespace ops
} // namespace blade
