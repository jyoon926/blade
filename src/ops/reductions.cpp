#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

// Helper: compute outer/inner dimensions around a reduction dim.
static void reduction_dims(const Tensor& a, int dim,
                           size_t& outer, size_t& inner, size_t& reduce_size,
                           std::vector<size_t>& out_shape, bool keepdim) {
    const int ndim = (int)a.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error("dim " + std::to_string(dim) +
            " out of range for " + std::to_string(ndim) + "-D tensor");
    const auto& sh = a.shape();
    reduce_size = sh[dim];
    outer = 1; inner = 1;
    for (int i = 0;       i < dim;  ++i) outer *= sh[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= sh[i];
    out_shape.reserve(keepdim ? ndim : ndim - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else          { out_shape.push_back(sh[i]); }
    }
    if (out_shape.empty()) out_shape.push_back(1);
}

// sum

Tensor sum(const Tensor& a) {
    Tensor out({1});
    float s = 0.f;
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static) reduction(+:s)
    for (size_t i = 0; i < n; ++i) s += pa[i];
    out.data_ptr()[0] = s;
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("SumBackward",
            [sa](const Tensor& g) -> std::vector<Tensor> {
                return {Tensor::full(sa->shape(), g.data_ptr()[0])};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    size_t outer, inner, reduce_size;
    std::vector<size_t> out_shape;
    reduction_dims(a, dim, outer, inner, reduce_size, out_shape, keepdim);

    Tensor out = Tensor::zeros(out_shape);
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t oi = 0; oi < outer; ++oi)
        for (size_t ii = 0; ii < inner; ++ii) {
            float s = 0.f;
            const size_t base = oi * reduce_size * inner + ii;
            for (size_t k = 0; k < reduce_size; ++k) s += pa[base + k * inner];
            po[oi * inner + ii] = s;
        }
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("SumDimBackward",
            [sa, outer, inner, reduce_size](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga = Tensor::zeros(sa->shape());
                const float* pg = g.data_ptr(); float* pga = ga.data_ptr();
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
                for (size_t oi = 0; oi < outer; ++oi)
                    for (size_t ii = 0; ii < inner; ++ii) {
                        const float gv = pg[oi * inner + ii];
                        const size_t base = oi * reduce_size * inner + ii;
                        for (size_t k = 0; k < reduce_size; ++k) pga[base + k * inner] += gv;
                    }
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor mean(const Tensor& a)                  { return mul(sum(a), 1.f / (float)a.numel()); }
Tensor mean(const Tensor& a, int dim, bool kd) {
    const int ndim = (int)a.ndim();
    int d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim) throw std::runtime_error("mean: dim out of range");
    return mul(sum(a, dim, kd), 1.f / (float)a.shape()[d]);
}

// max

Tensor max(const Tensor& a) {
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    size_t best_i = 0; float best = pa[0];
    for (size_t i = 1; i < n; ++i)
        if (pa[i] > best) { best = pa[i]; best_i = i; }
    Tensor out = Tensor::full({1}, best);
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("MaxBackward",
            [sa, best_i](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga = Tensor::zeros(sa->shape());
                ga.data_ptr()[best_i] = g.data_ptr()[0];
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor max(const Tensor& a, int dim, bool keepdim) {
    size_t outer, inner, reduce_size;
    std::vector<size_t> out_shape;
    reduction_dims(a, dim, outer, inner, reduce_size, out_shape, keepdim);

    Tensor out = Tensor::zeros(out_shape);
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    const size_t out_n = outer * inner;
    auto idx = std::make_shared<std::vector<size_t>>(out_n);

    for (size_t oi = 0; oi < outer; ++oi)
        for (size_t ii = 0; ii < inner; ++ii) {
            const size_t out_i = oi * inner + ii;
            const size_t base  = oi * reduce_size * inner + ii;
            float best = pa[base]; size_t best_k = 0;
            for (size_t k = 1; k < reduce_size; ++k) {
                float v = pa[base + k * inner];
                if (v > best) {
                    best = v;
                    best_k = k;
                }
            }
            po[out_i] = best;
            (*idx)[out_i] = base + best_k * inner;
        }
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("MaxDimBackward",
            [sa, idx](const Tensor& g) -> std::vector<Tensor> {
                Tensor ga = Tensor::zeros(sa->shape());
                const float* pg = g.data_ptr(); float* pga = ga.data_ptr();
                for (size_t i = 0; i < idx->size(); ++i)
                    pga[(*idx)[i]] += pg[i];
                return {ga};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

// no grad

Tensor argmax(const Tensor& a, int dim) {
    int ndim = (int)a.ndim();
    if (dim < 0) dim += ndim;
    const size_t reduce_size = a.shape()[dim];
    size_t outer = 1, inner = 1;
    for (int i = 0; i < dim;  ++i) outer *= a.shape()[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= a.shape()[i];
    std::vector<size_t> out_shape;
    for (int i = 0; i < ndim; ++i)
        if (i != dim) out_shape.push_back(a.shape()[i]);
    Tensor out = Tensor::zeros(out_shape);
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    for (size_t oi = 0; oi < outer; ++oi)
        for (size_t ii = 0; ii < inner; ++ii) {
            const size_t base = oi * reduce_size * inner + ii;
            float best = pa[base]; size_t best_k = 0;
            for (size_t k = 1; k < reduce_size; ++k) {
                float v = pa[base + k * inner];
                if (v > best) { best = v; best_k = k; }
            }
            po[oi * inner + ii] = (float)best_k;
        }
    return out;
}

Tensor eq(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("eq: shape mismatch");

    Tensor out(a.shape());

    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* po = out.data_ptr();

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        po[i] = (pa[i] == pb[i]) ? 1.f : 0.f;
    }

    return out;
}

} // namespace ops
} // namespace blade
