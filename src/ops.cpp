#include "blade/ops.h"
#include "blade/node.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <random>

#define NUM_THREADS 8

namespace blade {
namespace ops {

// ---- helpers ----------------------------------------------------------------

static bool any_requires_grad(std::initializer_list<const Tensor*> ts) {
    for (auto* t : ts)
        if (t && t->requires_grad()) return true;
    return false;
}

static std::shared_ptr<Tensor> share(const Tensor& t) {
    return std::make_shared<Tensor>(t);
}

// ---- element-wise arithmetic ------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("add: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        auto fn = [sa, sb](const Tensor& g) -> std::vector<Tensor> {
            // grad of add is identity for both inputs
            Tensor ga = Tensor::zeros(sa->shape());
            Tensor gb = Tensor::zeros(sb->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                ga.data_ptr()[i] = g.data_ptr()[i];
                gb.data_ptr()[i] = g.data_ptr()[i];
            }
            return {ga, gb};
        };
        out.set_grad_fn(std::make_shared<Node>("AddBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor add(const Tensor& a, float s) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] + s;

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            return {g};
        };
        out.set_grad_fn(std::make_shared<Node>("AddScalarBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("sub: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] - pb[i];

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        auto fn = [sa, sb](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga = g;
            Tensor gb(g.shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) gb.data_ptr()[i] = -g.data_ptr()[i];
            return {ga, gb};
        };
        out.set_grad_fn(std::make_shared<Node>("SubBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor sub(const Tensor& a, float s) { return add(a, -s); }

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("mul: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] * pb[i];

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        auto fn = [sa, sb](const Tensor& g) -> std::vector<Tensor> {
            // d/da = b, d/db = a
            Tensor ga(sa->shape()), gb(sb->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                ga.data_ptr()[i] = g.data_ptr()[i] * sb->data_ptr()[i];
                gb.data_ptr()[i] = g.data_ptr()[i] * sa->data_ptr()[i];
            }
            return {ga, gb};
        };
        out.set_grad_fn(std::make_shared<Node>("MulBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor mul(const Tensor& a, float s) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] * s;

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, s](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) ga.data_ptr()[i] = g.data_ptr()[i] * s;
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("MulScalarBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("div: shape mismatch");
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); const float* pb = b.data_ptr();
    float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] / pb[i];

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa = share(a), sb = share(b);
        auto fn = [sa, sb](const Tensor& g) -> std::vector<Tensor> {
            // d/da = 1/b,  d/db = -a/b^2
            Tensor ga(sa->shape()), gb(sb->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float bv = sb->data_ptr()[i];
                ga.data_ptr()[i] = g.data_ptr()[i] / bv;
                gb.data_ptr()[i] = -g.data_ptr()[i] * sa->data_ptr()[i] / (bv * bv);
            }
            return {ga, gb};
        };
        out.set_grad_fn(std::make_shared<Node>("DivBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa, sb}));
    }
    return out;
}

Tensor div(const Tensor& a, float s) { return mul(a, 1.f / s); }

Tensor neg(const Tensor& a) { return mul(a, -1.f); }

Tensor pow(const Tensor& a, float exp) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::pow(pa[i], exp);

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, exp](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = g.data_ptr()[i] * exp * std::pow(sa->data_ptr()[i], exp - 1.f);
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("PowBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sqrt(const Tensor& a) { return pow(a, 0.5f); }

Tensor exp(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::exp(pa[i]);

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sout = share(out);
        auto sa = share(a);
        auto fn = [sout](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sout->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = g.data_ptr()[i] * sout->data_ptr()[i];
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("ExpBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor log(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::log(pa[i]);

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = g.data_ptr()[i] / sa->data_ptr()[i];
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("LogBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor abs(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::abs(pa[i]);

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = g.data_ptr()[i] * (sa->data_ptr()[i] >= 0.f ? 1.f : -1.f);
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("AbsBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor clamp(const Tensor& a, float lo, float hi) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::min(std::max(pa[i], lo), hi);
    // grad: pass-through where lo < a < hi, else 0
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, lo, hi](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float v = sa->data_ptr()[i];
                ga.data_ptr()[i] = (v > lo && v < hi) ? g.data_ptr()[i] : 0.f;
            }
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("ClampBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

// ---- reduction --------------------------------------------------------------

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
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            // broadcast scalar gradient
            Tensor ga = Tensor::full(sa->shape(), g.data_ptr()[0]);
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("SumBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sum(const Tensor& a, int dim, bool keepdim) {
    
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error("sum: dim " + std::to_string(dim) + " out of range for tensor with " + std::to_string(ndim) + " dims");
 
    const auto& in_shape = a.shape();
    const size_t reduce_size = in_shape[dim];
 
    size_t outer = 1, inner = 1;
    for (int i = 0;       i < dim;  ++i) outer *= in_shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= in_shape[i];
 
    std::vector<size_t> out_shape;
    out_shape.reserve(keepdim ? ndim : ndim - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else          { out_shape.push_back(in_shape[i]); }
    }
    if (out_shape.empty()) out_shape.push_back(1);  // scalar result
 
    Tensor out = Tensor::zeros(out_shape);
    const float* pa = a.data_ptr();
    float*       po = out.data_ptr();
 
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
        for (size_t inner_i = 0; inner_i < inner; ++inner_i) {
            float s = 0.f;
            const size_t in_base = outer_i * reduce_size * inner + inner_i;
            for (size_t k = 0; k < reduce_size; ++k)
                s += pa[in_base + k * inner];
            po[outer_i * inner + inner_i] = s;
        }
    }
 
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, dim, outer, inner, reduce_size](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga = Tensor::zeros(sa->shape());
            const float* pg  = g.data_ptr();
            float*       pga = ga.data_ptr();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
                for (size_t inner_i = 0; inner_i < inner; ++inner_i) {
                    const float grad_val = pg[outer_i * inner + inner_i];
                    const size_t in_base = outer_i * reduce_size * inner + inner_i;
                    for (size_t k = 0; k < reduce_size; ++k)
                        pga[in_base + k * inner] += grad_val;
                }
            }
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("SumDimBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor s = sum(a);
    return mul(s, 1.f / (float)a.numel());
}

Tensor mean(const Tensor& a, int dim, bool keepdim) {
    const int ndim = (int)a.ndim();
    int d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim)
        throw std::runtime_error("mean: dim out of range");
    const float scale = 1.f / (float)a.shape()[d];
    return mul(sum(a, dim, keepdim), scale);
}

Tensor max(const Tensor& a) {
    // TODO: with backward
    float m = *std::max_element(a.data_ptr(), a.data_ptr() + a.numel());
    return Tensor::full({1}, m);
}

Tensor max(const Tensor& a, int dim, bool keepdim) {
    // TODO
    (void)a; (void)dim; (void)keepdim;
    return Tensor({1});
}

// ---- matmul -----------------------------------------------------------------

// (M x K) @ (K x N) -> (M x N)   or   (B x M x K) @ (B x K x N) -> (B x M x N)
Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& sa = a.shape();
    const auto& sb = b.shape();
    if (sa.size() < 2 || sb.size() < 2)
        throw std::runtime_error("matmul: inputs must be at least 2-D");

    size_t M = sa[sa.size()-2], K = sa[sa.size()-1];
    size_t Kb = sb[sb.size()-2], N = sb[sb.size()-1];
    if (K != Kb)
        throw std::runtime_error("matmul: inner dimension mismatch");

    bool batched = (sa.size() == 3);
    size_t B = batched ? sa[0] : 1;

    std::vector<size_t> out_shape = batched ? std::vector<size_t>{B, M, N}
                                            : std::vector<size_t>{M, N};
    Tensor out(out_shape);
    const float* A = a.data_ptr();
    const float* Bm = b.data_ptr();
    float* C = out.data_ptr();

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t bi = 0; bi < B; ++bi) {
        const float* Ab = A + bi * M * K;
        float* Cb = C + bi * M * N;
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float aik = Ab[i * K + k];
                if (aik == 0.f) continue;
                for (size_t j = 0; j < N; ++j)
                    Cb[i * N + j] += aik * Bm[k * N + j];
            }
        }
    }

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa_p = share(a), sb_p = share(b);
        auto fn = [sa_p, sb_p](const Tensor& g) -> std::vector<Tensor> {
            // grad_a = g @ b^T,  grad_b = a^T @ g
            Tensor grad_a = matmul(g, transpose(*sb_p, -2, -1));
            Tensor grad_b = matmul(transpose(*sa_p, -2, -1), g);
            return {grad_a, grad_b};
        };
        out.set_grad_fn(std::make_shared<Node>("MatmulBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa_p, sb_p}));
    }
    return out;
}

// ---- activations ------------------------------------------------------------

Tensor relu(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] > 0.f ? pa[i] : 0.f;

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = sa->data_ptr()[i] > 0.f ? g.data_ptr()[i] : 0.f;
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("ReLUBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor leaky_relu(const Tensor& a, float neg_slope) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = pa[i] > 0.f ? pa[i] : neg_slope * pa[i];

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, neg_slope](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sa->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i)
                ga.data_ptr()[i] = g.data_ptr()[i] *
                    (sa->data_ptr()[i] > 0.f ? 1.f : neg_slope);
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("LeakyReLUBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor sigmoid(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = 1.f / (1.f + std::exp(-pa[i]));

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sout = share(out);
        auto sa = share(a);
        auto fn = [sout](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sout->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float s = sout->data_ptr()[i];
                ga.data_ptr()[i] = g.data_ptr()[i] * s * (1.f - s);
            }
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("SigmoidBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor tanh(const Tensor& a) {
    Tensor out(a.shape());
    const size_t n = a.numel();
    const float* pa = a.data_ptr(); float* po = out.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) po[i] = std::tanh(pa[i]);

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sout = share(out);
        auto sa = share(a);
        auto fn = [sout](const Tensor& g) -> std::vector<Tensor> {
            Tensor ga(sout->shape());
            const size_t n = g.numel();
            #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float t = sout->data_ptr()[i];
                ga.data_ptr()[i] = g.data_ptr()[i] * (1.f - t * t);
            }
            return {ga};
        };
        out.set_grad_fn(std::make_shared<Node>("TanhBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor softmax(const Tensor& a, int dim) {
    // TODO: numerically stable softmax along `dim` with backward
    (void)dim;
    Tensor out(a.shape());
    return out;
}

Tensor log_softmax(const Tensor& a, int dim) {
    // TODO: log-softmax along `dim` with backward
    (void)dim;
    Tensor out(a.shape());
    return out;
}

// ---- neural network ops -----------------------------------------------------

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: (N, in)  weight: (out, in)  bias: (out,)  -> (N, out)
    // out = input @ weight^T + bias
    Tensor wt = transpose(weight, 0, 1);
    Tensor out = matmul(input, wt);
    // broadcast-add bias
    // TODO: implement properly with broadcasting
    (void)bias;
    return out;
}

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding) {
    // TODO: implement im2col-based conv2d with backward
    (void)input; (void)weight; (void)bias; (void)stride; (void)padding;
    return Tensor({1});
}

Tensor max_pool2d(const Tensor& input, int kernel_size, int stride, int padding) {
    // TODO: implement with index mask for backward
    (void)input; (void)kernel_size; (void)stride; (void)padding;
    return Tensor({1});
}

Tensor avg_pool2d(const Tensor& input, int kernel_size, int stride, int padding) {
    // TODO
    (void)input; (void)kernel_size; (void)stride; (void)padding;
    return Tensor({1});
}

Tensor batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  Tensor& running_mean, Tensor& running_var,
                  bool training, float momentum, float eps) {
    // TODO: compute per-channel mean/var; normalise; scale+shift
    (void)input; (void)weight; (void)bias;
    (void)running_mean; (void)running_var;
    (void)training; (void)momentum; (void)eps;
    return Tensor(input.shape());
}

Tensor dropout(const Tensor& input, float p, bool training) {
    if (!training || p == 0.f) return input;
    Tensor out(input.shape());
    // TODO: generate Bernoulli mask; scale by 1/(1-p)
    (void)p;
    return out;
}

// ---- shape ------------------------------------------------------------------

Tensor reshape(const Tensor& a, std::vector<size_t> new_shape) {
    size_t new_n = 1;
    for (auto d : new_shape) new_n *= d;
    if (new_n != a.numel())
        throw std::runtime_error("reshape: size mismatch");

    Tensor out = Tensor::from_data(new_shape, std::vector<float>(a.data_ptr(),
                                                                   a.data_ptr() + a.numel()));
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa](const Tensor& g) -> std::vector<Tensor> {
            return {reshape(g, sa->shape())};
        };
        out.set_grad_fn(std::make_shared<Node>("ReshapeBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor transpose(const Tensor& a, int dim0, int dim1) {
    int ndim = (int)a.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        throw std::runtime_error("transpose: invalid dims");

    // Build new shape
    std::vector<size_t> new_shape = a.shape();
    std::swap(new_shape[dim0], new_shape[dim1]);

    Tensor out(new_shape);
    // TODO: general N-D transpose via stride permutation (copy for now)
    // For 2-D case:
    if (ndim == 2) {
        size_t R = a.shape()[0], C = a.shape()[1];
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j)
                out.data_ptr()[j * R + i] = a.data_ptr()[i * C + j];
    } else {
        // TODO: N-D
    }

    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        auto fn = [sa, dim0, dim1](const Tensor& g) -> std::vector<Tensor> {
            return {transpose(g, dim0, dim1)};
        };
        out.set_grad_fn(std::make_shared<Node>("TransposeBackward", fn,
            std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor squeeze(const Tensor& a, int dim) {
    std::vector<size_t> new_shape;
    for (int i = 0; i < (int)a.ndim(); ++i) {
        if (dim < 0 || i == dim) {
            if (a.shape()[i] != 1) new_shape.push_back(a.shape()[i]);
        } else {
            new_shape.push_back(a.shape()[i]);
        }
    }
    return reshape(a, new_shape);
}

Tensor unsqueeze(const Tensor& a, int dim) {
    if (dim < 0) dim += (int)a.ndim() + 1;
    std::vector<size_t> new_shape = a.shape();
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(a, new_shape);
}

Tensor flatten(const Tensor& a, int start_dim, int end_dim) {
    int ndim = (int)a.ndim();
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0)   end_dim   += ndim;
    size_t flat = 1;
    std::vector<size_t> new_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i >= start_dim && i <= end_dim) {
            flat *= a.shape()[i];
            if (i == end_dim) new_shape.push_back(flat);
        } else {
            new_shape.push_back(a.shape()[i]);
        }
    }
    return reshape(a, new_shape);
}

Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    // TODO: concatenate along dim with backward
    (void)tensors; (void)dim;
    return Tensor({1});
}

} // namespace ops
} // namespace blade
