#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

// Broadcast-add bias: out[i,j] = mat[i,j] + bias[j]
// mat: (N, out_features)   bias: (out_features,)
static Tensor bias_add(const Tensor& mat, const Tensor& bias) {
    if (mat.ndim() != 2 || bias.ndim() != 1 || mat.shape()[1] != bias.shape()[0])
        throw std::runtime_error("bias_add: shape mismatch");
    const size_t N = mat.shape()[0], out = mat.shape()[1];
    Tensor result(mat.shape());
    const float* pm = mat.data_ptr(); const float* pb = bias.data_ptr();
    float* pr = result.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out; ++j)
            pr[i * out + j] = pm[i * out + j] + pb[j];
    if (any_requires_grad({&mat, &bias})) {
        result.set_requires_grad(true);
        auto sm = share(mat), sb = share(bias);
        result.set_grad_fn(std::make_shared<Node>("BiasAddBackward",
            [sm, sb, N, out](const Tensor& g) -> std::vector<Tensor> {
                // grad_mat = g,  grad_bias = sum(g, dim=0)
                Tensor gm(g.shape());
                const float* pg = g.data_ptr(); float* pgm = gm.data_ptr();
                for (size_t i = 0; i < N * out; ++i) pgm[i] = pg[i];
                Tensor gb = Tensor::zeros({out});
                float* pgb = gb.data_ptr();
                for (size_t i = 0; i < N; ++i)
                    for (size_t j = 0; j < out; ++j) pgb[j] += pg[i * out + j];
                return {gm, gb};
            }, std::vector<std::shared_ptr<Tensor>>{sm, sb}));
    }
    return result;
}

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: (N, in)  weight: (out, in)  bias: (out,)  -> (N, out)
    Tensor out = matmul(input, transpose(weight, 0, 1));
    return bias.shape().empty() ? out : bias_add(out, bias);
}

// ---- stubs (not yet implemented) --------------------------------------------

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding) {
    // TODO: im2col-based conv2d with backward
    (void)input; (void)weight; (void)bias; (void)stride; (void)padding;
    return Tensor({1});
}

Tensor max_pool2d(const Tensor& input, int kernel_size, int stride, int padding) {
    // TODO: index-mask backward
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
    // TODO: per-channel mean/var, normalise, scale+shift
    (void)weight; (void)bias; (void)running_mean; (void)running_var;
    (void)training; (void)momentum; (void)eps;
    return Tensor(input.shape());
}

Tensor dropout(const Tensor& input, float p, bool training) {
    // TODO: Bernoulli mask, scale by 1/(1-p)
    if (!training || p == 0.f) return input;
    (void)p;
    return Tensor(input.shape());
}

} // namespace ops
} // namespace blade
