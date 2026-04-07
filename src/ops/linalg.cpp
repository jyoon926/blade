#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

// (M×K)@(K×N) or (B×M×K)@(B×K×N) — row-major, sparse-skip optimisation.
Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& sa = a.shape();
    const auto& sb = b.shape();
    if (sa.size() < 2 || sb.size() < 2)
        throw std::runtime_error("matmul: inputs must be at least 2-D");
    const size_t M = sa[sa.size()-2], K = sa[sa.size()-1];
    const size_t Kb = sb[sb.size()-2], N = sb[sb.size()-1];
    if (K != Kb) throw std::runtime_error("matmul: inner dimension mismatch");

    const bool   batched   = (sa.size() == 3);
    const size_t B         = batched ? sa[0] : 1;
    const std::vector<size_t> out_shape = batched
        ? std::vector<size_t>{B, M, N} : std::vector<size_t>{M, N};

    Tensor out(out_shape);
    const float* A = a.data_ptr();
    const float* Bm = b.data_ptr();
    float* C = out.data_ptr();

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t bi = 0; bi < B; ++bi) {
        const float* Ab = A + bi * M * K;
        float*       Cb = C + bi * M * N;
        for (size_t i = 0; i < M; ++i)
            for (size_t k = 0; k < K; ++k) {
                float aik = Ab[i * K + k];
                if (aik == 0.f) continue;
                for (size_t j = 0; j < N; ++j)
                    Cb[i * N + j] += aik * Bm[k * N + j];
            }
    }

    if (any_requires_grad({&a, &b})) {
        out.set_requires_grad(true);
        auto sa_p = share(a), sb_p = share(b);
        out.set_grad_fn(std::make_shared<Node>("MatmulBackward",
            [sa_p, sb_p](const Tensor& g) -> std::vector<Tensor> {
                // grad_a = g @ b^T,  grad_b = a^T @ g
                return {matmul(g, transpose(*sb_p, -2, -1)),
                        matmul(transpose(*sa_p, -2, -1), g)};
            }, std::vector<std::shared_ptr<Tensor>>{sa_p, sb_p}));
    }
    return out;
}

} // namespace ops
} // namespace blade
