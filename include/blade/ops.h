#pragma once
#include "blade/tensor.h"
#include <vector>

// All differentiable operations.
// Each function creates a Node with a backward closure when any input requires grad.
// Parallelised with OpenMP where beneficial.

namespace blade {
namespace ops {

// ---- Element-wise arithmetic ------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, float s);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, float s);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, float s);
Tensor div(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, float s);
Tensor neg(const Tensor& a);
Tensor pow(const Tensor& a, float exp);
Tensor sqrt(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor clamp(const Tensor& a, float min, float max);

// ---- Reduction --------------------------------------------------------------
Tensor sum(const Tensor& a);
Tensor sum(const Tensor& a, int dim, bool keepdim = false);
Tensor mean(const Tensor& a);
Tensor mean(const Tensor& a, int dim, bool keepdim = false);
Tensor max(const Tensor& a);
Tensor max(const Tensor& a, int dim, bool keepdim = false);
Tensor argmax(const Tensor& a, int dim);  // returns indices (no grad)
Tensor eq(const Tensor& a, const Tensor& b);  // no grad

// ---- Linear algebra ---------------------------------------------------------
// (M x K) @ (K x N) or batched (B x M x K) @ (B x K x N)
Tensor matmul(const Tensor& a, const Tensor& b);

// ---- Activations ------------------------------------------------------------
Tensor relu(const Tensor& a);
Tensor leaky_relu(const Tensor& a, float neg_slope = 0.01f);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor softmax(const Tensor& a, int dim);
Tensor log_softmax(const Tensor& a, int dim);

// ---- Neural network ops -----------------------------------------------------
// input: (N, in_features)  weight: (out, in)  bias: (out,)
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

// ---- Shape (differentiable) -------------------------------------------------
Tensor reshape(const Tensor& a, std::vector<size_t> shape);
Tensor transpose(const Tensor& a, int dim0, int dim1);
Tensor squeeze(const Tensor& a, int dim = -1);
Tensor unsqueeze(const Tensor& a, int dim);
Tensor flatten(const Tensor& a, int start_dim = 0, int end_dim = -1);

} // namespace ops
} // namespace blade
